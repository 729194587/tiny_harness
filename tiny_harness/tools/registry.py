"""Tool registration, schema projection, and the shared execution pipeline."""

from __future__ import annotations

import copy
import json
from time import perf_counter
from types import MappingProxyType
from typing import Any

from tiny_harness.agent.messages import ToolCall, ToolResult
from tiny_harness.runtime.events import (
    NULL_EVENT_LOGGER,
    EventLogError,
    EventLogger,
    EventType,
    hash_json,
    hash_text,
)
from tiny_harness.runtime.hooks import HookExecutionError, ToolHookContext, ToolHooks
from tiny_harness.runtime.permissions import (
    DEFAULT_PERMISSION_POLICY,
    PermissionDecision,
    PermissionPolicy,
    PermissionPrompt,
    PermissionRejectionTracker,
    permission_denial_feedback,
    resolve_permission,
)
from tiny_harness.runtime.tool_trace import ToolTraceConfig
from tiny_harness.tools.definition import ToolDefinition


class ToolRegistry:
    """Store the Tool definitions available to one agent run."""

    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        """Register one definition, rejecting duplicate model-facing names."""

        if definition.name in self._definitions:
            raise ValueError(f"Duplicate tool name: {definition.name}")
        self._definitions[definition.name] = definition

    def lookup(self, name: str) -> ToolDefinition:
        """Return the named definition or fail with the dispatch error contract."""

        try:
            return self._definitions[name]
        except KeyError:
            raise ValueError(f"Unknown tool: {name}") from None

    def list(self) -> tuple[ToolDefinition, ...]:
        """List definitions in deterministic registration order."""

        return tuple(self._definitions.values())

    def model_schemas(self) -> list[dict[str, Any]]:
        """Project all definitions into Chat Completions Tool format."""

        return [definition.model_schema() for definition in self._definitions.values()]


def emit_tool_called(
    registry: ToolRegistry,
    call: ToolCall,
    event_logger: EventLogger,
    *,
    turn: int,
) -> None:
    """Record one model-produced ToolCall before any call in its batch runs."""

    if event_logger is NULL_EVENT_LOGGER:
        return
    hashable_arguments: Any = None
    try:
        hashable_arguments = json.loads(call.arguments_json)
        arguments_hash = hash_json(hashable_arguments)
    except (json.JSONDecodeError, TypeError, ValueError):
        arguments_hash = hash_text(call.arguments_json)

    metadata: dict[str, Any] = {}
    try:
        definition = registry.lookup(call.name)
        if isinstance(hashable_arguments, dict):
            candidate = definition.trace_metadata(copy.deepcopy(hashable_arguments))
            if isinstance(candidate, dict):
                metadata = candidate
    except Exception:
        # Observability metadata must never change dispatch behavior.
        pass

    event_logger.emit(
        EventType.TOOL_CALLED,
        {
            **metadata,
            "turn": turn,
            "tool_call_id": call.id,
            "tool_name": call.name,
            "arguments_hash": arguments_hash,
        },
    )


def dispatch(
    registry: ToolRegistry,
    call: ToolCall,
    *,
    permission_policy: PermissionPolicy = DEFAULT_PERMISSION_POLICY,
    permission_prompt: PermissionPrompt | None = None,
    event_logger: EventLogger = NULL_EVENT_LOGGER,
    tool_hooks: ToolHooks | None = None,
    permission_rejections: PermissionRejectionTracker | None = None,
    turn: int = 0,
    tool_called_logged: bool = False,
    tool_trace: ToolTraceConfig = ToolTraceConfig(),
) -> ToolResult:
    """Authorize and execute one registered Tool, converting failures to text."""

    identity = {
        "turn": turn,
        "tool_call_id": call.id,
        "tool_name": call.name,
    }
    if not tool_called_logged:
        emit_tool_called(registry, call, event_logger, turn=turn)

    try:
        definition = registry.lookup(call.name)
        arguments = json.loads(call.arguments_json)
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
    except EventLogError:
        raise
    except Exception as error:
        content = f"Error: {type(error).__name__}: {error}"
        event_logger.emit(
            EventType.TOOL_RESULT,
            {
                **identity,
                "outcome": "error",
                "error_type": type(error).__name__,
                "duration_ms": 0,
                "content_length": len(content),
                "content_hash": hash_text(content),
                **tool_trace.result_metadata(content),
            },
        )
        return ToolResult(tool_call_id=call.id, content=content)

    hook_context = ToolHookContext(
        tool_call_id=call.id,
        tool_name=call.name,
        arguments=MappingProxyType(copy.deepcopy(arguments)),
    )
    if tool_hooks is not None:
        try:
            blocked = tool_hooks.run_pre(hook_context)
        except HookExecutionError as error:
            event_logger.emit(
                EventType.TOOL_HOOK_FAILED,
                {
                    **identity,
                    "stage": error.stage,
                    "hook_index": error.hook_index,
                    "error_type": error.error_type,
                },
            )
            raise
        if blocked is not None:
            hook_index, decision = blocked
            event_logger.emit(
                EventType.TOOL_HOOK_BLOCKED,
                {
                    **identity,
                    "hook_index": hook_index,
                },
            )
            result = ToolResult(
                tool_call_id=call.id,
                content=(
                    "Error: Tool call blocked by PreToolUse hook: "
                    f"{decision.reason}"
                ),
            )
            event_logger.emit(
                EventType.TOOL_RESULT,
                {
                    **identity,
                    "outcome": "hook_blocked",
                    "duration_ms": 0,
                    "content_length": len(result.content),
                    "content_hash": hash_text(result.content),
                    **tool_trace.result_metadata(result.content),
                },
            )
            return result

    permission = resolve_permission(
        permission_policy,
        call.name,
        arguments,
        permission_prompt,
    )
    if permission is PermissionDecision.DENY:
        denial_streak = 1
        recovery_prompted = False
        if permission_rejections is not None:
            denial_streak, recovery_prompted = permission_rejections.record_denial(
                call.name
            )
        event_logger.emit(
            EventType.TOOL_DENIED,
            {
                **identity,
                "denial_streak": denial_streak,
                "recovery_prompted": recovery_prompted,
            },
        )
        result = ToolResult(
            tool_call_id=call.id,
            content=permission_denial_feedback(
                permission_policy,
                call.name,
                arguments,
                repeated=recovery_prompted,
            ),
        )
        event_logger.emit(
            EventType.TOOL_RESULT,
            {
                **identity,
                "outcome": "permission_denied",
                "duration_ms": 0,
                "content_length": len(result.content),
                "content_hash": hash_text(result.content),
                **tool_trace.result_metadata(result.content),
            },
        )
        return result

    if permission_rejections is not None:
        permission_rejections.record_allowed()

    event_logger.emit(
        EventType.TOOL_STARTED,
        identity,
    )
    started_at = perf_counter()
    error_type = None
    try:
        content = definition.execute(call, arguments)
        outcome = "returned"
    except EventLogError:
        raise
    except Exception as error:
        content = f"Error: {type(error).__name__}: {error}"
        outcome = "error"
        error_type = type(error).__name__

    duration_ms = round((perf_counter() - started_at) * 1000, 3)
    result = ToolResult(tool_call_id=call.id, content=content)
    if tool_hooks is not None:
        try:
            tool_hooks.run_post(hook_context, result)
        except HookExecutionError as error:
            event_logger.emit(
                EventType.TOOL_HOOK_FAILED,
                {
                    **identity,
                    "stage": error.stage,
                    "hook_index": error.hook_index,
                    "error_type": error.error_type,
                },
            )
            raise

    event_logger.emit(
        EventType.TOOL_RESULT,
        {
            **identity,
            "outcome": outcome,
            "duration_ms": duration_ms,
            **({"error_type": error_type} if error_type else {}),
            "content_length": len(content),
            "content_hash": hash_text(content),
            **tool_trace.result_metadata(content),
        },
    )

    return result
