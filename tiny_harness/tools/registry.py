"""Tool registration, schema projection, and the shared execution pipeline."""

from __future__ import annotations

import copy
import json
from types import MappingProxyType
from typing import Any

from tiny_harness.agent.messages import ToolCall, ToolResult
from tiny_harness.runtime.events import (
    NULL_EVENT_LOGGER,
    EventLogError,
    EventLogger,
    EventType,
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


def dispatch(
    registry: ToolRegistry,
    call: ToolCall,
    *,
    permission_policy: PermissionPolicy = DEFAULT_PERMISSION_POLICY,
    permission_prompt: PermissionPrompt | None = None,
    event_logger: EventLogger = NULL_EVENT_LOGGER,
    tool_hooks: ToolHooks | None = None,
    permission_rejections: PermissionRejectionTracker | None = None,
) -> ToolResult:
    """Authorize and execute one registered Tool, converting failures to text."""

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
            EventType.TOOL_FINISHED,
            {
                "tool_call_id": call.id,
                "tool_name": call.name,
                "outcome": "error",
                "content_length": len(content),
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
                    "tool_call_id": call.id,
                    "tool_name": call.name,
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
                    "tool_call_id": call.id,
                    "tool_name": call.name,
                    "hook_index": hook_index,
                },
            )
            return ToolResult(
                tool_call_id=call.id,
                content=(
                    "Error: Tool call blocked by PreToolUse hook: "
                    f"{decision.reason}"
                ),
            )

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
                "tool_call_id": call.id,
                "tool_name": call.name,
                "denial_streak": denial_streak,
                "recovery_prompted": recovery_prompted,
            },
        )
        return ToolResult(
            tool_call_id=call.id,
            content=permission_denial_feedback(
                permission_policy,
                call.name,
                arguments,
                repeated=recovery_prompted,
            ),
        )

    if permission_rejections is not None:
        permission_rejections.record_allowed()

    event_logger.emit(
        EventType.TOOL_STARTED,
        {
            "tool_call_id": call.id,
            "tool_name": call.name,
        },
    )
    try:
        content = definition.execute(call, arguments)
        outcome = "returned"
    except EventLogError:
        raise
    except Exception as error:
        content = f"Error: {type(error).__name__}: {error}"
        outcome = "error"

    result = ToolResult(tool_call_id=call.id, content=content)
    event_logger.emit(
        EventType.TOOL_FINISHED,
        {
            "tool_call_id": call.id,
            "tool_name": call.name,
            "outcome": outcome,
            "content_length": len(content),
        },
    )

    if tool_hooks is not None:
        try:
            tool_hooks.run_post(hook_context, result)
        except HookExecutionError as error:
            event_logger.emit(
                EventType.TOOL_HOOK_FAILED,
                {
                    "tool_call_id": call.id,
                    "tool_name": call.name,
                    "stage": error.stage,
                    "hook_index": error.hook_index,
                    "error_type": error.error_type,
                },
            )
            raise

    return result
