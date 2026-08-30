"""Minimal registration and permission-aware dispatch for tools."""

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tiny_harness.agent.messages import ToolCall, ToolResult
from tiny_harness.runtime.events import (
    NULL_EVENT_LOGGER,
    EventLogError,
    EventLogger,
    EventType,
)
from tiny_harness.runtime.context import CompactionRequest
from tiny_harness.runtime.hooks import (
    HookExecutionError,
    ToolHookContext,
    ToolHooks,
)
from tiny_harness.runtime.permissions import (
    DEFAULT_PERMISSION_POLICY,
    PermissionDecision,
    PermissionPolicy,
    PermissionPrompt,
    PermissionRejectionTracker,
    permission_denial_feedback,
    resolve_permission,
)
from tiny_harness.runtime.skills import SkillCatalog
from tiny_harness.runtime.todos import TodoManager
from tiny_harness.tools.filesystem import edit_file, list_files, read_file, write_file
from tiny_harness.tools.compact import compact
from tiny_harness.tools.shell import bash
from tiny_harness.tools.skill import load_skill
from tiny_harness.tools.task import SubagentRunner, task
from tiny_harness.tools.todo import todo_write


@dataclass(frozen=True)
class ToolRuntime:
    """Run-scoped dependencies available to every tool adapter."""

    workspace: Path
    todo_manager: TodoManager | None
    subagent_runner: SubagentRunner | None
    skill_catalog: SkillCatalog | None
    compaction_request: CompactionRequest | None


ToolExecutor = Callable[
    [ToolRuntime, ToolCall, dict[str, Any]],
    str,
]
ToolAvailability = Callable[[ToolRuntime], bool]


def _always_available(runtime: ToolRuntime) -> bool:
    return True


@dataclass(frozen=True)
class ToolAdapter:
    """Schema plus one uniform runtime-aware execution adapter."""

    description: str
    parameters: dict[str, Any]
    execute: ToolExecutor
    available: ToolAvailability = _always_available


def _workspace_tool(handler: Callable[..., str]) -> ToolExecutor:
    def execute(
        runtime: ToolRuntime,
        call: ToolCall,
        arguments: dict[str, Any],
    ) -> str:
        return handler(runtime.workspace, **arguments)

    return execute


def _execute_todo(
    runtime: ToolRuntime,
    call: ToolCall,
    arguments: dict[str, Any],
) -> str:
    if runtime.todo_manager is None:
        raise RuntimeError("todo_write requires a TodoManager")
    return todo_write(runtime.todo_manager, **arguments)


def _execute_task(
    runtime: ToolRuntime,
    call: ToolCall,
    arguments: dict[str, Any],
) -> str:
    if runtime.subagent_runner is None:
        raise RuntimeError("task requires a SubagentRunner")
    return task(runtime.subagent_runner, call.id, **arguments)


def _execute_skill(
    runtime: ToolRuntime,
    call: ToolCall,
    arguments: dict[str, Any],
) -> str:
    if runtime.skill_catalog is None:
        raise RuntimeError("load_skill requires a SkillCatalog")
    return load_skill(runtime.skill_catalog, **arguments)


def _execute_compact(
    runtime: ToolRuntime,
    call: ToolCall,
    arguments: dict[str, Any],
) -> str:
    if runtime.compaction_request is None:
        raise RuntimeError("compact requires a CompactionRequest")
    return compact(runtime.compaction_request, **arguments)


def _has_subagent(runtime: ToolRuntime) -> bool:
    return runtime.subagent_runner is not None


def _has_skill_catalog(runtime: ToolRuntime) -> bool:
    return runtime.skill_catalog is not None


def _has_compaction_request(runtime: ToolRuntime) -> bool:
    return runtime.compaction_request is not None


_TOOL_REGISTRY: dict[str, ToolAdapter] = {
    "read_file": ToolAdapter(
        "Read a UTF-8 text file inside the workspace.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        _workspace_tool(read_file),
    ),
    "write_file": ToolAdapter(
        "Write UTF-8 text to a file inside the workspace.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        _workspace_tool(write_file),
    ),
    "edit_file": ToolAdapter(
        "Replace exact text in a workspace file when it occurs exactly once.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
        _workspace_tool(edit_file),
    ),
    "list_files": ToolAdapter(
        "List the direct children of a directory inside the workspace.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "additionalProperties": False,
        },
        _workspace_tool(list_files),
    ),
    "bash": ToolAdapter(
        "Run a shell command with the workspace as the working directory.",
        {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
        _workspace_tool(bash),
    ),
    "todo_write": ToolAdapter(
        "Create and manage a task list for the current coding run.",
        {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "minLength": 1},
                            "status": {
                                "type": "string",
                                "enum": [
                                    "pending",
                                    "in_progress",
                                    "completed",
                                ],
                            },
                        },
                        "required": ["content", "status"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["todos"],
            "additionalProperties": False,
        },
        _execute_todo,
    ),
    "task": ToolAdapter(
        "Run a subagent with fresh context and return its final text.",
        {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "minLength": 1},
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
        _execute_task,
        _has_subagent,
    ),
    "load_skill": ToolAdapter(
        "Load one workspace Skill by its exact catalog name.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        _execute_skill,
        _has_skill_catalog,
    ),
    "compact": ToolAdapter(
        "Summarize earlier conversation after the current tool batch.",
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        _execute_compact,
        _has_compaction_request,
    ),
}


def tool_schemas(
    *,
    include_task: bool = True,
    include_skill: bool = False,
    include_compact: bool = False,
) -> list[dict[str, Any]]:
    """Return all registered tools in Chat Completions function-tool format."""

    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": adapter.description,
                "parameters": adapter.parameters,
            },
        }
        for name, adapter in _TOOL_REGISTRY.items()
        if (include_task or name != "task")
        and (include_skill or name != "load_skill")
        and (include_compact or name != "compact")
    ]


def dispatch(
    workspace: Path,
    call: ToolCall,
    *,
    permission_policy: PermissionPolicy = DEFAULT_PERMISSION_POLICY,
    permission_prompt: PermissionPrompt | None = None,
    event_logger: EventLogger = NULL_EVENT_LOGGER,
    tool_hooks: ToolHooks | None = None,
    todo_manager: TodoManager | None = None,
    subagent_runner: SubagentRunner | None = None,
    skill_catalog: SkillCatalog | None = None,
    compaction_request: CompactionRequest | None = None,
    permission_rejections: PermissionRejectionTracker | None = None,
) -> ToolResult:
    """Authorize and execute one tool call, converting failures to text."""

    runtime = ToolRuntime(
        workspace=workspace,
        todo_manager=todo_manager,
        subagent_runner=subagent_runner,
        skill_catalog=skill_catalog,
        compaction_request=compaction_request,
    )
    try:
        entry = _TOOL_REGISTRY.get(call.name)
        if entry is None or not entry.available(runtime):
            raise ValueError(f"Unknown tool: {call.name}")

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
                    "Error: Tool call blocked by PreToolUse hook: " f"{decision.reason}"
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
            denial_streak, recovery_prompted = (
                permission_rejections.record_denial(call.name)
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
        content = entry.execute(runtime, call, arguments)
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
