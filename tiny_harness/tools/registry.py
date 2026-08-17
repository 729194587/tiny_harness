"""Minimal registration and permission-aware dispatch for tools."""

import copy
import json
from collections.abc import Callable
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
    resolve_permission,
)
from tiny_harness.runtime.todos import TodoManager
from tiny_harness.tools.filesystem import edit_file, list_files, read_file, write_file
from tiny_harness.tools.shell import bash
from tiny_harness.tools.task import SubagentRunner, task
from tiny_harness.tools.todo import todo_write

ToolEntry = tuple[str, dict[str, Any], Callable[..., str]]


_TOOL_REGISTRY: dict[str, ToolEntry] = {
    "read_file": (
        "Read a UTF-8 text file inside the workspace.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        read_file,
    ),
    "write_file": (
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
        write_file,
    ),
    "edit_file": (
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
        edit_file,
    ),
    "list_files": (
        "List the direct children of a directory inside the workspace.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "additionalProperties": False,
        },
        list_files,
    ),
    "bash": (
        "Run a shell command with the workspace as the working directory.",
        {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
        bash,
    ),
    "todo_write": (
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
        todo_write,
    ),
    "task": (
        "Run a subagent with fresh context and return its final text.",
        {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "minLength": 1},
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
        task,
    ),
}


def tool_schemas(*, include_task: bool = True) -> list[dict[str, Any]]:
    """Return all registered tools in Chat Completions function-tool format."""

    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        for name, (description, parameters, _) in _TOOL_REGISTRY.items()
        if include_task or name != "task"
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
) -> ToolResult:
    """Authorize and execute one tool call, converting failures to text."""

    try:
        entry = _TOOL_REGISTRY.get(call.name)
        if entry is None or (call.name == "task" and subagent_runner is None):
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
        event_logger.emit(
            EventType.TOOL_DENIED,
            {
                "tool_call_id": call.id,
                "tool_name": call.name,
            },
        )
        return ToolResult(
            tool_call_id=call.id,
            content=f"Error: Permission denied for tool {call.name}",
        )

    event_logger.emit(
        EventType.TOOL_STARTED,
        {
            "tool_call_id": call.id,
            "tool_name": call.name,
        },
    )
    handler = entry[2]
    try:
        if call.name == "todo_write":
            if todo_manager is None:
                raise RuntimeError("todo_write requires a TodoManager")
            content = handler(todo_manager, **arguments)
        elif call.name == "task":
            content = handler(subagent_runner, call.id, **arguments)
        else:
            content = handler(workspace, **arguments)
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
