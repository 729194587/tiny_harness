"""Minimal registration and permission-aware dispatch for tools."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tiny_harness.agent.messages import ToolCall, ToolResult
from tiny_harness.runtime.permissions import (
    DEFAULT_PERMISSION_POLICY,
    PermissionDecision,
    PermissionPolicy,
    PermissionPrompt,
    resolve_permission,
)
from tiny_harness.tools.filesystem import edit_file, list_files, read_file, write_file
from tiny_harness.tools.shell import bash

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
}


def tool_schemas() -> list[dict[str, Any]]:
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
    ]


def dispatch(
    workspace: Path,
    call: ToolCall,
    *,
    permission_policy: PermissionPolicy = DEFAULT_PERMISSION_POLICY,
    permission_prompt: PermissionPrompt | None = None,
) -> ToolResult:
    """Authorize and execute one tool call, converting failures to text."""

    try:
        entry = _TOOL_REGISTRY.get(call.name)
        if entry is None:
            raise ValueError(f"Unknown tool: {call.name}")

        arguments = json.loads(call.arguments_json)
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")

        permission = resolve_permission(
            permission_policy,
            call.name,
            arguments,
            permission_prompt,
        )
        if permission is PermissionDecision.DENY:
            return ToolResult(
                tool_call_id=call.id,
                content=f"Error: Permission denied for tool {call.name}",
            )

        handler = entry[2]
        content = handler(workspace, **arguments)
    except Exception as error:
        content = f"Error: {type(error).__name__}: {error}"

    return ToolResult(tool_call_id=call.id, content=content)
