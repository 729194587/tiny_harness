"""Filesystem tools constrained to a workspace directory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tiny_harness.tools.definition import ToolDefinition

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext


def _resolve_path(workspace: Path, path: str) -> Path:
    """Resolve *path* and reject targets outside *workspace*."""

    workspace = workspace.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = candidate.resolve()

    if not candidate.is_relative_to(workspace):
        raise ValueError(f"Path escapes workspace: {path}")
    return candidate


def read_file(workspace: Path, path: str) -> str:
    """Read a UTF-8 text file inside the workspace."""

    return _resolve_path(workspace, path).read_text(encoding="utf-8")


def write_file(workspace: Path, path: str, content: str) -> str:
    """Write a UTF-8 text file inside the workspace."""

    file_path = _resolve_path(workspace, path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    byte_count = len(content.encode("utf-8"))
    return f"Wrote {byte_count} bytes to {path}"


def edit_file(workspace: Path, path: str, old_text: str, new_text: str) -> str:
    """Replace text only when it occurs exactly once in a workspace file."""

    if not old_text:
        raise ValueError("old_text must not be empty")

    file_path = _resolve_path(workspace, path)
    content = file_path.read_text(encoding="utf-8")
    occurrence_count = content.count(old_text)
    if occurrence_count == 0:
        raise ValueError(f"Text not found in {path}")
    if occurrence_count > 1:
        raise ValueError(
            f"Text is not unique in {path}: found {occurrence_count} occurrences"
        )

    file_path.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Edited {path}"


def list_files(workspace: Path, path: str = ".") -> str:
    """List the direct children of a workspace directory."""

    directory = _resolve_path(workspace, path)
    entries = sorted(directory.iterdir(), key=lambda item: item.name.casefold())
    if not entries:
        return "(no files)"

    workspace = workspace.resolve()
    lines = []
    for entry in entries:
        relative_path = entry.relative_to(workspace).as_posix()
        lines.append(f"{relative_path}/" if entry.is_dir() else relative_path)
    return "\n".join(lines)


def build_tools(context: AgentRunContext) -> tuple[ToolDefinition, ...]:
    """Bind all filesystem Tool definitions to this run's workspace."""

    workspace = context.workspace

    def definition(
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., str],
    ) -> ToolDefinition:
        return ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            execute=lambda call, arguments: handler(workspace, **arguments),
        )

    return (
        definition(
            "read_file",
            "Read a UTF-8 text file inside the workspace.",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            read_file,
        ),
        definition(
            "write_file",
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
        definition(
            "edit_file",
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
        definition(
            "list_files",
            "List the direct children of a directory inside the workspace.",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
            list_files,
        ),
    )
