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


def read_file(
    workspace: Path, path: str,
    start_line: int | None = None, end_line: int | None = None,
) -> str:
    """Read a UTF-8 text file inside the workspace."""

    for name, value in (("start_line", start_line), ("end_line", end_line)):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer or null")
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
    if start_line is not None and end_line is not None and start_line > end_line:
        raise ValueError("start_line must be <= end_line")
    file_path = _resolve_path(workspace, path)
    if start_line is None and end_line is None:
        return file_path.read_text(encoding="utf-8")
    with file_path.open(encoding="utf-8") as stream:
        lines = []
        for number, line in enumerate(stream, 1):
            if end_line is not None and number > end_line:
                break
            if number >= (start_line or 1):
                lines.append(line)
        return "".join(lines)


def write_file(workspace: Path, path: str, content: str) -> str:
    """Write a UTF-8 text file inside the workspace."""

    file_path = _resolve_path(workspace, path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(content)
    byte_count = len(content.encode("utf-8"))
    return f"Wrote {byte_count} bytes to {path}"


def _file_newline(content: str) -> str | None:
    if "\r\n" in content:
        return "\r\n"
    if "\n" in content:
        return "\n"
    if "\r" in content:
        return "\r"
    return None


def _normalize_newlines(content: str, newline: str | None) -> str:
    if newline is None:
        return content
    return content.replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)


def edit_file(workspace: Path, path: str, old_text: str, new_text: str) -> str:
    """Replace text only when it occurs exactly once in a workspace file."""

    if not old_text:
        raise ValueError("old_text must not be empty")

    file_path = _resolve_path(workspace, path)
    with file_path.open("r", encoding="utf-8", newline="") as stream:
        content = stream.read()
    newline = _file_newline(content)
    normalized_old_text = _normalize_newlines(old_text, newline)
    normalized_new_text = _normalize_newlines(new_text, newline)
    occurrence_count = content.count(normalized_old_text)
    if occurrence_count == 0:
        raise ValueError(f"Text not found in {path}")
    if occurrence_count > 1:
        raise ValueError(
            f"Text is not unique in {path}: found {occurrence_count} occurrences"
        )

    updated = content.replace(normalized_old_text, normalized_new_text, 1)
    with file_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(updated)
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
        trace_fields: tuple[str, ...],
    ) -> ToolDefinition:
        return ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            execute=lambda call, arguments: handler(workspace, **arguments),
            trace_metadata=lambda arguments: {
                field: arguments.get(field, ".")
                for field in trace_fields
                if field in arguments or (name == "list_files" and field == "path")
            },
        )

    return (
        definition(
            "read_file",
            "Read UTF-8 text inside the workspace; optional 1-based, inclusive line range.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": ["integer", "null"], "minimum": 1},
                    "end_line": {"type": ["integer", "null"], "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            read_file,
            ("path", "start_line", "end_line"),
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
            ("path",),
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
            ("path",),
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
            ("path",),
        ),
    )
