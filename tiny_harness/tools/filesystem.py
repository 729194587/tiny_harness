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


MAX_READ_OUTPUT_CHARS = 30_000


def read_file(
    workspace: Path, path: str,
    start_line: int | None = None, end_line: int | None = None,
    start_column: int | None = None,
) -> str:
    """Read a UTF-8 text file inside the workspace."""

    for name, value in (("start_line", start_line), ("end_line", end_line),
                        ("start_column", start_column)):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer or null")
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
    if start_line is not None and end_line is not None and start_line > end_line:
        raise ValueError("start_line must be <= end_line")
    file_path = _resolve_path(workspace, path)
    relative = file_path.relative_to(workspace.resolve()).as_posix()
    # Reserve room for navigation metadata and continuation instructions.
    budget = MAX_READ_OUTPUT_CHARS - len(relative) - 512
    if budget <= 0:
        raise ValueError("File path is too long for bounded read output")
    first = start_line or 1
    parts = []
    used = total = 0
    last = None
    continuation = None
    with file_path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            total = number
            if number < first or (end_line is not None and number > end_line):
                continue
            if continuation is not None:
                continue
            column = (start_column or 1) if number == first else 1
            selected = line[column - 1:]
            if not selected:
                continue
            remaining = budget - used
            # Prefer complete lines; split only a line that cannot fit by itself.
            if len(selected) > remaining and parts:
                continuation = (number, column)
                continue
            piece = selected[:remaining]
            parts.append(piece)
            used += len(piece)
            last = number
            if len(piece) < len(selected):
                continuation = (number, column + len(piece))
    returned = f"{first}-{last}" if last is not None else "none"
    header = f"[lines {returned} of {total} | {relative}]\n\n"
    footer = ""
    if continuation is not None:
        line, column = continuation
        footer = (
            f"\n\n[Read bounded; continue with start_line={line}, start_column={column}"
            + ". Keep the original end_line if set. "
            "start_column is 1-based within the starting line.]"
        )
    return header + "".join(parts) + footer


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


def list_files(
    workspace: Path, path: str = ".", recursive: bool = False,
    pattern: str | None = None,
) -> str:
    """List direct children, or recursively discover files, inside workspace."""

    # Reuse the search tools' glob validation and resolved-path boundary checks.
    from tiny_harness.tools.search import _matching_files, _validate_pattern

    if not isinstance(recursive, bool):
        raise TypeError("recursive must be a boolean")
    if pattern is not None:
        if not isinstance(pattern, str):
            raise TypeError("pattern must be a string or null")
        _validate_pattern(pattern)

    directory = _resolve_path(workspace, path)
    if recursive:
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        include = pattern or "*"
        if len(Path(include).parts) == 1:
            include = "**/" + include
        entries = sorted(
            (entry for entry, _ in _matching_files(workspace, directory, include)),
            key=lambda item: (item.as_posix().casefold(), item.as_posix()),
        )
        return "\n".join(entry.relative_to(workspace.resolve()).as_posix()
                         for entry in entries) or "(no files)"
    entries = sorted(directory.iterdir(), key=lambda item: item.name.casefold())
    if pattern is not None:
        entries = [entry for entry in entries if entry.relative_to(directory).match(pattern)]
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
            "Preferred source/file reading inside the workspace (UTF-8), with ranged navigation; "
            "optional 1-based, inclusive line range. "
            "Output is bounded to 30000 characters and includes total line count, returned range, "
            "and continuation arguments when truncated. Optional start_column is a 1-based "
            "character position in start_line, for continuing an oversized single line.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": ["integer", "null"], "minimum": 1},
                    "end_line": {"type": ["integer", "null"], "minimum": 1},
                    "start_column": {"type": ["integer", "null"], "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            read_file,
            ("path", "start_line", "end_line", "start_column"),
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
            "Inspect a workspace directory by listing its direct children. Set recursive=true to list "
            "files recursively. Optional pattern is a relative glob: '*.py' matches "
            "filenames at every depth with recursive=true; 'src/**/*.py' selects a subtree.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean", "default": False},
                    "pattern": {"type": ["string", "null"]},
                },
                "additionalProperties": False,
            },
            list_files,
            ("path",),
        ),
    )
