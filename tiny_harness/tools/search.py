"""Read-only file discovery and text search tools for a workspace."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from tiny_harness.tools.definition import ToolDefinition

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext


DEFAULT_GLOB_RESULTS = 200
DEFAULT_GREP_RESULTS = 100
MAX_RESULTS = 500
MAX_SEARCH_FILE_BYTES = 2 * 1024 * 1024
MAX_MATCH_LINE_CHARS = 500


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


def _validate_pattern(pattern: str) -> None:
    """Reject empty, absolute, or parent-traversing glob patterns."""

    if not pattern:
        raise ValueError("pattern must not be empty")
    parsed = Path(pattern)
    if parsed.is_absolute() or parsed.drive:
        raise ValueError("pattern must be relative to the search path")
    if ".." in parsed.parts:
        raise ValueError("pattern must not contain '..'")


def _validate_max_results(max_results: int) -> None:
    if isinstance(max_results, bool) or not isinstance(max_results, int):
        raise TypeError("max_results must be an integer")
    if not 1 <= max_results <= MAX_RESULTS:
        raise ValueError(f"max_results must be between 1 and {MAX_RESULTS}")


def _matching_files(
    workspace: Path,
    root: Path,
    pattern: str,
) -> Iterator[tuple[Path, Path]]:
    """Yield lexical and resolved file paths that remain inside workspace."""

    workspace = workspace.resolve()
    for candidate in root.glob(pattern):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_relative_to(workspace):
            continue
        try:
            is_file = resolved.is_file()
        except OSError:
            continue
        if is_file:
            yield candidate, resolved


def _matching_file_sort_key(item: tuple[Path, Path]) -> tuple[str, str]:
    """Return a deterministic lexical sort key for matching files."""

    lexical = item[0].as_posix()
    return lexical.casefold(), lexical


def glob_files(
    workspace: Path,
    pattern: str,
    path: str = ".",
    max_results: int = DEFAULT_GLOB_RESULTS,
) -> str:
    """Find files matching a relative glob pattern inside the workspace."""

    _validate_pattern(pattern)
    _validate_max_results(max_results)

    workspace = workspace.resolve()
    root = _resolve_path(workspace, path)
    if not root.is_dir():
        raise ValueError(f"Search path is not a directory: {path}")

    candidates = sorted(
        _matching_files(workspace, root, pattern),
        key=_matching_file_sort_key,
    )

    matches: list[str] = []
    truncated = False
    for candidate, _resolved in candidates:
        relative = candidate.relative_to(workspace).as_posix()
        matches.append(relative)
        if len(matches) > max_results:
            truncated = True
            matches.pop()
            break
    if not matches:
        return "(no matches)"
    if truncated:
        matches.append(f"... truncated after {max_results} results")
    return "\n".join(matches)


def grep_text(
    workspace: Path,
    query: str,
    path: str = ".",
    include: str = "**/*",
    case_sensitive: bool = True,
    max_results: int = DEFAULT_GREP_RESULTS,
) -> str:
    """Search UTF-8 text files for a literal string inside the workspace."""

    if not query:
        raise ValueError("query must not be empty")
    _validate_pattern(include)
    _validate_max_results(max_results)

    workspace = workspace.resolve()
    root = _resolve_path(workspace, path)
    if root.is_file():
        candidates = ((root, root),)
    elif root.is_dir():
        candidates = sorted(
            _matching_files(workspace, root, include),
            key=_matching_file_sort_key,
        )
    else:
        raise ValueError(f"Search path does not exist: {path}")

    needle = query if case_sensitive else query.casefold()
    matches: list[str] = []
    truncated = False

    for candidate, resolved in candidates:
        try:
            if resolved.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
        except OSError:
            continue

        file_matches: list[str] = []
        try:
            with resolved.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    haystack = line if case_sensitive else line.casefold()
                    if needle not in haystack:
                        continue
                    text = line.rstrip("\r\n")
                    if len(text) > MAX_MATCH_LINE_CHARS:
                        text = text[:MAX_MATCH_LINE_CHARS] + "..."
                    relative = candidate.relative_to(workspace).as_posix()
                    file_matches.append(f"{relative}:{line_number}:{text}")
                    if len(matches) + len(file_matches) > max_results:
                        truncated = True
                        file_matches.pop()
                        break
        except (OSError, UnicodeDecodeError):
            continue

        matches.extend(file_matches)
        if truncated:
            break

    if not matches:
        return "(no matches)"
    if truncated:
        matches.append(f"... truncated after {max_results} results")
    return "\n".join(matches)


def build_tools(context: AgentRunContext) -> tuple[ToolDefinition, ...]:
    """Bind read-only search Tool definitions to this run's workspace."""

    workspace = context.workspace
    return (
        ToolDefinition(
            name="glob",
            description=(
                "Find files inside the workspace by a relative glob pattern, "
                "such as '**/*.py' or 'tests/**/test_*.py'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_RESULTS,
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            execute=lambda call, arguments: glob_files(workspace, **arguments),
        ),
        ToolDefinition(
            name="grep",
            description=(
                "Search UTF-8 text files inside the workspace for a literal "
                "string. Use include to restrict files with a glob pattern."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "include": {"type": "string"},
                    "case_sensitive": {"type": "boolean"},
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_RESULTS,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            execute=lambda call, arguments: grep_text(workspace, **arguments),
        ),
    )