"""Filesystem tools constrained to a workspace directory."""

from pathlib import Path


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
