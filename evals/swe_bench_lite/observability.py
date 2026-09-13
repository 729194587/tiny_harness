"""Read-only SWE rollout provenance and workspace content observations."""

import hashlib
import os
import stat
import subprocess
from pathlib import Path

from tiny_harness.runtime.events import EventType


def source_metadata() -> dict:
    """Inspect TinyHarness's checkout, never the benchmark task repository."""
    root = Path(__file__).resolve().parents[2]
    try:
        def git(*args):
            return subprocess.run(
                ["git", "-C", str(root), *args], check=True, capture_output=True,
                text=True, timeout=10,
            ).stdout.strip()

        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
        return {"tinyharness_git_commit": commit, "tinyharness_git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"tinyharness_git_commit": None, "tinyharness_git_dirty": None}


def workspace_snapshot(workspace: Path) -> dict:
    """Hash files and link targets without following links or reading special files.

    Include Git tracked and non-ignored untracked files; exclude harness artifacts.
    Equal bytes are equal state regardless of timestamps. Empty directories and
    permissions are not content. A scan error invalidates the entire observation.
    """
    result = {}
    listed = subprocess.run(
        ["git", "-C", str(workspace), "ls-files", "--cached", "--others",
         "--exclude-standard", "-z"],
        check=True, capture_output=True, timeout=10,
    ).stdout
    selected = {os.fsdecode(name) for name in listed.split(b"\0") if name}
    parents = {parent.as_posix() for name in selected for parent in Path(name).parents}

    def scan(directory):
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name in {".git", ".tinyharness"}:
                    continue
                path = Path(entry.path)
                relative = path.relative_to(workspace).as_posix()
                if relative not in selected and relative not in parents:
                    continue
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    if relative not in selected:
                        raise OSError("Tracked directory replaced by a symlink")
                    result[relative] = ("link", hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest())
                elif getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                    raise OSError("Unsupported workspace reparse point")
                elif stat.S_ISDIR(info.st_mode):
                    scan(path)
                elif stat.S_ISREG(info.st_mode):
                    digest = hashlib.sha256()
                    # O_NOFOLLOW prevents a regular-file-to-symlink race on POSIX.
                    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) |
                                 getattr(os, "O_NOFOLLOW", 0))
                    with os.fdopen(fd, "rb") as stream:
                        opened = os.fstat(stream.fileno())
                        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                            raise OSError("Workspace changed during observation")
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                        after = os.fstat(stream.fileno())
                        if (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                            raise OSError("Workspace changed during observation")
                    result[relative] = ("file", digest.hexdigest())
                else:
                    raise OSError("Unsupported workspace entry")

    scan(workspace)
    return result


class WorkspaceMutationLogger:
    """Observe completed tool intervals, including nested task scopes.

    Snapshots never enter model input/results. Scan failures yield unknown;
    downstream event failures retain the normal fatal event-log contract.
    Changes restored within one tool interval and asynchronous writes outside
    these intervals cannot be attributed by boundary snapshots.
    """

    def __init__(self, downstream, workspace: Path):
        self.downstream = downstream
        self.workspace = workspace
        self.pending = {}

    def _snapshot(self):
        try:
            return workspace_snapshot(self.workspace)
        except Exception:
            return None

    def emit(self, event_type, data=None):
        data = dict(data or {})
        key = (data.get("parent_tool_call_id"), data.get("tool_call_id"))
        if event_type == EventType.TOOL_STARTED:
            root_turn = data.get("turn")
            if data.get("parent_tool_call_id") is not None:
                root_turn = next((item[2] for scope, item in self.pending.items()
                                  if scope[0] is None), None)
            self.pending[key] = (self._snapshot(), data, root_turn)
        self.downstream.emit(event_type, data)
        if event_type in {EventType.TOOL_RESULT, EventType.TOOL_HOOK_FAILED}:
            if key in self.pending:
                self._finish(key)
        elif event_type == EventType.RUN_FAILED:
            for pending_key in list(self.pending):
                if pending_key[0] == data.get("parent_tool_call_id"):
                    self._finish(pending_key)

    def _finish(self, key):
        before, identity, root_turn = self.pending.pop(key)
        after = self._snapshot()
        known = before is not None and after is not None
        changes = sum(before.get(path) != after.get(path) for path in before.keys() | after.keys()) if known else None
        self.downstream.emit(EventType.WORKSPACE_OBSERVED, {
            **{name: identity[name] for name in (
                "turn", "tool_call_id", "tool_name", "parent_tool_call_id", "agent_scope"
            ) if name in identity},
            "root_turn": root_turn, "observation_status": "complete" if known else "unknown",
            "workspace_changed": bool(changes) if known else None,
            "changed_paths": changes,
        })
