"""Injectable execution capability used by the shell tool."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol

DEFAULT_SHELL_TIMEOUT_SECONDS = 120


class ShellRunner(Protocol):
    """Execute one model-requested shell command for a workspace."""

    def run(self, workspace: Path, command: str) -> str:
        """Return a mechanical exit-code and combined output report."""
        ...


class SubprocessShellRunner:
    """Preserve TinyHarness' original host-shell behavior."""

    def __init__(self, timeout_seconds: float = DEFAULT_SHELL_TIMEOUT_SECONDS) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Shell runner timeout must be positive")
        self.timeout_seconds = timeout_seconds

    def run(self, workspace: Path, command: str) -> str:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=workspace.resolve(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
        )
        output = (completed.stdout + completed.stderr).strip()
        detail = f"\n{output}" if output else ""
        return f"Exit code: {completed.returncode}{detail}"


DEFAULT_SHELL_RUNNER = SubprocessShellRunner()
