"""Optional fixed-command test execution capability."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DEFAULT_TEST_TIMEOUT_SECONDS = 120


class TestRunner(Protocol):
    """Execute one runtime-configured canonical test suite."""

    def run(self, workspace: Path) -> str:
        """Return a mechanical exit-code and output report."""
        ...


@dataclass(frozen=True)
class SubprocessTestRunner:
    """Run immutable argv without invoking a shell."""

    argv: tuple[str, ...]
    timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.argv or any(not argument for argument in self.argv):
            raise ValueError("Test runner argv must contain non-empty arguments")
        if self.timeout_seconds <= 0:
            raise ValueError("Test runner timeout must be positive")

    def run(self, workspace: Path) -> str:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            list(self.argv),
            shell=False,
            cwd=workspace.resolve(),
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
        )
        output = (completed.stdout + completed.stderr).strip()
        detail = f"\n{output}" if output else ""
        return f"Exit code: {completed.returncode}{detail}"
