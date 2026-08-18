"""Shell tool executed with the workspace as its working directory."""

import subprocess
from pathlib import Path


def bash(workspace: Path, command: str) -> str:
    """Run a shell command and return its combined text output."""

    completed = subprocess.run(
        command,
        shell=True,
        cwd=workspace.resolve(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    output = (completed.stdout + completed.stderr).strip()

    detail = f"\n{output}" if output else ""
    return f"Exit code: {completed.returncode}{detail}"
