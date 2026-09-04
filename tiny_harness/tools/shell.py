"""Shell tool executed with the workspace as its working directory."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from tiny_harness.tools.definition import ToolDefinition

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext


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


def build_tools(context: AgentRunContext) -> tuple[ToolDefinition, ...]:
    """Bind the shell Tool to this run's workspace."""

    workspace = context.workspace
    return (
        ToolDefinition(
            name="bash",
            description="Run a shell command with the workspace as the working directory.",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
            execute=lambda call, arguments: bash(workspace, **arguments),
        ),
    )
