"""Shell tool bound to an injectable runtime execution capability."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tiny_harness.runtime.shell_runner import DEFAULT_SHELL_RUNNER, ShellRunner
from tiny_harness.runtime.tool_trace import ToolTraceConfig
from tiny_harness.tools.definition import ToolDefinition

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext


def bash(workspace: Path, command: str, runner: ShellRunner = DEFAULT_SHELL_RUNNER) -> str:
    """Run a shell command and return its combined text output."""
    return runner.run(workspace, command)


def build_tools(context: AgentRunContext) -> tuple[ToolDefinition, ...]:
    """Bind the shell Tool to this run's workspace."""

    workspace = context.workspace
    runner = getattr(context, "shell_runner", DEFAULT_SHELL_RUNNER)
    return (
        ToolDefinition(
            name="bash",
            description=(
                "General-purpose shell for tests, builds, Git, scripts, and operations not "
                "covered by dedicated tools, with the workspace as the working directory. "
                "For ordinary repository listing, search, and file reading, prefer "
                "list_files, glob, grep, search_code, and read_file. "
                "Shell commands may require interactive approval."
            ),
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
            trace_metadata=lambda arguments: (
                {"command": arguments["command"]}
                if getattr(context, "tool_trace", ToolTraceConfig()).enabled
                and isinstance(arguments.get("command"), str) else {}
            ),
            execute=lambda call, arguments: bash(
                workspace, runner=runner, **arguments
            ),
        ),
    )
