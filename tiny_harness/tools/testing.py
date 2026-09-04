"""Tool adapter for the configured repository test runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tiny_harness.tools.definition import ToolDefinition

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext


def build_tools(context: AgentRunContext) -> tuple[ToolDefinition, ...]:
    """Build run_tests only when the run has a configured TestRunner."""

    runner = context.test_runner
    if runner is None:
        return ()
    workspace = context.workspace

    def execute(call, arguments):
        if arguments:
            raise TypeError("run_tests does not accept arguments")
        return runner.run(workspace)

    return (
        ToolDefinition(
            name="run_tests",
            description="Run the configured repository test suite.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            execute=execute,
        ),
    )
