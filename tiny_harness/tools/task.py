"""Synchronous subagent delegation tool."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from tiny_harness.runtime.events import hash_text
from tiny_harness.tools.definition import ToolDefinition

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext

SubagentRunner = Callable[[str, str], str]


def task(
    runner: SubagentRunner,
    tool_call_id: str,
    prompt: str,
) -> str:
    """Run one delegated prompt and return only the child final text."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    return runner(prompt.strip(), tool_call_id)


def build_tools(context: AgentRunContext) -> tuple[ToolDefinition, ...]:
    """Build the task Tool only when a subagent runner is available."""

    runner = context.subagent_runner
    if runner is None:
        return ()
    return (
        ToolDefinition(
            name="task",
            description="Run a subagent with fresh context and return its final text.",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "minLength": 1},
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
            execute=lambda call, arguments: task(runner, call.id, **arguments),
            trace_metadata=lambda arguments: {
                "prompt_hash": hash_text(str(arguments.get("prompt", ""))),
                "prompt_length": len(str(arguments.get("prompt", ""))),
            },
        ),
    )
