"""Manual context-compaction tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tiny_harness.runtime.context import CompactionRequest
from tiny_harness.tools.definition import ToolDefinition

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext


def compact(manager: CompactionRequest) -> str:
    """Request compaction after the current tool batch is fully closed."""

    return manager.request()


def build_tools(context: AgentRunContext) -> tuple[ToolDefinition, ...]:
    """Build compact only when this run has a compaction request channel."""

    request = context.compaction_request
    if request is None:
        return ()
    return (
        ToolDefinition(
            name="compact",
            description="Summarize earlier conversation after the current tool batch.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            execute=lambda call, arguments: compact(request, **arguments),
        ),
    )
