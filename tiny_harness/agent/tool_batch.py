"""Sequential tool-batch execution and result history updates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tiny_harness.agent.messages import ToolCall, validate_tool_call_batch
from tiny_harness.runtime.context import retain_tool_result
from tiny_harness.tools.registry import dispatch, emit_tool_called

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext


def execute_tool_batch(
    messages: list[dict[str, Any]],
    calls: list[ToolCall],
    context: AgentRunContext,
) -> None:
    """Execute calls sequentially and append results in model order.

    With the matching assistant message already appended, normal return
    closes its tool calls. All TOOL_CALLED events precede dispatch.
    Fatal failures retain committed history and side effects; no missing
    results are fabricated and calls are not replayed. AgentSession then
    requires clear() before further submissions.
    """

    validate_tool_call_batch(calls)

    for call in calls:
        emit_tool_called(
            context.tool_registry,
            call,
            context.event_logger,
            turn=context.current_turn,
        )

    for call in calls:
        result = dispatch(
            context.tool_registry,
            call,
            permission_policy=context.permission_policy,
            permission_prompt=context.permission_prompt,
            event_logger=context.event_logger,
            tool_hooks=context.tool_hooks,
            permission_rejections=context.permission_rejections,
            turn=context.current_turn,
            tool_called_logged=True,
            tool_trace=context.tool_trace,
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "content": retain_tool_result(
                    context.workspace, call, result.content,
                    turn=context.current_turn, event_logger=context.event_logger,
                    config=context.compaction_config,
                ),
            }
        )
