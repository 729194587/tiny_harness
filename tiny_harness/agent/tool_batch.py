"""按顺序执行一次完整的模型工具调用批次。"""

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
    """顺序执行全部工具调用并追加结果。

    正常返回时保证 assistant/tool 协议闭合。致命异常保留已有历史和副作用，
    不补造结果或重放调用；AgentSession 会禁止继续提交，直到显式 clear。
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
