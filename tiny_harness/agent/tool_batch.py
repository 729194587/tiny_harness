"""按顺序执行一次完整的模型工具调用批次。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tiny_harness.agent.messages import ToolCall, validate_tool_call_batch
from tiny_harness.runtime.events import EventType
from tiny_harness.tools.registry import dispatch, emit_tool_called

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext

TODO_REMINDER_ROUNDS = 3


def _update_todo_reminder(
    messages: list[dict[str, Any]],
    previous_revision: int,
    context: AgentRunContext,
) -> None:
    """更新 Todo 提醒计数，并在阈值处追加协议安全的提醒。"""

    if context.todo_manager.revision != previous_revision:
        context.rounds_since_todo = 0
    else:
        context.rounds_since_todo += 1

    if context.rounds_since_todo < TODO_REMINDER_ROUNDS:
        return

    reminder = (
        "<todo-reminder>\n"
        "Update your todo list.\n\n"
        "Current todos:\n"
        f"{context.todo_manager.render()}\n"
        "</todo-reminder>"
    )
    messages[-1]["content"] += f"\n\n{reminder}"
    context.event_logger.emit(
        EventType.TODO_REMINDER,
        {
            "turn": context.current_turn,
            "rounds_since_todo": context.rounds_since_todo,
            "todo_count": len(context.todo_manager.items),
        },
    )
    context.rounds_since_todo = 0


def _apply_manual_compaction(
    messages: list[dict[str, Any]],
    previous_revision: int,
    context: AgentRunContext,
) -> None:
    """仅在完整工具批次闭合后处理成功提交的 compact 请求。"""

    if (
        context.compactor is None
        or context.compaction_request is None
        or context.compaction_request.revision == previous_revision
    ):
        return

    prepared = context.compactor.compact_history(
        messages,
        context.todo_manager.render(),
        reason="manual",
    )
    messages[:] = prepared.messages


def execute_tool_batch(
    messages: list[dict[str, Any]],
    calls: list[ToolCall],
    context: AgentRunContext,
) -> None:
    """顺序执行全部工具调用，追加结果后再进行批次边界维护。

    正常返回时保证 assistant/tool 协议闭合。致命异常保留已有历史和副作用，
    不补造结果或重放调用；AgentSession 会禁止继续提交，直到显式 clear。
    """

    validate_tool_call_batch(calls)
    todo_revision = context.todo_manager.revision
    compact_revision = (
        context.compaction_request.revision
        if context.compaction_request is not None
        else 0
    )

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
                "content": result.content,
            }
        )

    _update_todo_reminder(messages, todo_revision, context)
    _apply_manual_compaction(messages, compact_revision, context)
