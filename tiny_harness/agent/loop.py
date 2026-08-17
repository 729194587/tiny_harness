"""The minimal TinyHarness agent loop."""

from pathlib import Path
from typing import Any

from tiny_harness.models.base import ModelProvider
from tiny_harness.runtime.context import prepare_context
from tiny_harness.runtime.events import (
    NULL_EVENT_LOGGER,
    EventLogError,
    EventLogger,
    EventType,
)
from tiny_harness.runtime.hooks import ToolHooks
from tiny_harness.runtime.permissions import (
    DEFAULT_PERMISSION_POLICY,
    PermissionPolicy,
    PermissionPrompt,
)
from tiny_harness.runtime.todos import TodoManager
from tiny_harness.tools.registry import dispatch, tool_schemas

TODO_REMINDER_ROUNDS = 3


def agent_loop(
    provider: ModelProvider,
    workspace: Path,
    messages: list[dict[str, Any]],
    *,
    max_turns: int = 20,
    permission_policy: PermissionPolicy = DEFAULT_PERMISSION_POLICY,
    permission_prompt: PermissionPrompt | None = None,
    event_logger: EventLogger = NULL_EVENT_LOGGER,
    max_context_chars: int | None = None,
    tool_hooks: ToolHooks | None = None,
) -> str:
    """Call the model and tools until a final text response is returned."""

    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    if max_context_chars is not None and max_context_chars < 1:
        raise ValueError("max_context_chars must be at least 1")

    tools = tool_schemas()
    todo_manager = TodoManager()
    rounds_since_todo = 0
    current_turn = 0
    run_data = {"max_turns": max_turns}
    if max_context_chars is not None:
        run_data["max_context_chars"] = max_context_chars
    event_logger.emit(EventType.RUN_STARTED, run_data)

    try:
        for current_turn in range(1, max_turns + 1):
            request_messages = messages
            if max_context_chars is not None:
                prepared = prepare_context(messages, tools, max_context_chars)
                request_messages = prepared.messages
                if prepared.dropped_blocks:
                    event_logger.emit(
                        EventType.CONTEXT_TRIMMED,
                        {
                            "turn": current_turn,
                            "before_chars": prepared.before_chars,
                            "after_chars": prepared.after_chars,
                            "dropped_blocks": prepared.dropped_blocks,
                            "dropped_messages": prepared.dropped_messages,
                        },
                    )
            event_logger.emit(
                EventType.MODEL_REQUESTED,
                {"turn": current_turn},
            )
            response = provider.complete(request_messages, tools)
            event_logger.emit(
                EventType.MODEL_RESPONDED,
                {
                    "turn": current_turn,
                    "finish_reason": response.finish_reason,
                    "tool_call_count": len(response.tool_calls),
                    "content_length": len(response.content or ""),
                },
            )

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": response.content,
            }
            if response.reasoning_content is not None:
                assistant_message["reasoning_content"] = response.reasoning_content
            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments_json,
                        },
                    }
                    for call in response.tool_calls
                ]
            messages.append(assistant_message)

            if not response.tool_calls:
                if response.finish_reason != "stop":
                    raise RuntimeError(
                        "Model stopped without a final answer: "
                        f"{response.finish_reason}"
                    )
                answer = response.content or ""
                event_logger.emit(
                    EventType.RUN_FINISHED,
                    {
                        "turns": current_turn,
                        "answer_length": len(answer),
                    },
                )
                return answer

            todo_revision = todo_manager.revision
            for call in response.tool_calls:
                result = dispatch(
                    workspace,
                    call,
                    permission_policy=permission_policy,
                    permission_prompt=permission_prompt,
                    event_logger=event_logger,
                    tool_hooks=tool_hooks,
                    todo_manager=todo_manager,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": result.content,
                    }
                )

            if todo_manager.revision != todo_revision:
                rounds_since_todo = 0
            else:
                rounds_since_todo += 1

            if rounds_since_todo >= TODO_REMINDER_ROUNDS:
                reminder = (
                    "<todo-reminder>\n"
                    "Update your todo list.\n\n"
                    "Current todos:\n"
                    f"{todo_manager.render()}\n"
                    "</todo-reminder>"
                )
                messages[-1]["content"] += f"\n\n{reminder}"
                event_logger.emit(
                    EventType.TODO_REMINDER,
                    {
                        "turn": current_turn,
                        "rounds_since_todo": rounds_since_todo,
                        "todo_count": len(todo_manager.items),
                    },
                )
                rounds_since_todo = 0

        raise RuntimeError(f"Maximum model turns reached: {max_turns}")
    except EventLogError:
        raise
    except Exception as error:
        event_logger.emit(
            EventType.RUN_FAILED,
            {
                "turn": current_turn,
                "error_type": type(error).__name__,
            },
        )
        raise
