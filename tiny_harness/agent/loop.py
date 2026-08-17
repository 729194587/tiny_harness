"""The minimal TinyHarness agent loop."""

from pathlib import Path
from typing import Any

from tiny_harness.models.base import ModelProvider
from tiny_harness.runtime.events import (
    NULL_EVENT_LOGGER,
    EventLogError,
    EventLogger,
    EventType,
)
from tiny_harness.runtime.permissions import (
    DEFAULT_PERMISSION_POLICY,
    PermissionPolicy,
    PermissionPrompt,
)
from tiny_harness.tools.registry import dispatch, tool_schemas


def agent_loop(
    provider: ModelProvider,
    workspace: Path,
    messages: list[dict[str, Any]],
    *,
    max_turns: int = 20,
    permission_policy: PermissionPolicy = DEFAULT_PERMISSION_POLICY,
    permission_prompt: PermissionPrompt | None = None,
    event_logger: EventLogger = NULL_EVENT_LOGGER,
) -> str:
    """Call the model and tools until a final text response is returned."""

    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")

    tools = tool_schemas()
    current_turn = 0
    event_logger.emit(EventType.RUN_STARTED, {"max_turns": max_turns})

    try:
        for current_turn in range(1, max_turns + 1):
            event_logger.emit(
                EventType.MODEL_REQUESTED,
                {"turn": current_turn},
            )
            response = provider.complete(messages, tools)
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

            for call in response.tool_calls:
                result = dispatch(
                    workspace,
                    call,
                    permission_policy=permission_policy,
                    permission_prompt=permission_prompt,
                    event_logger=event_logger,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": result.content,
                    }
                )

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
