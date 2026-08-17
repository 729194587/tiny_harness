"""The minimal Phase 1 agent loop."""

from pathlib import Path
from typing import Any

from tiny_harness.models.base import ModelProvider
from tiny_harness.tools.registry import dispatch, tool_schemas


def agent_loop(
    provider: ModelProvider,
    workspace: Path,
    messages: list[dict[str, Any]],
    *,
    max_turns: int = 20,
) -> str:
    """Call the model and tools until a final text response is returned."""

    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")

    tools = tool_schemas()
    for _ in range(max_turns):
        response = provider.complete(messages, tools)

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
                    f"Model stopped without a final answer: {response.finish_reason}"
                )
            return response.content or ""

        for call in response.tool_calls:
            result = dispatch(workspace, call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                }
            )

    raise RuntimeError(f"Maximum model turns reached: {max_turns}")
