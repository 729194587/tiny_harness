"""Minimal messages exchanged by the model, agent loop, and tools."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments_json: str


@dataclass
class ToolResult:
    """Text returned for one model-requested tool invocation."""

    tool_call_id: str
    content: str


@dataclass
class ModelResponse:
    """The subset of a model response needed by the agent loop."""

    content: str | None
    reasoning_content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str


class ModelProtocolError(RuntimeError):
    """A model response cannot be safely committed or executed."""


def validate_tool_call_batch(calls: list[ToolCall]) -> None:
    """Validate the entire batch before any tool can produce side effects."""

    if not isinstance(calls, list) or not calls:
        raise ModelProtocolError("Tool-call batch must be a non-empty list")
    seen: set[str] = set()
    for call in calls:
        if not isinstance(call, ToolCall):
            raise ModelProtocolError("Each tool call must be a ToolCall")
        if not isinstance(call.id, str) or not call.id.strip():
            raise ModelProtocolError("Tool call ID must be a non-empty string")
        if call.id in seen:
            raise ModelProtocolError("Tool call IDs must be unique within a batch")
        seen.add(call.id)
        if not isinstance(call.name, str) or not call.name.strip():
            raise ModelProtocolError("Tool call name must be a non-empty string")
        if not isinstance(call.arguments_json, str):
            raise ModelProtocolError("Tool call arguments must be a JSON string")
        # JSON decoding and handler parameter errors remain ordinary tool errors.


def validate_model_response(response: ModelResponse) -> None:
    """Reject a response that cannot be safely committed or executed."""

    if not isinstance(response.tool_calls, list):
        raise ModelProtocolError("Model tool_calls must be a list")
    if response.finish_reason == "stop":
        if response.tool_calls:
            raise ModelProtocolError(
                "Model response is not executable: stop with tool calls"
            )
        return
    if response.finish_reason == "tool_calls":
        if not response.tool_calls:
            raise ModelProtocolError(
                "Model response is not executable: tool_calls without calls"
            )
        validate_tool_call_batch(response.tool_calls)
        return
    raise ModelProtocolError(
        f"Model response is not executable: {response.finish_reason}"
    )


def assistant_message_from_response(
    response: ModelResponse,
) -> dict[str, Any]:
    """把模型返回的 ModelResponse 转换成标准 assistant message"""

    message: dict[str, Any] = {
        "role": "assistant",
        "content": response.content,
    }
    if response.reasoning_content is not None:
        message["reasoning_content"] = response.reasoning_content
    if response.tool_calls:
        message["tool_calls"] = [
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
    return message
