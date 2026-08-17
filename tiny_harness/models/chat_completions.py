"""Adapter for Chat Completions-compatible model APIs."""

from typing import Any

from openai import OpenAI

from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.models.base import ModelProvider


class ChatCompletionsProvider(ModelProvider):
    """Normalize a Chat Completions response for the agent loop."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._client = client or OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        """Request one completion and return the fields used by the loop."""

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
        )
        if not response.choices:
            raise RuntimeError("Model API returned no choices")

        choice = response.choices[0]
        if choice.finish_reason is None:
            raise RuntimeError("Model API returned no finish reason")

        message = choice.message
        tool_calls = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments_json=call.function.arguments,
            )
            for call in (message.tool_calls or [])
        ]
        return ModelResponse(
            content=message.content,
            reasoning_content=getattr(message, "reasoning_content", None),
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
        )
