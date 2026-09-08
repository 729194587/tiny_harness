"""Adapter for Chat Completions-compatible model APIs."""

import math
from typing import Any

from openai import OpenAI

from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.models.base import (
    ModelErrorKind,
    ModelProviderError,
    ToolChoice,
    ToolChoiceModelProvider,
)

_CONTEXT_ERROR_MARKERS = (
    "context_length_exceeded",
    "prompt_too_long",
    "maximum context length",
    "max_context_window",
    "too many tokens",
)
_SERVER_STATUS_CODES = frozenset({500, 502, 503, 504, 529})


def _status_code(error: Exception) -> int | None:
    value = getattr(error, "status_code", None)
    return value if isinstance(value, int) else None


def _error_code(error: Exception) -> str:
    code = getattr(error, "code", None)
    if isinstance(code, str):
        return code.lower()
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        nested = body.get("error", body)
        if isinstance(nested, dict) and isinstance(nested.get("code"), str):
            return nested["code"].lower()
    return ""


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after")
        if value is None:
            value = headers.get("Retry-After")
        seconds = float(value)
    except (AttributeError, TypeError, ValueError):
        return None
    return seconds if seconds >= 0 and math.isfinite(seconds) else None


def _normalize_error(error: Exception) -> ModelProviderError:
    """Convert SDK-compatible exceptions without exposing response bodies."""

    if isinstance(error, ModelProviderError):
        return error

    status = _status_code(error)
    error_name = type(error).__name__.lower()
    code = _error_code(error)
    text = str(error).lower()
    context_hint = any(
        marker in code or marker in text for marker in _CONTEXT_ERROR_MARKERS
    )

    if context_hint and status in {None, 400, 413, 422}:
        kind = ModelErrorKind.CONTEXT_LENGTH
    elif status == 429 or "ratelimit" in error_name:
        kind = ModelErrorKind.RATE_LIMIT
    elif status in _SERVER_STATUS_CODES or "overloaded" in error_name:
        kind = ModelErrorKind.SERVER_UNAVAILABLE
    elif any(
        marker in error_name
        for marker in ("connection", "timeout", "apitimeout")
    ):
        kind = ModelErrorKind.CONNECTION
    else:
        kind = ModelErrorKind.FATAL

    return ModelProviderError(
        kind,
        status_code=status,
        retry_after_seconds=_retry_after_seconds(error),
    )


class ChatCompletionsProvider(ToolChoiceModelProvider):
    """Normalize a Chat Completions response for the agent loop."""

    supports_tool_choice = True

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: ToolChoice | None = None,
    ) -> ModelResponse:
        """Request one completion and return the fields used by the loop."""

        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools or tool_choice == "none":
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as error:
            raise _normalize_error(error) from error
        if not response.choices:
            raise ModelProviderError(
                ModelErrorKind.FATAL,
                message="Model API returned no choices",
            )

        choice = response.choices[0]
        if choice.finish_reason is None:
            raise ModelProviderError(
                ModelErrorKind.FATAL,
                message="Model API returned no finish reason",
            )
        if choice.finish_reason == "insufficient_system_resource":
            raise ModelProviderError(ModelErrorKind.SERVER_UNAVAILABLE)
        if choice.finish_reason in {"length", "content_filter"}:
            raise ModelProviderError(
                ModelErrorKind.FATAL,
                message=f"Model generation stopped: {choice.finish_reason}",
            )

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
