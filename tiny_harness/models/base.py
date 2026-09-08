"""Contract implemented by model adapters."""

from enum import Enum
from typing import Any, Literal, Protocol

from tiny_harness.agent.messages import ModelResponse

ToolChoice = Literal["auto", "none"]


class ModelErrorKind(str, Enum):
    """Provider-neutral failure categories understood by the runtime."""

    RATE_LIMIT = "rate_limit"
    SERVER_UNAVAILABLE = "server_unavailable"
    CONNECTION = "connection"
    CONTEXT_LENGTH = "context_length"
    FATAL = "fatal"


class ModelProviderError(RuntimeError):
    """A safe, normalized model-provider failure."""

    def __init__(
        self,
        kind: ModelErrorKind,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
        message: str | None = None,
    ) -> None:
        self.kind = kind
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            message
            or (
                f"Model provider request failed: {kind.value}"
                + (f" (HTTP {status_code})" if status_code is not None else "")
            )
        )


class ModelProvider(Protocol):
    """Synchronous model interface required by the agent loop."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        """Return one normalized model response."""
        ...


class ToolChoiceModelProvider(ModelProvider, Protocol):
    """Optional provider capability for explicit tool-selection policy."""

    supports_tool_choice: bool

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: ToolChoice | None = None,
    ) -> ModelResponse:
        """Return one response using the requested tool-selection policy."""
        ...


def complete_with_tool_choice(
    provider: ModelProvider,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: ToolChoice | None,
) -> ModelResponse:
    """Use explicit tool choice when supported, preserving legacy providers."""

    if getattr(provider, "supports_tool_choice", False):
        return provider.complete(messages, tools, tool_choice=tool_choice)
    return provider.complete(messages, tools)
