"""Contract implemented by model adapters."""

from enum import Enum
from typing import Any, Protocol

from tiny_harness.agent.messages import ModelResponse


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
