"""Contract implemented by model adapters."""

from typing import Any, Protocol

from tiny_harness.agent.messages import ModelResponse


class ModelProvider(Protocol):
    """Synchronous model interface required by the Phase 1 agent loop."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        """Return one normalized model response."""
        ...
