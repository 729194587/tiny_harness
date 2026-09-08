"""Lightweight, provider-independent context token estimation."""

from __future__ import annotations

import json
import math
from typing import Any, Protocol


class TokenMeter(Protocol):
    """Estimate tokens for a complete model request context."""

    def estimate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> int:
        """Return an estimated token count for messages and tool schemas."""


class HeuristicTokenMeter:
    """Estimate one token per four compact-JSON characters.

    This deliberately avoids binding context management to a model tokenizer.
    Provider-specific meters or calibrated implementations can replace it through
    the :class:`TokenMeter` protocol.
    """

    def estimate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> int:
        try:
            serialized = json.dumps(
                {"messages": messages, "tools": tools},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Context must be JSON serializable") from error
        return math.ceil(len(serialized) / 4)


DEFAULT_TOKEN_METER: TokenMeter = HeuristicTokenMeter()
