"""Lightweight, provider-independent context token estimation."""

from __future__ import annotations

import copy
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


class CalibratedTokenMeter:
    """Calibrate estimates with observed prompt usage for one history prefix.

    Reuse the offset only while tool schemas are equal and recorded history
    remains a prefix. estimate() does not invalidate state on a mismatch;
    reconcile() and estimate_request() do. Request projection costs are
    accounted for separately from canonical history.
    """

    def __init__(self, heuristic: TokenMeter = DEFAULT_TOKEN_METER) -> None:
        self.heuristic = heuristic
        self.reset()

    def reset(self) -> None:
        """Discard the observed history, schemas, and calibration offset."""

        self._messages = None
        self._tools = None
        self._offset = 0

    def _matches(self, messages, tools) -> bool:
        return (self._messages is not None and tools == self._tools
                and messages[:len(self._messages)] == self._messages)

    def reconcile(self, messages, tools) -> None:
        """Invalidate on committed rewrites, including in-place mutations."""
        if not self._matches(messages, tools):
            self.reset()

    def estimate(self, messages, tools) -> int:
        """Use a matching offset or the heuristic without changing calibration."""

        estimate = self.heuristic.estimate(messages, tools)
        return max(0, estimate + self._offset) if self._matches(messages, tools) else estimate

    def estimate_request(self, messages, request_messages, tools) -> int:
        """Reconcile canonical state, then add the heuristic request-projection delta."""

        self.reconcile(messages, tools)
        return max(0, self.estimate(messages, tools)
                   + self.heuristic.estimate(request_messages, tools)
                   - self.heuristic.estimate(messages, tools))

    def observe(self, messages, request_messages, tools, prompt_tokens) -> None:
        """Record observed usage against copies of history and tool schemas.

        Measure the offset against request_messages. Missing or invalid
        prompt usage resets calibration instead.
        """

        if type(prompt_tokens) is not int or prompt_tokens < 0:
            self.reset()
            return
        self._messages = copy.deepcopy(messages)
        self._tools = copy.deepcopy(tools)
        self._offset = prompt_tokens - self.heuristic.estimate(request_messages, tools)
