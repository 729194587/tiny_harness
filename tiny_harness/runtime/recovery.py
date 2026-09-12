"""Bounded retry and backoff for physical model-provider attempts."""

import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from tiny_harness.agent.messages import ModelResponse
from tiny_harness.models.base import (
    ModelErrorKind,
    ModelProvider,
    ModelProviderError,
    ToolChoice,
    complete_with_tool_choice,
)
from tiny_harness.runtime.events import NULL_EVENT_LOGGER, EventLogger, EventType
from tiny_harness.context.attribution import request_attribution
from tiny_harness.context.token_meter import DEFAULT_TOKEN_METER, TokenMeter

TRANSIENT_ERROR_KINDS = frozenset(
    {
        ModelErrorKind.RATE_LIMIT,
        ModelErrorKind.SERVER_UNAVAILABLE,
        ModelErrorKind.CONNECTION,
    }
)


@dataclass(frozen=True)
class RecoveryPolicy:
    """Retry limits and delay calculation for one agent run."""

    max_retries: int = 2
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 8.0
    jitter_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be at least 0")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must be at least 0")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError(
                "max_delay_seconds must be at least base_delay_seconds"
            )
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1")


@dataclass
class RecoveryState:
    """Counters shared across transient and reactive attempts of one request."""

    attempt: int = 0
    transient_retries_used: int = 0
    reactive_compact_used: bool = False


class RecoveryExecutor:
    """Execute provider attempts without knowing any context-compaction policy."""

    def __init__(
        self,
        policy: RecoveryPolicy = RecoveryPolicy(),
        *,
        event_logger: EventLogger = NULL_EVENT_LOGGER,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self.policy = policy
        self.event_logger = event_logger
        self._sleep = sleep
        self._random_value = random_value

    def _delay(self, error: ModelProviderError, retry_index: int) -> float:
        if error.retry_after_seconds is not None:
            return min(error.retry_after_seconds, self.policy.max_delay_seconds)
        base = min(
            self.policy.base_delay_seconds * (2**retry_index),
            self.policy.max_delay_seconds,
        )
        return min(
            base + base * self.policy.jitter_ratio * self._random_value(),
            self.policy.max_delay_seconds,
        )

    @staticmethod
    def _failure_data(
        error: Exception,
        *,
        purpose: str,
        turn: int,
        attempt: int,
        will_retry: bool,
    ) -> dict[str, Any]:
        if isinstance(error, ModelProviderError):
            error_kind = error.kind.value
            status_code = error.status_code
        else:
            error_kind = "unclassified"
            status_code = None
        return {
            "purpose": purpose,
            "turn": turn,
            "attempt": attempt,
            "error_kind": error_kind,
            "status_code": status_code,
            "will_retry": will_retry,
        }

    def complete(
        self,
        provider: ModelProvider,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        purpose: str,
        turn: int,
        state: RecoveryState,
        context_recovery_available: bool = False,
        request_metadata: Mapping[str, Any] | None = None,
        finalization: bool = False,
        tool_choice: ToolChoice | None = None,
        token_meter: TokenMeter = DEFAULT_TOKEN_METER,
    ) -> ModelResponse:
        """Return a response or raise after bounded transient retries."""

        while True:
            state.attempt += 1
            attribution = request_attribution(messages, tools, token_meter)
            context_tokens = (request_metadata or {}).get("context_tokens")
            if isinstance(context_tokens, int):
                attribution["calibration_adjustment_tokens"] = (
                    context_tokens - attribution["estimated_tokens"]
                )
            self.event_logger.emit(
                EventType.MODEL_REQUESTED,
                {
                    "purpose": purpose,
                    "turn": turn,
                    "attempt": state.attempt,
                    **({"finalization": True} if finalization else {}),
                    **dict(request_metadata or {}),
                    "context_attribution": attribution,
                },
            )
            try:
                response = complete_with_tool_choice(
                    provider,
                    messages,
                    tools,
                    tool_choice,
                )
            except Exception as error:
                transient = (
                    isinstance(error, ModelProviderError)
                    and error.kind in TRANSIENT_ERROR_KINDS
                )
                can_retry = (
                    transient
                    and state.transient_retries_used < self.policy.max_retries
                )
                context_will_retry = (
                    isinstance(error, ModelProviderError)
                    and error.kind is ModelErrorKind.CONTEXT_LENGTH
                    and context_recovery_available
                    and not state.reactive_compact_used
                )
                self.event_logger.emit(
                    EventType.MODEL_REQUEST_FAILED,
                    self._failure_data(
                        error,
                        purpose=purpose,
                        turn=turn,
                        attempt=state.attempt,
                        will_retry=can_retry or context_will_retry,
                    ),
                )
                if not can_retry:
                    if transient:
                        self.event_logger.emit(
                            EventType.MODEL_RETRY_EXHAUSTED,
                            {
                                "purpose": purpose,
                                "turn": turn,
                                "attempt": state.attempt,
                                "error_kind": error.kind.value,
                                "retries_used": state.transient_retries_used,
                            },
                        )
                    raise

                retry_index = state.transient_retries_used
                state.transient_retries_used += 1
                delay = self._delay(error, retry_index)
                self.event_logger.emit(
                    EventType.MODEL_RETRY_SCHEDULED,
                    {
                        "purpose": purpose,
                        "turn": turn,
                        "attempt": state.attempt,
                        "retry_number": state.transient_retries_used,
                        "delay_ms": round(delay * 1000),
                        "error_kind": error.kind.value,
                    },
                )
                self._sleep(delay)
                continue

            self.event_logger.emit(
                EventType.MODEL_RESPONDED,
                {
                    "purpose": purpose,
                    "turn": turn,
                    "attempt": state.attempt,
                    "finish_reason": response.finish_reason,
                    **{name: getattr(response, name) for name in
                       ("prompt_tokens", "completion_tokens", "total_tokens")
                       if getattr(response, name) is not None},
                    "tool_call_count": len(response.tool_calls),
                    "content_length": len(response.content or ""),
                    **({"finalization": True} if finalization else {}),
                },
            )
            return response
