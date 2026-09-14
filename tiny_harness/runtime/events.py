"""Minimal synchronous event logging for TinyHarness runs."""

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4


class EventType(str, Enum):
    """Lifecycle events recorded by the harness."""

    RUN_STARTED = "run_started"
    CONTEXT_PREPARED = "context_prepared"
    CONTEXT_TRIMMED = "context_trimmed"
    CONTEXT_COMPACTED = "context_compacted"
    CONTEXT_SUMMARY_REQUESTED = "context_summary_requested"
    CONTEXT_SUMMARY_RESPONDED = "context_summary_responded"
    MODEL_REQUESTED = "model_requested"
    MODEL_RESPONDED = "model_responded"
    MODEL_REQUEST_FAILED = "model_request_failed"
    MODEL_RETRY_SCHEDULED = "model_retry_scheduled"
    MODEL_RETRY_EXHAUSTED = "model_retry_exhausted"
    MEMORY_SELECTED = "memory_selected"
    MEMORY_EXTRACTION_REQUESTED = "memory_extraction_requested"
    MEMORY_EXTRACTION_COMPLETED = "memory_extraction_completed"
    MEMORY_EXTRACTION_FAILED = "memory_extraction_failed"
    MEMORY_CONSOLIDATION_REQUESTED = "memory_consolidation_requested"
    MEMORY_CONSOLIDATION_COMPLETED = "memory_consolidation_completed"
    MEMORY_CONSOLIDATION_SKIPPED = "memory_consolidation_skipped"
    MEMORY_CONSOLIDATION_FAILED = "memory_consolidation_failed"
    TOOL_HOOK_BLOCKED = "tool_hook_blocked"
    TOOL_HOOK_FAILED = "tool_hook_failed"
    TOOL_CALLED = "tool_called"
    TOOL_STARTED = "tool_started"
    TOOL_DENIED = "tool_denied"
    TOOL_RESULT = "tool_result"
    TOOL_RESULT_RETAINED = "tool_result_retained"
    WORKSPACE_OBSERVED = "workspace_observed"
    TODO_UPDATED = "todo_updated"
    SUBAGENT_STARTED = "subagent_started"
    SUBAGENT_FINISHED = "subagent_finished"
    SUBAGENT_FAILED = "subagent_failed"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"


@dataclass(frozen=True)
class Event:
    """One ordered event in a single agent run."""

    run_id: str
    sequence: int
    timestamp: str
    event_type: str
    data: dict[str, Any]


class EventLogError(RuntimeError):
    """Raised when an enabled event log cannot be written."""


def hash_text(value: str) -> str:
    """Return a deterministic SHA-256 identity without retaining text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_json(value: Any) -> str:
    """Canonicalize a JSON value before returning its SHA-256 identity."""

    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hash_text(canonical)


class EventLogger(Protocol):
    """Synchronous event sink used by the agent and tool runtime."""

    def emit(
        self,
        event_type: EventType,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist one event or raise EventLogError."""
        ...


class NullEventLogger:
    """No-op logger used when event logging is disabled."""

    def emit(
        self,
        event_type: EventType,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        del event_type, data


NULL_EVENT_LOGGER = NullEventLogger()


class ScopedEventLogger:
    """Attach fixed correlation metadata to every delegated event."""

    def __init__(
        self,
        logger: EventLogger,
        metadata: Mapping[str, Any],
    ) -> None:
        self._logger = logger
        self._metadata = dict(metadata)

    def emit(
        self,
        event_type: EventType,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        merged = dict(data or {})
        merged.update(self._metadata)
        self._logger.emit(event_type, merged)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JsonlEventLogger:
    """Append ordered events to a JSON Lines file."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.path = path.resolve()
        self.run_id = run_id or uuid4().hex
        self._clock = clock
        self._sequence = 0

    def emit(
        self,
        event_type: EventType,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        self._sequence += 1
        timestamp = self._clock().astimezone(timezone.utc).isoformat()
        event = Event(
            run_id=self.run_id,
            sequence=self._sequence,
            timestamp=timestamp,
            event_type=event_type.value,
            data=dict(data or {}),
        )

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as event_file:
                event_file.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
                event_file.flush()
        except (OSError, TypeError, ValueError) as error:
            raise EventLogError(f"Failed to write event log: {self.path}") from error


class CompositeEventLogger:
    """Fan out synchronously; sink failures retain the EventLogError contract."""

    def __init__(self, *loggers: EventLogger) -> None:
        self.loggers = loggers

    def emit(self, event_type: EventType, data: Mapping[str, Any] | None = None) -> None:
        for logger in self.loggers:
            logger.emit(event_type, dict(data or {}))
