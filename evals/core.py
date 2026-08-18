"""Shared contracts and event-derived metrics for Phase 11 evals."""

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tiny_harness.runtime.events import EventType


@dataclass(frozen=True)
class EvalCase:
    """One real coding task and its external grading configuration."""

    id: str
    task: str
    goal: str
    max_turns: int
    allowed_bash: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunMetrics:
    """Control-flow counts derived only from Event Log metadata."""

    main_model_attempts: int = 0
    goal_model_attempts: int = 0
    summary_model_attempts: int = 0
    total_model_attempts: int = 0
    turns: int = 0
    retries: int = 0
    continuations: int = 0
    tool_calls: int = 0


@dataclass
class EvalResult:
    """Normalized result used by all three Phase 11 report sections."""

    category: str
    case_id: str
    profile: str
    repetition: int = 1
    verified_success: bool = False
    false_success: bool = False
    explicit_failure: bool = False
    recovery_success: bool | None = None
    invariant_passed: bool | None = None
    side_effect_violation: bool = False
    fault_expected: bool = False
    fault_triggered: bool = False
    agent_returned: bool = False
    error_type: str | None = None
    grader_exit_code: int | None = None
    elapsed_ms: int = 0
    metrics: RunMetrics = field(default_factory=RunMetrics)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RecordingEventLogger:
    """In-memory event sink for deterministic offline scenarios."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(
        self,
        event_type: EventType,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        self.events.append(
            {"event_type": event_type.value, "data": dict(data or {})}
        )


def load_cases(path: Path) -> list[EvalCase]:
    """Load the deliberately small JSON case format."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"cases"}:
        raise ValueError("Eval suite must be an object containing only 'cases'")
    raw_cases = value["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Eval suite 'cases' must be a non-empty list")

    cases: list[EvalCase] = []
    seen: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError("Each eval case must be an object")
        allowed_fields = {
            "id",
            "task",
            "goal",
            "max_turns",
            "allowed_bash",
        }
        unexpected = set(raw) - allowed_fields
        if unexpected:
            raise ValueError(
                "Unexpected eval case fields: " + ", ".join(sorted(unexpected))
            )
        case_id = raw.get("id")
        task = raw.get("task")
        goal = raw.get("goal")
        max_turns = raw.get("max_turns")
        allowed_bash = raw.get("allowed_bash", [])
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Eval case id must be a non-empty string")
        if re.fullmatch(r"[A-Za-z0-9_-]+", case_id) is None:
            raise ValueError(
                "Eval case id may contain only letters, digits, '_' and '-'"
            )
        if case_id in seen:
            raise ValueError(f"Duplicate eval case id: {case_id}")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Eval case {case_id} requires a task")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError(f"Eval case {case_id} requires a goal")
        if not isinstance(max_turns, int) or isinstance(max_turns, bool):
            raise ValueError(f"Eval case {case_id} max_turns must be an integer")
        if max_turns < 1:
            raise ValueError(f"Eval case {case_id} max_turns must be positive")
        if not isinstance(allowed_bash, list) or not all(
            isinstance(command, str) and command
            for command in allowed_bash
        ):
            raise ValueError(
                f"Eval case {case_id} allowed_bash must be a string list"
            )
        seen.add(case_id)
        cases.append(
            EvalCase(
                id=case_id,
                task=task.strip(),
                goal=goal.strip(),
                max_turns=max_turns,
                allowed_bash=tuple(allowed_bash),
            )
        )
    return cases


def read_jsonl_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                events.append(value)
    return events


def collect_metrics(events: list[dict[str, Any]]) -> RunMetrics:
    requested = [
        event
        for event in events
        if event.get("event_type") == "model_requested"
    ]

    def attempts(purpose: str) -> int:
        return sum(
            event.get("data", {}).get("purpose") == purpose
            for event in requested
        )

    parent_main_turns = [
        event.get("data", {}).get("turn", 0)
        for event in requested
        if event.get("data", {}).get("purpose") == "main"
        and event.get("data", {}).get("agent_scope") is None
    ]
    goal_events = [
        event
        for event in events
        if event.get("event_type") == "goal_evaluated"
        and event.get("data", {}).get("agent_scope") is None
    ]
    return RunMetrics(
        main_model_attempts=attempts("main"),
        goal_model_attempts=attempts("goal_evaluation"),
        summary_model_attempts=attempts("summary"),
        total_model_attempts=len(requested),
        turns=max(parent_main_turns, default=0),
        retries=sum(
            event.get("event_type") == "model_retry_scheduled"
            for event in events
        ),
        continuations=max(
            (
                event.get("data", {}).get("retries_used", 0)
                for event in goal_events
            ),
            default=0,
        ),
        tool_calls=sum(
            event.get("event_type") == "tool_started"
            for event in events
        ),
    )
