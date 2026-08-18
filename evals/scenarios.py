"""Deterministic failure recovery and safety invariant scenarios."""

import copy
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from evals.core import (
    EvalResult,
    RecordingEventLogger,
    collect_metrics,
)
from tiny_harness.agent.loop import agent_loop
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.runtime.goal import GoalEvaluation
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.runtime.recovery import RecoveryPolicy


class SequenceProvider:
    """Return scripted outcomes and prove which injected call was reached."""

    def __init__(
        self,
        outcomes: Iterable[ModelResponse | Exception],
        *,
        fault_calls: Iterable[int] = (),
    ) -> None:
        self.outcomes = list(outcomes)
        self.fault_calls = set(fault_calls)
        self.calls = 0
        self.fault_triggered = False

    def complete(self, messages, tools):
        del messages, tools
        self.calls += 1
        if self.calls in self.fault_calls:
            self.fault_triggered = True
        if not self.outcomes:
            raise AssertionError("SequenceProvider has no outcome left")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class SequenceGoalEvaluator:
    def __init__(self, outcomes: Iterable[GoalEvaluation]) -> None:
        self.outcomes = list(outcomes)

    def evaluate(self, condition, messages, candidate_answer):
        del condition, messages, candidate_answer
        if not self.outcomes:
            raise AssertionError("SequenceGoalEvaluator has no outcome left")
        return self.outcomes.pop(0)


class DenyAllPolicy:
    def decide(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> PermissionDecision:
        del tool_name, arguments
        return PermissionDecision.DENY


def _tool_block(call_id: str, result: str, *, text: str = ""):
    return [
        {
            "role": "assistant",
            "content": text or None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"evidence.txt"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _controlled_result(
    *,
    case_id: str,
    profile: str,
    provider: SequenceProvider,
    logger: RecordingEventLogger,
    agent_returned: bool,
    recovered: bool,
    error: Exception | None,
) -> EvalResult:
    return EvalResult(
        category="controlled_failure_recovery",
        case_id=case_id,
        profile=profile,
        verified_success=recovered,
        false_success=agent_returned and not recovered,
        explicit_failure=not agent_returned,
        recovery_success=recovered,
        fault_expected=True,
        fault_triggered=provider.fault_triggered,
        agent_returned=agent_returned,
        error_type=type(error).__name__ if error is not None else None,
        metrics=collect_metrics(logger.events),
    )


def _transient_retry(profile: str) -> EvalResult:
    provider = SequenceProvider(
        [
            ModelProviderError(ModelErrorKind.SERVER_UNAVAILABLE),
            ModelResponse("done", None, [], "stop"),
        ],
        fault_calls={1},
    )
    logger = RecordingEventLogger()
    error = None
    returned = False
    with tempfile.TemporaryDirectory() as directory:
        try:
            agent_loop(
                provider,
                Path(directory),
                [{"role": "user", "content": "task"}],
                max_turns=1,
                allow_subagent=True,
                event_logger=logger,
                recovery_policy=RecoveryPolicy(
                    max_retries=1 if profile == "reliable" else 0,
                    base_delay_seconds=0,
                    max_delay_seconds=0,
                    jitter_ratio=0,
                ),
            )
            returned = True
        except Exception as caught:
            error = caught
    return _controlled_result(
        case_id="transient_retry",
        profile=profile,
        provider=provider,
        logger=logger,
        agent_returned=returned,
        recovered=returned and provider.calls == 2,
        error=error,
    )


def _reactive_context(profile: str) -> EvalResult:
    outcomes: list[ModelResponse | Exception] = [
        ModelProviderError(ModelErrorKind.CONTEXT_LENGTH),
    ]
    if profile == "reliable":
        outcomes.extend(
            [
                ModelResponse("FACTUAL_SUMMARY", None, [], "stop"),
                ModelResponse("done", None, [], "stop"),
            ]
        )
    provider = SequenceProvider(outcomes, fault_calls={1})
    logger = RecordingEventLogger()
    messages = [
        {"role": "user", "content": "ORIGINAL_TASK"},
        *_tool_block("old", "old evidence", text="X" * 10_000),
        *_tool_block("latest", "LATEST_EVIDENCE"),
    ]
    error = None
    returned = False
    with tempfile.TemporaryDirectory() as directory:
        try:
            agent_loop(
                provider,
                Path(directory),
                copy.deepcopy(messages),
                max_turns=1,
                allow_subagent=True,
                event_logger=logger,
                max_context_chars=100_000 if profile == "reliable" else None,
                recovery_policy=RecoveryPolicy(
                    max_retries=0,
                    base_delay_seconds=0,
                    max_delay_seconds=0,
                    jitter_ratio=0,
                ),
            )
            returned = True
        except Exception as caught:
            error = caught
    compacted = any(
        event["event_type"] == "context_compacted"
        and event["data"].get("reason") == "reactive"
        for event in logger.events
    )
    return _controlled_result(
        case_id="reactive_context",
        profile=profile,
        provider=provider,
        logger=logger,
        agent_returned=returned,
        recovered=returned and compacted,
        error=error,
    )


def _premature_goal(profile: str) -> EvalResult:
    outcomes = [ModelResponse("premature", None, [], "stop")]
    evaluator = None
    if profile == "reliable":
        outcomes.extend(
            [
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall(
                            "write-1",
                            "write_file",
                            '{"path":"done.txt","content":"OK"}',
                        )
                    ],
                    "tool_calls",
                ),
                ModelResponse("verified", None, [], "stop"),
            ]
        )
        evaluator = SequenceGoalEvaluator(
            [
                GoalEvaluation(False, "missing file evidence"),
                GoalEvaluation(True, "file evidence present"),
            ]
        )
    provider = SequenceProvider(outcomes, fault_calls={1})
    logger = RecordingEventLogger()
    error = None
    returned = False
    file_exists = False
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        try:
            agent_loop(
                provider,
                workspace,
                [{"role": "user", "content": "create done.txt"}],
                max_turns=3,
                allow_subagent=True,
                event_logger=logger,
                recovery_policy=RecoveryPolicy(max_retries=0),
                goal_condition=(
                    "done.txt exists with content OK"
                    if profile == "reliable"
                    else None
                ),
                max_goal_retries=1,
                goal_evaluator=evaluator,
            )
            returned = True
        except Exception as caught:
            error = caught
        file_exists = (workspace / "done.txt").read_text(
            encoding="utf-8"
        ) == "OK" if (workspace / "done.txt").exists() else False
    return _controlled_result(
        case_id="premature_goal",
        profile=profile,
        provider=provider,
        logger=logger,
        agent_returned=returned,
        recovered=returned and file_exists,
        error=error,
    )


def run_controlled_scenarios() -> list[EvalResult]:
    results = []
    for profile in ("basic_ablation", "reliable"):
        results.extend(
            [
                _transient_retry(profile),
                _reactive_context(profile),
                _premature_goal(profile),
            ]
        )
    return results


def _permission_invariant() -> EvalResult:
    provider = SequenceProvider(
        [
            ModelResponse(
                None,
                None,
                [
                    ToolCall(
                        "write-denied",
                        "write_file",
                        '{"path":"forbidden.txt","content":"BAD"}',
                    )
                ],
                "tool_calls",
            ),
            ModelResponse("denied", None, [], "stop"),
        ],
        fault_calls={1},
    )
    logger = RecordingEventLogger()
    error = None
    returned = False
    side_effect = False
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        try:
            agent_loop(
                provider,
                workspace,
                [{"role": "user", "content": "write forbidden.txt"}],
                max_turns=2,
                permission_policy=DenyAllPolicy(),
                event_logger=logger,
                recovery_policy=RecoveryPolicy(max_retries=0),
            )
            returned = True
        except Exception as caught:
            error = caught
        side_effect = (workspace / "forbidden.txt").exists()
    passed = returned and not side_effect
    return EvalResult(
        category="safety_invariant",
        case_id="permission_deny_no_side_effect",
        profile="runtime",
        verified_success=passed,
        explicit_failure=not returned,
        invariant_passed=passed,
        side_effect_violation=side_effect,
        fault_expected=True,
        fault_triggered=provider.fault_triggered,
        agent_returned=returned,
        error_type=type(error).__name__ if error else None,
        metrics=collect_metrics(logger.events),
    )


def _finish_reason_invariant() -> EvalResult:
    provider = SequenceProvider(
        [
            ModelResponse(
                "partial",
                None,
                [
                    ToolCall(
                        "unsafe-write",
                        "write_file",
                        '{"path":"unsafe.txt","content":"BAD"}',
                    )
                ],
                "length",
            )
        ],
        fault_calls={1},
    )
    logger = RecordingEventLogger()
    error = None
    side_effect = False
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        try:
            agent_loop(
                provider,
                workspace,
                [{"role": "user", "content": "task"}],
                max_turns=1,
                event_logger=logger,
                recovery_policy=RecoveryPolicy(max_retries=0),
            )
        except Exception as caught:
            error = caught
        side_effect = (workspace / "unsafe.txt").exists()
    passed = isinstance(error, RuntimeError) and not side_effect
    return EvalResult(
        category="safety_invariant",
        case_id="invalid_finish_reason_no_side_effect",
        profile="runtime",
        verified_success=passed,
        explicit_failure=True,
        invariant_passed=passed,
        side_effect_violation=side_effect,
        fault_expected=True,
        fault_triggered=provider.fault_triggered,
        agent_returned=False,
        error_type=type(error).__name__ if error else None,
        metrics=collect_metrics(logger.events),
    )


def run_safety_invariants() -> list[EvalResult]:
    return [_permission_invariant(), _finish_reason_invariant()]


def run_offline_scenarios() -> list[EvalResult]:
    return [*run_controlled_scenarios(), *run_safety_invariants()]
