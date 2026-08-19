"""CLI for real coding and deterministic reliability evals."""

import argparse
import json
import os
import statistics
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.core import (
    EvalCase,
    EvalResult,
    collect_metrics,
    load_cases,
    read_jsonl_events,
)
from evals.graders import (
    hidden_grader_changed,
    prepare_case,
    run_hidden_grader,
)
from evals.scenarios import run_offline_scenarios
from tiny_harness.__main__ import DEFAULT_BASE_URL, DEFAULT_MODEL
from tiny_harness.agent.loop import run_agent as agent_loop
from tiny_harness.models.chat_completions import ChatCompletionsProvider
from tiny_harness.runtime.events import JsonlEventLogger
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.runtime.recovery import RecoveryPolicy

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "cases.json"
DEFAULT_FIXTURES = ROOT / "fixtures"


class EvalPermissionPolicy:
    """Use identical tool access and an exact Bash allowlist in both profiles."""

    _NON_SHELL = frozenset(
        {
            "read_file",
            "write_file",
            "edit_file",
            "list_files",
            "todo_write",
            "task",
            "compact",
        }
    )

    def __init__(self, allowed_bash: Sequence[str]) -> None:
        self.allowed_bash = frozenset(allowed_bash)

    def decide(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> PermissionDecision:
        if tool_name in self._NON_SHELL:
            return PermissionDecision.ALLOW
        if (
            tool_name == "bash"
            and arguments.get("command") in self.allowed_bash
        ):
            return PermissionDecision.ALLOW
        return PermissionDecision.DENY


def _system_prompt(workspace: Path) -> str:
    return (
        f"You are a coding agent working in {workspace}. Complete the task "
        "using available tools. Do not modify tests. Use todo_write for "
        "multi-step work. Use task only for a focused delegated subtask. "
        "Run the task's declared verification command and rely on direct tool "
        "results rather than unsupported completion claims."
    )


def run_real_case(
    case: EvalCase,
    *,
    profile: str,
    repetition: int,
    provider: ChatCompletionsProvider,
    fixtures_root: Path,
    results_root: Path,
) -> EvalResult:
    prepared = prepare_case(
        case,
        fixtures_root=fixtures_root,
        results_root=results_root,
        profile=profile,
        repetition=repetition,
    )
    logger = JsonlEventLogger(prepared.event_log)
    messages = [
        {"role": "system", "content": _system_prompt(prepared.workspace)},
        {"role": "user", "content": case.task},
    ]
    started = time.monotonic()
    returned = False
    error: Exception | None = None
    try:
        agent_loop(
            provider,
            prepared.workspace,
            messages,
            max_turns=case.max_turns,
            permission_policy=EvalPermissionPolicy(case.allowed_bash),
            event_logger=logger,
            max_context_chars=None,
            subagent_max_turns=case.max_turns,
            allow_subagent=True,
            recovery_policy=RecoveryPolicy(
                max_retries=2 if profile == "reliable" else 0,
            ),
            goal_condition=case.goal if profile == "reliable" else None,
            max_goal_retries=2,
        )
        returned = True
    except Exception as caught:
        error = caught
    elapsed_ms = round((time.monotonic() - started) * 1000)

    grader_was_changed = hidden_grader_changed(prepared)
    grade = run_hidden_grader(prepared)
    verified = returned and grade.passed and not grader_was_changed
    events = read_jsonl_events(prepared.event_log)
    return EvalResult(
        category="real_coding",
        case_id=case.id,
        profile=profile,
        repetition=repetition,
        verified_success=verified,
        false_success=returned and not verified,
        explicit_failure=not returned,
        side_effect_violation=grader_was_changed,
        agent_returned=returned,
        error_type=type(error).__name__ if error is not None else None,
        grader_exit_code=grade.exit_code,
        elapsed_ms=elapsed_ms,
        metrics=collect_metrics(events),
    )


def run_real_suite(
    cases: list[EvalCase],
    *,
    profiles: Sequence[str],
    repetitions: int,
    fixtures_root: Path,
    results_root: Path,
) -> list[EvalResult]:
    api_key = os.getenv("TINYHARNESS_API_KEY")
    if not api_key:
        raise RuntimeError("TINYHARNESS_API_KEY is required for real evals")
    results = []
    for case in cases:
        for profile in profiles:
            for repetition in range(1, repetitions + 1):
                provider = ChatCompletionsProvider(
                    api_key=api_key,
                    model=os.getenv("TINYHARNESS_MODEL", DEFAULT_MODEL),
                    base_url=os.getenv(
                        "TINYHARNESS_BASE_URL",
                        DEFAULT_BASE_URL,
                    ),
                )
                print(f"[eval] {case.id} / {profile} / run {repetition}")
                results.append(
                    run_real_case(
                        case,
                        profile=profile,
                        repetition=repetition,
                        provider=provider,
                        fixtures_root=fixtures_root,
                        results_root=results_root,
                    )
                )
    return results


def _mean(results: list[EvalResult], field: str) -> str:
    if not results:
        return "0.0"
    values = [getattr(result.metrics, field) for result in results]
    return f"{statistics.mean(values):.1f}"


def render_markdown(results: list[EvalResult]) -> str:
    lines = [
        "# TinyHarness Reliability Eval Report",
        "",
        "## Real Coding: Basic vs Reliable",
        "",
        "| Profile | Runs | Verified | False success | Explicit failure "
        "| Avg main | Avg goal | Avg summary | Avg total | Avg turns |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    real = [result for result in results if result.category == "real_coding"]
    for profile in ("basic_ablation", "reliable"):
        selected = [result for result in real if result.profile == profile]
        lines.append(
            f"| {profile} | {len(selected)} | "
            f"{sum(result.verified_success for result in selected)} | "
            f"{sum(result.false_success for result in selected)} | "
            f"{sum(result.explicit_failure for result in selected)} | "
            f"{_mean(selected, 'main_model_attempts')} | "
            f"{_mean(selected, 'goal_model_attempts')} | "
            f"{_mean(selected, 'summary_model_attempts')} | "
            f"{_mean(selected, 'total_model_attempts')} | "
            f"{_mean(selected, 'turns')} |"
        )

    lines.extend(
        [
            "",
            "### Real Coding Per-run Results",
            "",
            "| Case | Profile | Run | Verified | False success | Explicit "
            "failure | Model attempts | Turns | Tool calls |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in real:
        lines.append(
            f"| {result.case_id} | {result.profile} | {result.repetition} | "
            f"{int(result.verified_success)} | {int(result.false_success)} | "
            f"{int(result.explicit_failure)} | "
            f"{result.metrics.total_model_attempts} | "
            f"{result.metrics.turns} | {result.metrics.tool_calls} |"
        )

    lines.extend(
        [
            "",
            "## Controlled Failure Recovery",
            "",
            "| Scenario | Profile | Fault triggered | Recovered | Attempts "
            "| Retries | Continuations |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    controlled = [
        result
        for result in results
        if result.category == "controlled_failure_recovery"
    ]
    for result in controlled:
        lines.append(
            f"| {result.case_id} | {result.profile} | "
            f"{int(result.fault_triggered)} | "
            f"{int(bool(result.recovery_success))} | "
            f"{result.metrics.total_model_attempts} | "
            f"{result.metrics.retries} | {result.metrics.continuations} |"
        )

    lines.extend(
        [
            "",
            "## Safety Invariants",
            "",
            "| Invariant | Fault triggered | Passed | Side-effect violation |",
            "|---|---:|---:|---:|",
        ]
    )
    invariants = [
        result for result in results if result.category == "safety_invariant"
    ]
    for result in invariants:
        lines.append(
            f"| {result.case_id} | {int(result.fault_triggered)} | "
            f"{int(bool(result.invariant_passed))} | "
            f"{int(result.side_effect_violation)} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(results_root: Path, results: list[EvalResult]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": os.getenv("TINYHARNESS_MODEL", DEFAULT_MODEL),
        "results": [result.to_dict() for result in results],
    }
    (results_root / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (results_root / "report.md").write_text(
        render_markdown(results),
        encoding="utf-8",
    )


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TinyHarness reliability evals")
    parser.add_argument(
        "--suite",
        choices=("offline", "real", "all"),
        default="offline",
    )
    parser.add_argument(
        "--profile",
        choices=("both", "basic_ablation", "reliable"),
        default="both",
    )
    parser.add_argument("--repetitions", type=_positive_int, default=1)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--cases-file", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--fixtures-root", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--results-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_root = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else (ROOT / "results" / stamp).resolve()
    )
    results: list[EvalResult] = []
    if args.suite in {"offline", "all"}:
        results.extend(run_offline_scenarios())
    if args.suite in {"real", "all"}:
        cases = load_cases(args.cases_file.resolve())
        if args.case_ids:
            selected = set(args.case_ids)
            cases = [case for case in cases if case.id in selected]
            missing = selected - {case.id for case in cases}
            if missing:
                raise ValueError("Unknown eval cases: " + ", ".join(sorted(missing)))
        profiles = (
            ("basic_ablation", "reliable")
            if args.profile == "both"
            else (args.profile,)
        )
        results.extend(
            run_real_suite(
                cases,
                profiles=profiles,
                repetitions=args.repetitions,
                fixtures_root=args.fixtures_root.resolve(),
                results_root=results_root,
            )
        )
    write_reports(results_root, results)
    print(f"Reports written to {results_root}")

    invalid_fault = any(
        result.fault_expected and not result.fault_triggered
        for result in results
    )
    invariant_failure = any(
        result.invariant_passed is False for result in results
    )
    recovery_failure = any(
        result.category == "controlled_failure_recovery"
        and result.profile == "reliable"
        and result.recovery_success is not True
        for result in results
    )
    return 1 if invalid_fault or invariant_failure or recovery_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
