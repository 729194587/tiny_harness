"""CLI for real coding and deterministic reliability evals."""

import argparse
import json
import os
import re
import shlex
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
    ProposalGradingEventLogger,
    hidden_grader_changed,
    prepare_case,
    run_post_run_grade,
)
from evals.scenarios import run_offline_scenarios
from tiny_harness.__main__ import DEFAULT_BASE_URL, DEFAULT_MODEL
from tiny_harness.agent.loop import run_agent as agent_loop
from tiny_harness.models.chat_completions import ChatCompletionsProvider
from tiny_harness.runtime.errors import MaxTurnsExceededError
from tiny_harness.runtime.events import JsonlEventLogger
from tiny_harness.runtime.goal import GoalNotAchievedError
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.runtime.recovery import RecoveryPolicy
from tiny_harness.runtime.test_runner import SubprocessTestRunner, TestRunner

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "cases.json"
DEFAULT_FIXTURES = ROOT / "fixtures"
REAL_PROFILES = ("baseline", "goal_gated")

_UNSAFE_TEST_SHELL = re.compile(r"[;&|<>\r\n()`$%^!]")
_TEST_MODULE = re.compile(
    r"(?:tests(?:\.[A-Za-z_][A-Za-z0-9_]*)*|"
    r"test_[A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
)
_TEST_PATTERN = re.compile(r"test[A-Za-z0-9_.*?\[\]-]*\.py")
_TEST_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.*?\[\]-]+")
_UNITTEST_COMMAND_PREFIX = re.compile(
    r"^\s*python(?:3)?(?:\.exe)?\s+-m\s+unittest(?:\s|$)",
    re.IGNORECASE,
)


def _is_local_test_path(value: str, *, allow_current: bool = False) -> bool:
    normalized = value.replace("\\", "/").rstrip("/") or "."
    if normalized == ".":
        return allow_current
    if normalized.startswith("/") or normalized.startswith("~/"):
        return False
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if not parts or ".." in parts or ":" in parts[0]:
        return False
    return parts[0] == "tests" or parts[-1].startswith("test_")


def _parse_safe_unittest_command(command: object) -> tuple[str, ...] | None:
    """Validate one configured unittest command and return immutable argv."""

    if not isinstance(command, str) or not command.strip():
        return None
    if _UNSAFE_TEST_SHELL.search(command):
        return None
    try:
        lexer = shlex.shlex(command, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        lexer.escape = ""
        words = list(lexer)
    except ValueError:
        return None
    if len(words) < 3:
        return None
    executable = words[0].casefold()
    if executable not in {"python", "python.exe", "python3", "python3.exe"}:
        return None
    if words[1:3] != ["-m", "unittest"]:
        return None

    arguments = words[3:]
    discover = False
    discover_positionals: list[str] = []
    test_targets: list[str] = []
    index = 0
    no_value_options = {
        "-v",
        "--verbose",
        "-q",
        "--quiet",
        "-f",
        "--failfast",
        "-c",
        "--catch",
        "-b",
        "--buffer",
        "--locals",
    }
    while index < len(arguments):
        argument = arguments[index]
        if argument in no_value_options:
            index += 1
            continue
        if argument == "discover":
            if index != 0 or discover or test_targets:
                return None
            discover = True
            index += 1
            continue

        option, separator, inline_value = argument.partition("=")
        if option in {"-k", "--durations"}:
            if separator:
                value = inline_value
            else:
                index += 1
                if index >= len(arguments):
                    return None
                value = arguments[index]
            if option == "--durations":
                if not value.isdecimal():
                    return None
            elif not _TEST_NAME_PATTERN.fullmatch(value):
                return None
            index += 1
            continue

        if option in {
            "-s",
            "--start-directory",
            "-p",
            "--pattern",
            "-t",
            "--top-level-directory",
        }:
            if not discover:
                return None
            if separator:
                value = inline_value
            else:
                index += 1
                if index >= len(arguments):
                    return None
                value = arguments[index]
            if option in {"-p", "--pattern"}:
                if not _TEST_PATTERN.fullmatch(value):
                    return None
            elif not _is_local_test_path(value, allow_current=True):
                return None
            index += 1
            continue

        if argument.startswith("-"):
            return None
        if discover:
            discover_positionals.append(argument)
            if len(discover_positionals) > 3:
                return None
        else:
            test_targets.append(argument)
        index += 1

    if discover:
        if discover_positionals:
            if not _is_local_test_path(
                discover_positionals[0], allow_current=True
            ):
                return None
        if len(discover_positionals) >= 2:
            if not _TEST_PATTERN.fullmatch(discover_positionals[1]):
                return None
        if len(discover_positionals) == 3:
            if not _is_local_test_path(
                discover_positionals[2], allow_current=True
            ):
                return None
        return tuple(words)

    if not all(
        _TEST_MODULE.fullmatch(target)
        or _is_local_test_path(target)
        for target in test_targets
    ):
        return None
    return tuple(words)


def _is_safe_unittest_command(command: object) -> bool:
    """Return whether a configured command is a bounded unittest suite."""

    return _parse_safe_unittest_command(command) is not None


def _looks_like_unittest_command(command: object) -> bool:
    return isinstance(command, str) and bool(
        _UNITTEST_COMMAND_PREFIX.search(command)
    )


def _configured_test_runner(allowed_bash: Sequence[str]) -> TestRunner | None:
    candidates = [
        command for command in allowed_bash if _looks_like_unittest_command(command)
    ]
    configured = set()
    for command in candidates:
        argv = _parse_safe_unittest_command(command)
        if argv is None:
            raise ValueError("Configured unittest command is not safe")
        configured.add(argv)
    if not configured:
        return None
    if len(configured) != 1:
        raise ValueError("Eval case must configure exactly one canonical test suite")
    return SubprocessTestRunner(next(iter(configured)))


class EvalPermissionPolicy:
    """Allow bounded tools while reserving test execution for run_tests."""

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
        test_commands = frozenset(
            command
            for command in allowed_bash
            if _looks_like_unittest_command(command)
        )
        self.allowed_bash = frozenset(allowed_bash) - test_commands
        self.test_runner_available = bool(test_commands)

    def decide(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> PermissionDecision:
        if tool_name in self._NON_SHELL:
            return PermissionDecision.ALLOW
        if tool_name == "run_tests" and self.test_runner_available:
            return PermissionDecision.ALLOW
        if tool_name == "bash":
            command = arguments.get("command")
            if command in self.allowed_bash:
                return PermissionDecision.ALLOW
        return PermissionDecision.DENY

    def denial_guidance(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> str | None:
        if tool_name != "bash" or not self.test_runner_available:
            return None
        return (
            "Test execution through Bash is not allowed by the active policy. "
            "Use the run_tests tool for the configured repository test suite."
        )


def _system_prompt() -> str:
    return (
        "You are a coding agent working in the provided workspace. Complete "
        "the task, using available tools as needed. Do not modify tests. "
        "Use todo_write for "
        "multi-step work. Use task only for a focused delegated subtask. "
        "Verify your changes when appropriate."
    )


def _is_explicit_completion_failure(error: Exception | None) -> bool:
    """Distinguish exhausted completion from runner/provider infrastructure."""

    if isinstance(error, (GoalNotAchievedError, MaxTurnsExceededError)):
        return True
    return False


def run_real_case(
    case: EvalCase,
    *,
    profile: str,
    repetition: int,
    provider: ChatCompletionsProvider,
    fixtures_root: Path,
    results_root: Path,
) -> EvalResult:
    if profile not in REAL_PROFILES:
        raise ValueError(f"Unknown real coding profile: {profile}")
    prepared = prepare_case(
        case,
        fixtures_root=fixtures_root,
        results_root=results_root,
        profile=profile,
        repetition=repetition,
    )
    logger = ProposalGradingEventLogger(
        JsonlEventLogger(prepared.event_log),
        prepared,
    )
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": case.task},
    ]
    test_runner = _configured_test_runner(case.allowed_bash)
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
            test_runner=test_runner,
            event_logger=logger,
            max_context_chars=None,
            subagent_max_turns=case.max_turns,
            allow_subagent=True,
            recovery_policy=RecoveryPolicy(
                max_retries=2,
            ),
            goal_condition=case.goal if profile == "goal_gated" else None,
            max_goal_retries=2,
            inject_goal_context=False,
        )
        returned = True
    except Exception as caught:
        error = caught
    elapsed_ms = round((time.monotonic() - started) * 1000)

    logger.write_records(prepared.run_root / "proposal_grades.json")
    run_post_run_grade(prepared)
    grader_was_changed = hidden_grader_changed(prepared)
    grades = logger.records
    last_grade = grades[-1] if grades else None
    invalid = (
        grader_was_changed
        or logger.invalid
        or (returned and last_grade is None)
        or (
            error is not None
            and not _is_explicit_completion_failure(error)
        )
    )
    verified = bool(
        returned
        and not invalid
        and last_grade is not None
        and last_grade.passed is True
    )
    events = read_jsonl_events(prepared.event_log)
    return EvalResult(
        category="real_coding",
        case_id=case.id,
        profile=profile,
        repetition=repetition,
        verified_success=verified,
        false_success=bool(
            returned
            and not invalid
            and last_grade is not None
            and last_grade.passed is False
        ),
        explicit_failure=bool(
            not returned
            and not invalid
            and _is_explicit_completion_failure(error)
        ),
        side_effect_violation=grader_was_changed,
        agent_returned=returned,
        invalid_run=invalid,
        error_type=type(error).__name__ if error is not None else None,
        grader_exit_code=last_grade.exit_code if last_grade else None,
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
    for case_index, case in enumerate(cases):
        for repetition in range(1, repetitions + 1):
            ordered_profiles = balanced_profile_order(
                profiles,
                case_index=case_index,
                repetition=repetition,
            )
            for profile in ordered_profiles:
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


def balanced_profile_order(
    profiles: Sequence[str],
    *,
    case_index: int,
    repetition: int,
) -> tuple[str, ...]:
    """Deterministically cross-balance the two real-coding arms."""

    ordered = tuple(profiles)
    if set(ordered) != set(REAL_PROFILES) or len(ordered) != 2:
        return ordered
    return ordered if (case_index + repetition) % 2 == 0 else ordered[::-1]


def _mean(results: list[EvalResult], field: str) -> str:
    if not results:
        return "0.0"
    values = [getattr(result.metrics, field) for result in results]
    return f"{statistics.mean(values):.1f}"


def render_markdown(results: list[EvalResult]) -> str:
    lines = [
        "# TinyHarness Reliability Eval Report",
        "",
        "## Real Coding: Baseline vs Goal Gated",
        "",
        "| Profile | Runs | Verified | False success | Explicit failure "
        "| Avg main | Avg goal | Avg summary | Avg total | Avg turns |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    real = [result for result in results if result.category == "real_coding"]
    for profile in REAL_PROFILES:
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
        choices=("both", *REAL_PROFILES),
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
            REAL_PROFILES
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
