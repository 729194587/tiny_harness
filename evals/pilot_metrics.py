"""Run ledgers and aggregate reports for the single-profile Pilot."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from evals.core import EvalResult, read_jsonl_events
from evals.graders import POST_RUN_GRADE_FILENAME

SCHEMA_VERSION = 2
TERMINAL_GRADE = "terminal_grade.json"
RUN_LEDGER = "run.json"


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def count_tool_denied(
    events: Sequence[Mapping[str, Any]],
    *,
    tool_name: str | None = None,
) -> int:
    return sum(
        event.get("event_type") == "tool_denied"
        and event.get("data", {}).get("agent_scope") is None
        and (tool_name is None or event.get("data", {}).get("tool_name") == tool_name)
        for event in events
    )


def unavailable_post_run_grade() -> dict[str, Any]:
    return {
        "available": False,
        "valid": None,
        "passed": None,
        "exit_code": None,
        "snapshot_digest": None,
        "elapsed_ms": None,
    }


def build_run_ledger(result: EvalResult, run_root: Path) -> dict[str, Any]:
    events = read_jsonl_events(run_root / "events.jsonl")
    terminal = _read_json(run_root / TERMINAL_GRADE, None)
    final_grade = _read_json(
        run_root / POST_RUN_GRADE_FILENAME,
        unavailable_post_run_grade(),
    )
    if result.invalid_run:
        outcome = "invalid"
    elif result.verified_success:
        outcome = "verified"
    elif result.false_success:
        outcome = "premature_terminal_completion"
    else:
        outcome = "explicit_failure"
    error_type = result.error_type
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": result.case_id,
        "profile": result.profile,
        "repetition": result.repetition,
        "outcome": outcome,
        "verified": result.verified_success,
        "premature_terminal_completion": result.false_success,
        "explicit_failure": result.explicit_failure,
        "max_turns_exceeded": error_type == "MaxTurnsExceededError",
        "invalid_run": result.invalid_run,
        "agent_returned": result.agent_returned,
        "error_type": error_type,
        "grader_exit_code": result.grader_exit_code,
        "elapsed_ms": result.elapsed_ms,
        "model_attempts": result.metrics.total_model_attempts,
        "main_model_attempts": result.metrics.main_model_attempts,
        "summary_model_attempts": result.metrics.summary_model_attempts,
        "turns": result.metrics.turns,
        "retries": result.metrics.retries,
        "tool_calls": result.metrics.tool_calls,
        "run_tests_calls": result.metrics.run_tests_calls,
        "tool_denied": count_tool_denied(events),
        "bash_test_denied": count_tool_denied(events, tool_name="bash"),
        "terminal_grade": terminal,
        "final_workspace_grade": final_grade,
    }


def write_run_ledger(result: EvalResult, run_root: Path) -> dict[str, Any]:
    ledger = build_run_ledger(result, run_root)
    (run_root / RUN_LEDGER).write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ledger


def load_run_ledgers(results_root: Path) -> list[dict[str, Any]]:
    runs_root = results_root / "runs"
    if not runs_root.exists():
        return []
    ledgers = []
    for path in sorted(runs_root.glob(f"*/{RUN_LEDGER}")):
        value = _read_json(path, None)
        if isinstance(value, dict):
            ledgers.append(value)
    return ledgers


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _average(values: Sequence[int | float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [run for run in runs if not run.get("invalid_run")]
    verified = sum(bool(run.get("verified")) for run in valid)
    premature = sum(bool(run.get("premature_terminal_completion")) for run in valid)
    explicit = sum(bool(run.get("explicit_failure")) for run in valid)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": len(runs),
        "valid_runs": len(valid),
        "invalid_runs": len(runs) - len(valid),
        "verified": verified,
        "premature_terminal_completion": premature,
        "explicit_failure": explicit,
        "verified_rate": _rate(verified, len(valid)),
        "premature_terminal_rate": _rate(premature, len(valid)),
        "max_turns_exceeded": sum(bool(r.get("max_turns_exceeded")) for r in valid),
        "tool_denied": sum(int(r.get("tool_denied", 0)) for r in runs),
        "bash_test_denied": sum(int(r.get("bash_test_denied", 0)) for r in runs),
        "run_tests_calls": sum(int(r.get("run_tests_calls", 0)) for r in runs),
        "average_turns": _average([int(r.get("turns", 0)) for r in valid]),
        "average_tool_calls": _average([int(r.get("tool_calls", 0)) for r in valid]),
        "average_model_attempts": _average([int(r.get("model_attempts", 0)) for r in valid]),
    }


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# TinyHarness Coding Pilot",
            "",
            "Single ordinary Harness profile with external hidden grading.",
            "",
            "| Runs | Valid | Invalid | Verified | Premature terminal | Explicit failure |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {summary['runs']} | {summary['valid_runs']} | {summary['invalid_runs']} | {summary['verified']} | {summary['premature_terminal_completion']} | {summary['explicit_failure']} |",
            "",
            f"Verified rate: **{_percent(summary['verified_rate'])}**  ",
            f"Premature terminal rate: **{_percent(summary['premature_terminal_rate'])}**",
            "",
            "## Diagnostics",
            "",
            "| MaxTurnsExceeded | Tool denied | Bash test denied | run_tests calls | Avg turns | Avg tool calls | Avg model attempts |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {summary['max_turns_exceeded']} | {summary['tool_denied']} | {summary['bash_test_denied']} | {summary['run_tests_calls']} | {summary['average_turns'] or 0:.2f} | {summary['average_tool_calls'] or 0:.2f} | {summary['average_model_attempts'] or 0:.2f} |",
            "",
        ]
    )


CSV_FIELDS = (
    "case_id", "repetition", "outcome", "verified",
    "premature_terminal_completion", "explicit_failure",
    "max_turns_exceeded", "invalid_run", "error_type", "turns",
    "tool_calls", "run_tests_calls", "tool_denied", "model_attempts",
    "elapsed_ms",
)


def write_reports(results_root: Path, runs: Sequence[Mapping[str, Any]]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    summary = aggregate_runs(runs)
    (results_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_root / "summary.md").write_text(
        render_summary_markdown(summary), encoding="utf-8"
    )
    with (results_root / "runs.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
