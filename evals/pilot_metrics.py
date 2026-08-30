"""Pure ledger construction and aggregation for the Goal Gate Pilot."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from evals.core import EvalResult, read_jsonl_events
from evals.graders import POST_RUN_GRADE_FILENAME

SCHEMA_VERSION = 1
PROPOSAL_RECORDS = "proposal_grades.json"
FINAL_GRADE = "final_grade.json"
RUN_LEDGER = "run.json"


def unavailable_post_run_grade() -> dict[str, Any]:
    return {
        "available": False,
        "valid": None,
        "passed": None,
        "exit_code": None,
        "snapshot_digest": None,
        "elapsed_ms": None,
    }


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _root_event(event: Mapping[str, Any], event_type: str) -> bool:
    data = event.get("data")
    return (
        event.get("event_type") == event_type
        and isinstance(data, Mapping)
        and data.get("agent_scope") is None
    )


def count_tool_denied(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count denied tools offline by Agent scope and shell category."""

    diagnostics = {
        "total": 0,
        "root": {"total": 0, "bash": 0, "other": 0},
        "subagent": {"total": 0, "bash": 0, "other": 0},
    }
    for event in events:
        if event.get("event_type") != "tool_denied":
            continue
        data = event.get("data")
        if not isinstance(data, Mapping):
            continue
        scope = "root" if data.get("agent_scope") is None else "subagent"
        category = "bash" if data.get("tool_name") == "bash" else "other"
        diagnostics["total"] += 1
        diagnostics[scope]["total"] += 1
        diagnostics[scope][category] += 1
    return diagnostics


def _load_post_run_grade(run_root: Path) -> dict[str, Any]:
    value = _read_json(
        run_root / POST_RUN_GRADE_FILENAME,
        unavailable_post_run_grade(),
    )
    return value if isinstance(value, dict) else unavailable_post_run_grade()


def enrich_proposals(
    proposals: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach observation-only continuation metadata to proposal records."""

    stop_events = [
        (index, event)
        for index, event in enumerate(events)
        if _root_event(event, "stop_proposed")
    ]
    enriched = []
    for proposal_index, raw in enumerate(proposals):
        proposal = dict(raw)
        event_index = (
            stop_events[proposal_index][0]
            if proposal_index < len(stop_events)
            else None
        )
        continued = False
        if event_index is not None:
            continued = any(
                later_index > event_index
                and _root_event(event, "model_requested")
                and event.get("data", {}).get("purpose") == "main"
                for later_index, event in enumerate(events)
            )
        proposal["agent_continued"] = continued
        if not proposal.get("valid", False):
            proposal["external_grade"] = "INVALID"
        elif proposal.get("passed") is True:
            proposal["external_grade"] = "PASS"
        elif proposal.get("passed") is False:
            proposal["external_grade"] = "FAIL"
        else:
            proposal["external_grade"] = "INVALID"
        enriched.append(proposal)
    return enriched


def _final_grade(
    result: EvalResult,
    proposals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not result.agent_returned or not proposals:
        return {
            "available": False,
            "valid": None,
            "passed": None,
            "exit_code": None,
            "snapshot_digest": None,
            "proposal": None,
            "turn": None,
        }
    final = proposals[-1]
    valid = bool(final.get("valid", False))
    return {
        "available": True,
        "valid": valid,
        "passed": final.get("passed") if valid else None,
        "exit_code": final.get("exit_code") if valid else None,
        "snapshot_digest": final.get("snapshot_digest"),
        "proposal": final.get("proposal"),
        "turn": final.get("turn"),
    }


def build_run_ledger(result: EvalResult, run_root: Path) -> dict[str, Any]:
    """Build one sanitized run ledger from runner metadata and observations."""

    raw_proposals = _read_json(run_root / PROPOSAL_RECORDS, [])
    if not isinstance(raw_proposals, list):
        raw_proposals = []
    events = read_jsonl_events(run_root / "events.jsonl")
    proposals = enrich_proposals(raw_proposals, events)
    final_grade = _final_grade(result, proposals)
    post_run_grade = _load_post_run_grade(run_root)

    invalid = bool(result.invalid_run)
    if invalid:
        outcome = "invalid_run"
    elif result.verified_success:
        outcome = "verified_terminal_completion"
    elif result.false_success:
        outcome = "premature_terminal_completion"
    else:
        outcome = "explicit_failure"

    useful_intervention = bool(
        not invalid
        and final_grade["valid"] is True
        and final_grade["passed"] is True
        and any(
            proposal.get("valid") is True
            and proposal.get("passed") is False
            and proposal.get("gate_action") == "block"
            and proposal.get("agent_continued") is True
            for proposal in proposals
        )
    )
    metrics = asdict(result.metrics)
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "case_id": result.case_id,
        "profile": result.profile,
        "repetition": result.repetition,
        "outcome": outcome,
        "verified_terminal_completion": outcome
        == "verified_terminal_completion",
        "premature_terminal_completion": outcome
        == "premature_terminal_completion",
        "explicit_failure": outcome == "explicit_failure",
        "invalid_run": invalid,
        "invalid_reason": result.error_type if invalid else None,
        "error_type": result.error_type,
        "agent_returned": result.agent_returned,
        "main_model_attempts": metrics["main_model_attempts"],
        "goal_evaluator_calls": metrics["goal_model_attempts"],
        "logical_turns": metrics["turns"],
        "tool_calls": metrics["tool_calls"],
        "wall_time_ms": result.elapsed_ms,
        "useful_intervention": useful_intervention,
        "final_grade": final_grade,
        "post_run_grade": post_run_grade,
        "tool_denied": count_tool_denied(events),
        "proposals": proposals,
    }
    return ledger


def write_run_ledger(result: EvalResult, run_root: Path) -> dict[str, Any]:
    ledger = build_run_ledger(result, run_root)
    (run_root / RUN_LEDGER).write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_root / FINAL_GRADE).write_text(
        json.dumps(ledger["final_grade"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    post_run_path = run_root / POST_RUN_GRADE_FILENAME
    if not post_run_path.exists():
        post_run_path.write_text(
            json.dumps(
                ledger["post_run_grade"],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return ledger


def load_run_ledgers(results_root: Path) -> list[dict[str, Any]]:
    ledgers = []
    for path in sorted((results_root / "runs").glob(f"*/{RUN_LEDGER}")):
        value = _read_json(path, None)
        if not isinstance(value, dict):
            raise ValueError(f"Invalid run ledger: {path}")
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported run ledger schema: {path}")
        run_root = path.parent
        value["post_run_grade"] = _load_post_run_grade(run_root)
        value["tool_denied"] = count_tool_denied(
            read_jsonl_events(run_root / "events.jsonl")
        )
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        ledgers.append(value)
    if not ledgers:
        raise ValueError(f"No run ledgers found under {results_root / 'runs'}")
    return ledgers


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _average(values: Sequence[int | float]) -> float | None:
    return sum(values) / len(values) if values else None


def is_useful_intervention(run: Mapping[str, Any]) -> bool:
    """Derive Useful Intervention from primitive run/proposal observations."""

    return bool(
        run.get("profile") == "goal_gated"
        and run.get("invalid_run") is not True
        and run.get("verified_terminal_completion") is True
        and any(
            proposal.get("valid") is True
            and proposal.get("passed") is False
            and proposal.get("gate_action") == "block"
            and proposal.get("agent_continued") is True
            for proposal in run.get("proposals", [])
        )
    )


def _aggregate_group(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [run for run in runs if not run.get("invalid_run", False)]
    verified = sum(
        run.get("verified_terminal_completion") is True for run in eligible
    )
    premature = sum(
        run.get("premature_terminal_completion") is True for run in eligible
    )
    explicit = sum(run.get("explicit_failure") is True for run in eligible)

    valid_proposals = [
        proposal
        for run in eligible
        for proposal in run.get("proposals", [])
        if proposal.get("valid") is True
        and proposal.get("passed") in {True, False}
    ]
    premature_proposals = sum(
        proposal.get("passed") is False for proposal in valid_proposals
    )

    gated_runs = [run for run in eligible if run.get("profile") == "goal_gated"]
    gated_proposals = [
        proposal
        for run in gated_runs
        for proposal in run.get("proposals", [])
        if proposal.get("valid") is True
        and proposal.get("passed") in {True, False}
    ]
    correct_block = sum(
        proposal.get("passed") is False
        and proposal.get("gate_action") == "block"
        for proposal in gated_proposals
    )
    false_accept = sum(
        proposal.get("passed") is False
        and proposal.get("gate_action") == "allow"
        for proposal in gated_proposals
    )
    unnecessary_block = sum(
        proposal.get("passed") is True
        and proposal.get("gate_action") == "block"
        for proposal in gated_proposals
    )
    correct_allow = sum(
        proposal.get("passed") is True
        and proposal.get("gate_action") == "allow"
        for proposal in gated_proposals
    )
    correct_block_runs = [
        run
        for run in gated_runs
        if any(
            proposal.get("valid") is True
            and proposal.get("passed") is False
            and proposal.get("gate_action") == "block"
            for proposal in run.get("proposals", [])
        )
    ]
    useful = sum(is_useful_intervention(run) for run in gated_runs)

    all_runs = list(runs)
    post_grades = [
        run.get("post_run_grade", unavailable_post_run_grade())
        for run in all_runs
    ]
    explicit_post_grades = [
        run.get("post_run_grade", unavailable_post_run_grade())
        for run in all_runs
        if run.get("explicit_failure") is True
    ]

    def post_counts(grades: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {
            "pass": sum(
                grade.get("available") is True
                and grade.get("valid") is True
                and grade.get("passed") is True
                for grade in grades
            ),
            "fail": sum(
                grade.get("available") is True
                and grade.get("valid") is True
                and grade.get("passed") is False
                for grade in grades
            ),
            "invalid": sum(
                grade.get("available") is True
                and grade.get("valid") is not True
                for grade in grades
            ),
            "unavailable": sum(
                grade.get("available") is not True for grade in grades
            ),
        }

    denied = [
        run.get("tool_denied", count_tool_denied([])) for run in all_runs
    ]
    return {
        "runs": {
            "total": len(all_runs),
            "valid": len(eligible),
            "invalid": len(all_runs) - len(eligible),
            "verified_terminal_completion": verified,
            "premature_terminal_completion": premature,
            "explicit_failure": explicit,
        },
        "run_rates": {
            "verified_completion_rate": _rate(verified, len(eligible)),
            "premature_terminal_completion_rate": _rate(
                premature, len(eligible)
            ),
        },
        "stop_proposals": {
            "valid": len(valid_proposals),
            "premature": premature_proposals,
        },
        "goal_gate": {
            "eligible_runs": len(gated_runs),
            "correct_block": correct_block,
            "false_accept": false_accept,
            "unnecessary_block": unnecessary_block,
            "correct_allow": correct_allow,
            "correct_block_rate": _rate(
                correct_block, correct_block + false_accept
            ),
            "false_accept_rate": _rate(
                false_accept, correct_block + false_accept
            ),
            "unnecessary_block_rate": _rate(
                unnecessary_block, unnecessary_block + correct_allow
            ),
        },
        "useful_intervention": {
            "runs": useful,
            "runs_with_correct_block": len(correct_block_runs),
            "recovery_after_correct_block_rate": _rate(
                useful, len(correct_block_runs)
            ),
        },
        "resource_usage": {
            "total_main_model_attempts": sum(
                int(run.get("main_model_attempts", 0)) for run in all_runs
            ),
            "total_goal_evaluator_calls": sum(
                int(run.get("goal_evaluator_calls", 0)) for run in all_runs
            ),
            "total_logical_turns": sum(
                int(run.get("logical_turns", 0)) for run in all_runs
            ),
            "total_tool_calls": sum(
                int(run.get("tool_calls", 0)) for run in all_runs
            ),
            "total_wall_time_ms": sum(
                int(run.get("wall_time_ms", 0)) for run in all_runs
            ),
            "average_main_model_attempts": _average(
                [int(run.get("main_model_attempts", 0)) for run in all_runs]
            ),
            "average_goal_evaluator_calls": _average(
                [int(run.get("goal_evaluator_calls", 0)) for run in all_runs]
            ),
            "average_logical_turns": _average(
                [int(run.get("logical_turns", 0)) for run in all_runs]
            ),
            "average_tool_calls": _average(
                [int(run.get("tool_calls", 0)) for run in all_runs]
            ),
            "average_wall_time_ms": _average(
                [int(run.get("wall_time_ms", 0)) for run in all_runs]
            ),
        },
        "diagnostics": {
            "post_run_grade": post_counts(post_grades),
            "explicit_failure_post_run_grade": post_counts(
                explicit_post_grades
            ),
            "tool_denied": {
                "total": sum(int(item.get("total", 0)) for item in denied),
                "root": {
                    "total": sum(
                        int(item.get("root", {}).get("total", 0))
                        for item in denied
                    ),
                    "bash": sum(
                        int(item.get("root", {}).get("bash", 0))
                        for item in denied
                    ),
                    "other": sum(
                        int(item.get("root", {}).get("other", 0))
                        for item in denied
                    ),
                },
                "subagent": {
                    "total": sum(
                        int(item.get("subagent", {}).get("total", 0))
                        for item in denied
                    ),
                    "bash": sum(
                        int(item.get("subagent", {}).get("bash", 0))
                        for item in denied
                    ),
                    "other": sum(
                        int(item.get("subagent", {}).get("other", 0))
                        for item in denied
                    ),
                },
            },
        },
    }


def aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    profiles = sorted({str(run.get("profile")) for run in runs})
    invalid_runs = [
        {
            "case_id": run.get("case_id"),
            "profile": run.get("profile"),
            "repetition": run.get("repetition"),
            "reason": run.get("invalid_reason") or run.get("error_type"),
        }
        for run in runs
        if run.get("invalid_run") is True
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "definitions": {
            "valid_run_denominator": (
                "All non-invalid runs, including explicit failures."
            ),
            "premature_terminal_completion": (
                "Agent returned terminally and the final external grade was FAIL."
            ),
            "verified_terminal_completion": (
                "Agent returned terminally and the final external grade was PASS."
            ),
            "correct_block_rate": (
                "Correct Block / (Correct Block + False Accept), goal_gated only."
            ),
            "false_accept_rate": (
                "False Accept / (Correct Block + False Accept), goal_gated only."
            ),
            "unnecessary_block_rate": (
                "Unnecessary Block / (Unnecessary Block + Correct Allow), "
                "goal_gated only."
            ),
            "recovery_after_correct_block_rate": (
                "Useful Intervention runs / goal_gated runs with at least one "
                "Correct Block."
            ),
            "invalid_exclusion": (
                "Invalid runs are excluded from completion rates and all "
                "stop-level outcome metrics."
            ),
            "post_run_grade": (
                "Independent final-workspace diagnostic collected only after "
                "the Agent run ends; it never changes the run outcome."
            ),
            "tool_denied": (
                "Offline diagnostic counts from tool_denied events, split by "
                "root/subagent and bash/other."
            ),
        },
        "overall": _aggregate_group(runs),
        "by_profile": {
            profile: _aggregate_group(
                [run for run in runs if run.get("profile") == profile]
            )
            for profile in profiles
        },
        "invalid_runs": invalid_runs,
    }


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    groups = [("overall", summary["overall"]), *summary["by_profile"].items()]
    lines = [
        "# Goal Gate Completion Verification Pilot",
        "",
        "## Run outcomes",
        "",
        "| Group | Runs | Valid | Invalid | Verified | Premature | Explicit "
        "failure | Verified rate | Premature rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in groups:
        counts = group["runs"]
        rates = group["run_rates"]
        lines.append(
            f"| {name} | {counts['total']} | {counts['valid']} | "
            f"{counts['invalid']} | "
            f"{counts['verified_terminal_completion']} | "
            f"{counts['premature_terminal_completion']} | "
            f"{counts['explicit_failure']} | "
            f"{_percent(rates['verified_completion_rate'])} | "
            f"{_percent(rates['premature_terminal_completion_rate'])} |"
        )

    overall = summary["overall"]
    gate = overall["goal_gate"]
    useful = overall["useful_intervention"]
    lines.extend(
        [
            "",
            "## Stop proposals",
            "",
            f"Valid proposals: {overall['stop_proposals']['valid']}; "
            f"premature proposals: {overall['stop_proposals']['premature']}.",
            "",
            "Gate confusion matrix below includes only valid `goal_gated` runs.",
            "",
            "| Correct Block | False Accept | Unnecessary Block | Correct Allow | "
            "Correct block rate | False accept rate | Unnecessary block rate |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {gate['correct_block']} | {gate['false_accept']} | "
            f"{gate['unnecessary_block']} | {gate['correct_allow']} | "
            f"{_percent(gate['correct_block_rate'])} | "
            f"{_percent(gate['false_accept_rate'])} | "
            f"{_percent(gate['unnecessary_block_rate'])} |",
            "",
            "## Useful Intervention",
            "",
            f"Useful Intervention runs: {useful['runs']} / "
            f"{useful['runs_with_correct_block']} runs with a Correct Block "
            f"({_percent(useful['recovery_after_correct_block_rate'])}).",
            "",
            "## Resource usage",
            "",
            "| Group | Avg main attempts | Avg Goal calls | Avg turns | "
            "Avg tool calls | Avg wall time (ms) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, group in groups:
        usage = group["resource_usage"]
        lines.append(
            f"| {name} | {usage['average_main_model_attempts'] or 0:.2f} | "
            f"{usage['average_goal_evaluator_calls'] or 0:.2f} | "
            f"{usage['average_logical_turns'] or 0:.2f} | "
            f"{usage['average_tool_calls'] or 0:.2f} | "
            f"{usage['average_wall_time_ms'] or 0:.2f} |"
        )

    diagnostics = overall["diagnostics"]
    post = diagnostics["post_run_grade"]
    explicit_post = diagnostics["explicit_failure_post_run_grade"]
    denied = diagnostics["tool_denied"]
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "Post-run grading is observation-only and does not change any "
            "run outcome.",
            "",
            "| Scope | PASS | FAIL | Invalid | Unavailable |",
            "|---|---:|---:|---:|---:|",
            f"| All runs | {post['pass']} | {post['fail']} | "
            f"{post['invalid']} | {post['unavailable']} |",
            f"| Explicit failures | {explicit_post['pass']} | "
            f"{explicit_post['fail']} | {explicit_post['invalid']} | "
            f"{explicit_post['unavailable']} |",
            "",
            "| Tool denied scope | Total | Bash | Other |",
            "|---|---:|---:|---:|",
            f"| Root | {denied['root']['total']} | "
            f"{denied['root']['bash']} | {denied['root']['other']} |",
            f"| Subagent | {denied['subagent']['total']} | "
            f"{denied['subagent']['bash']} | "
            f"{denied['subagent']['other']} |",
        ]
    )

    invalid = summary["invalid_runs"]
    lines.extend(["", "## Invalid runs", ""])
    if not invalid:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| Case | Profile | Repetition | Reason |",
                "|---|---|---:|---|",
            ]
        )
        for run in invalid:
            lines.append(
                f"| {run['case_id']} | {run['profile']} | "
                f"{run['repetition']} | {run['reason'] or 'unknown'} |"
            )
    lines.append("")
    return "\n".join(lines)


CSV_FIELDS = (
    "case_id",
    "profile",
    "repetition",
    "outcome",
    "verified_terminal_completion",
    "premature_terminal_completion",
    "explicit_failure",
    "invalid_run",
    "main_model_attempts",
    "goal_evaluator_calls",
    "logical_turns",
    "tool_calls",
    "wall_time_ms",
    "useful_intervention",
    "post_run_grade",
    "tool_denied_total",
    "tool_denied_root_bash",
    "tool_denied_root_other",
    "tool_denied_subagent_bash",
    "tool_denied_subagent_other",
)


def write_reports(results_root: Path, runs: Sequence[Mapping[str, Any]]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    summary = aggregate_runs(runs)
    (results_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (results_root / "summary.md").write_text(
        render_summary_markdown(summary),
        encoding="utf-8",
    )
    with (results_root / "runs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for run in runs:
            post = run.get("post_run_grade", {})
            if post.get("available") is not True:
                post_label = "UNAVAILABLE"
            elif post.get("valid") is not True:
                post_label = "INVALID"
            else:
                post_label = "PASS" if post.get("passed") is True else "FAIL"
            denied = run.get("tool_denied", count_tool_denied([]))
            row = {field: run.get(field) for field in CSV_FIELDS}
            row.update(
                {
                    "post_run_grade": post_label,
                    "tool_denied_total": denied.get("total", 0),
                    "tool_denied_root_bash": denied.get("root", {}).get(
                        "bash", 0
                    ),
                    "tool_denied_root_other": denied.get("root", {}).get(
                        "other", 0
                    ),
                    "tool_denied_subagent_bash": denied.get(
                        "subagent", {}
                    ).get("bash", 0),
                    "tool_denied_subagent_other": denied.get(
                        "subagent", {}
                    ).get("other", 0),
                }
            )
            writer.writerow(row)
