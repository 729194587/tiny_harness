"""CLI for the first SWE-bench Lite Dev selected-task pipeline."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from tiny_harness.__main__ import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MODEL,
)
from tiny_harness.models.chat_completions import ChatCompletionsProvider

from .calibration import CALIBRATED, CALIBRATION_FAILED, CalibrationResult, calibrate_task
from .data import load_agent_tasks, load_evaluation_bundles
from .evaluator import new_run_id, run_official_evaluation
from .pipeline import (
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SELECTED_TASKS,
    run_selected_smoke,
    select_instances,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evals.swe_bench_lite")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "run"):
        command = subparsers.add_parser(name)
        selection = command.add_mutually_exclusive_group()
        selection.add_argument("--instance-id")
        if name == "calibrate":
            selection.add_argument(
                "--all-selected", action="store_true",
                help="Calibrate all selected smoke tasks sequentially (the default)",
            )
        command.add_argument("--selected", type=Path, default=DEFAULT_SELECTED_TASKS)
        command.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
        command.add_argument("--run-id")
    run = subparsers.choices["run"]
    run.add_argument("--max-turns", type=int, default=20)
    run.add_argument("--subagent-max-turns", type=int, default=10)
    context_group = run.add_mutually_exclusive_group()
    context_group.add_argument(
        "--max-context-tokens",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TOKENS,
    )
    context_group.add_argument(
        "--no-context-compaction",
        dest="max_context_tokens",
        action="store_const",
        const=None,
    )
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("--dataset", type=Path, default=DEFAULT_SELECTED_TASKS)
    evaluate.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    evaluate.add_argument("--instance-id", action="append", default=[])
    evaluate.add_argument("--evaluation-run-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "evaluate":
        result = run_official_evaluation(
            args.predictions,
            args.dataset,
            args.results_root,
            instance_ids=tuple(args.instance_id),
            evaluation_run_id=args.evaluation_run_id,
        )
        print(result.output_dir)
        return 0
    if args.command == "calibrate":
        run_id = args.run_id or new_run_id("calibration")
        run_dir = (args.results_root / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        tasks = select_instances(load_agent_tasks(args.selected), args.instance_id)
        bundles = {
            bundle.instance_id: bundle
            for bundle in select_instances(
                load_evaluation_bundles(args.selected), args.instance_id
            )
        }
        results = []
        for task in tasks:
            try:
                result = calibrate_task(
                    task,
                    bundles[task.instance_id],
                    run_dir / "tasks" / task.instance_id / "calibration",
                )
            except Exception as error:
                # Include failures outside calibrate_task's guard, e.g. output I/O.
                result = CalibrationResult(
                    task.instance_id, CALIBRATION_FAILED,
                    False, False, False, False, type(error).__name__,
                )
            results.append(result)
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "benchmark": "SWE-bench Lite Dev reference calibration",
                    "results": [asdict(result) for result in results],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        rows = [
            {**asdict(result), "status": "ERROR" if result.error_type else result.status}
            for result in results
        ]
        counts = {
            status: sum(row["status"] == status for row in rows)
            for status in (CALIBRATED, CALIBRATION_FAILED, "ERROR")
        }
        exit_code = 2 if counts["ERROR"] else (
            0 if rows and counts[CALIBRATED] == len(rows) else 1
        )
        summary = {
            "status": "ERROR" if exit_code == 2 else (
                CALIBRATED if exit_code == 0 else CALIBRATION_FAILED
            ),
            "counts": counts,
            "results": rows,
        }
        summary_path = run_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        columns = (
            "instance_id", "status", "baseline_ftp_failed", "baseline_ptp_passed",
            "gold_ftp_passed", "gold_ptp_passed", "error_type",
        )
        print("\t".join(columns))
        for row in rows:
            print("\t".join(str(row[column]) if row[column] is not None else "-" for column in columns))
        print(f"汇总 JSON: {summary_path}")
        return exit_code
    api_key = os.getenv("TINYHARNESS_API_KEY")
    if not api_key:
        raise SystemExit("TINYHARNESS_API_KEY is required for run")
    model = os.getenv("TINYHARNESS_MODEL", DEFAULT_MODEL)
    provider = ChatCompletionsProvider(
        api_key=api_key,
        model=model,
        base_url=os.getenv("TINYHARNESS_BASE_URL", DEFAULT_BASE_URL),
    )
    run_dir = run_selected_smoke(
        provider,
        model_name_or_path=model,
        selected_path=args.selected,
        results_root=args.results_root,
        run_id=args.run_id,
        instance_id=args.instance_id,
        max_turns=args.max_turns,
        subagent_max_turns=args.subagent_max_turns,
        max_context_tokens=args.max_context_tokens,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
