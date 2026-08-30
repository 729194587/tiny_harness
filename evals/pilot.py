"""Run and offline-summarize the Completion Verification Pilot."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from evals.core import EvalCase, EvalResult, RunMetrics, load_cases
from evals.pilot_metrics import (
    PROPOSAL_RECORDS,
    load_run_ledgers,
    write_reports,
    write_run_ledger,
)
from evals.run import REAL_PROFILES, balanced_profile_order, run_real_case
from tiny_harness.__main__ import DEFAULT_BASE_URL, DEFAULT_MODEL
from tiny_harness.models.chat_completions import ChatCompletionsProvider

ROOT = Path(__file__).resolve().parent
PILOT_CASES = ROOT / "pilot_cases.json"
PILOT_FIXTURES = ROOT / "pilot_fixtures"
DEFAULT_RESULTS = ROOT / "results" / "pilot"

ProviderFactory = Callable[[], ChatCompletionsProvider]


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _default_provider_factory() -> ChatCompletionsProvider:
    api_key = os.getenv("TINYHARNESS_API_KEY")
    if not api_key:
        raise RuntimeError("TINYHARNESS_API_KEY is required to run the Pilot")
    return ChatCompletionsProvider(
        api_key=api_key,
        model=os.getenv("TINYHARNESS_MODEL", DEFAULT_MODEL),
        base_url=os.getenv("TINYHARNESS_BASE_URL", DEFAULT_BASE_URL),
    )


def select_cases(case_ids: Sequence[str] | None) -> list[EvalCase]:
    cases = load_cases(PILOT_CASES)
    if not case_ids:
        return cases
    selected = set(case_ids)
    chosen = [case for case in cases if case.id in selected]
    missing = selected - {case.id for case in chosen}
    if missing:
        raise ValueError("Unknown Pilot tasks: " + ", ".join(sorted(missing)))
    return chosen


def _infra_result(
    case: EvalCase,
    profile: str,
    repetition: int,
    error: Exception,
) -> EvalResult:
    return EvalResult(
        category="real_coding",
        case_id=case.id,
        profile=profile,
        repetition=repetition,
        invalid_run=True,
        error_type=type(error).__name__,
        metrics=RunMetrics(),
    )


def run_pilot(
    *,
    cases: Sequence[EvalCase],
    profiles: Sequence[str],
    repetitions: int,
    results_root: Path,
    provider_factory: ProviderFactory = _default_provider_factory,
) -> list[dict]:
    """Run selected Pilot cells and persist each ledger immediately."""

    ledgers = []
    for case_index, case in enumerate(cases):
        for repetition in range(1, repetitions + 1):
            ordered_profiles = balanced_profile_order(
                profiles,
                case_index=case_index,
                repetition=repetition,
            )
            for profile in ordered_profiles:
                run_root = (
                    results_root
                    / "runs"
                    / f"{case.id}-{profile}-{repetition}"
                )
                if run_root.exists():
                    raise FileExistsError(
                        f"Pilot run directory already exists: {run_root}"
                    )
                try:
                    provider = provider_factory()
                    result = run_real_case(
                        case,
                        profile=profile,
                        repetition=repetition,
                        provider=provider,
                        fixtures_root=PILOT_FIXTURES,
                        results_root=results_root,
                    )
                except Exception as error:
                    run_root.mkdir(parents=True, exist_ok=True)
                    proposal_path = run_root / PROPOSAL_RECORDS
                    if not proposal_path.exists():
                        proposal_path.write_text("[]\n", encoding="utf-8")
                    result = _infra_result(
                        case,
                        profile,
                        repetition,
                        error,
                    )
                ledger = write_run_ledger(result, run_root)
                ledgers.append(ledger)
                write_reports(results_root, ledgers)
    return ledgers


def summarize_existing(results_root: Path) -> list[dict]:
    ledgers = load_run_ledgers(results_root)
    write_reports(results_root, ledgers)
    return ledgers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or summarize the Goal Gate Completion Pilot"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run Pilot tasks")
    run_parser.add_argument(
        "--task",
        action="append",
        dest="task_ids",
        help="Pilot task id; repeat to select multiple tasks",
    )
    run_parser.add_argument(
        "--profile",
        choices=("both", *REAL_PROFILES),
        default="both",
    )
    run_parser.add_argument("--repetitions", type=_positive_int, default=1)
    run_parser.add_argument("--results-dir", type=Path)

    summarize_parser = subparsers.add_parser(
        "summarize", help="rebuild reports without model/API calls"
    )
    summarize_parser.add_argument("results_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "summarize":
        results_root = args.results_dir.resolve()
        ledgers = summarize_existing(results_root)
        print(f"Summarized {len(ledgers)} runs in {results_root}")
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_root = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else (DEFAULT_RESULTS / stamp).resolve()
    )
    cases = select_cases(args.task_ids)
    profiles = REAL_PROFILES if args.profile == "both" else (args.profile,)
    ledgers = run_pilot(
        cases=cases,
        profiles=profiles,
        repetitions=args.repetitions,
        results_root=results_root,
    )
    print(f"Pilot reports written to {results_root}")
    return 1 if any(ledger["invalid_run"] for ledger in ledgers) else 0


if __name__ == "__main__":
    raise SystemExit(main())
