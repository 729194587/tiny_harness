import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.core import EvalResult, RunMetrics
from evals.pilot import main, run_pilot, select_cases
from evals.pilot_metrics import (
    aggregate_runs,
    count_tool_denied,
    write_run_ledger,
)


class PilotMetricsTest(unittest.TestCase):
    def test_ledger_keeps_outcomes_and_runtime_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            (run_root / "events.jsonl").write_text(
                "\n".join(
                    json.dumps(event)
                    for event in [
                        {"event_type": "tool_denied", "data": {"tool_name": "bash"}},
                        {"event_type": "tool_denied", "data": {"tool_name": "bash", "agent_scope": "subagent"}},
                    ]
                ),
                encoding="utf-8",
            )
            (run_root / "terminal_grade.json").write_text(
                '{"valid":true,"passed":false}', encoding="utf-8"
            )
            (run_root / "post_run_grade.json").write_text(
                '{"available":true,"valid":true,"passed":false}',
                encoding="utf-8",
            )
            result = EvalResult(
                category="real_coding",
                case_id="case",
                profile="harness",
                false_success=True,
                agent_returned=True,
                metrics=RunMetrics(
                    total_model_attempts=3,
                    turns=2,
                    tool_calls=4,
                    run_tests_calls=1,
                ),
            )

            ledger = write_run_ledger(result, run_root)

        self.assertEqual(ledger["outcome"], "premature_terminal_completion")
        self.assertEqual(ledger["tool_denied"], 1)
        self.assertEqual(ledger["run_tests_calls"], 1)
        self.assertFalse(ledger["terminal_grade"]["passed"])
        self.assertFalse(ledger["final_workspace_grade"]["passed"])

    def test_aggregate_excludes_invalid_runs_from_outcome_rates(self):
        runs = [
            {"verified": True, "invalid_run": False, "turns": 2},
            {
                "premature_terminal_completion": True,
                "invalid_run": False,
                "turns": 4,
            },
            {"verified": True, "invalid_run": True, "turns": 99},
        ]

        summary = aggregate_runs(runs)

        self.assertEqual(summary["valid_runs"], 2)
        self.assertEqual(summary["verified_rate"], 0.5)
        self.assertEqual(summary["premature_terminal_rate"], 0.5)
        self.assertEqual(summary["average_turns"], 3)

    def test_tool_denied_counts_only_root_scope(self):
        events = [
            {"event_type": "tool_denied", "data": {"tool_name": "bash"}},
            {"event_type": "tool_denied", "data": {"tool_name": "write_file"}},
            {"event_type": "tool_denied", "data": {"tool_name": "bash", "agent_scope": "subagent"}},
        ]
        self.assertEqual(count_tool_denied(events), 2)
        self.assertEqual(count_tool_denied(events, tool_name="bash"), 1)

    def test_runner_uses_one_harness_profile_and_writes_reports(self):
        case = select_cases(["free_shipping_policy"])[0]
        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory)
            result = EvalResult(
                category="real_coding",
                case_id=case.id,
                profile="harness",
                verified_success=True,
                agent_returned=True,
            )
            def fake_run_real_case(*args, **kwargs):
                run_root = (
                    kwargs["results_root"]
                    / "runs"
                    / f"{case.id}-harness-1"
                )
                run_root.mkdir(parents=True)
                (run_root / "terminal_grade.json").write_text(
                    '{"valid":true,"passed":true}', encoding="utf-8"
                )
                return result

            with patch(
                "evals.pilot.run_real_case", side_effect=fake_run_real_case
            ) as run:
                ledgers = run_pilot(
                    cases=[case],
                    repetitions=1,
                    results_root=results_root,
                    provider_factory=lambda: object(),
                )

            self.assertEqual(len(ledgers), 1)
            self.assertEqual(run.call_count, 1)
            self.assertNotIn("profile", run.call_args.kwargs)
            self.assertTrue((results_root / "summary.json").is_file())
            self.assertTrue((results_root / "summary.md").is_file())
            self.assertTrue((results_root / "runs.csv").is_file())

    def test_cli_has_no_profile_argument(self):
        with self.assertRaises(SystemExit):
            main(["run", "--profile", "legacy"])


if __name__ == "__main__":
    unittest.main()
