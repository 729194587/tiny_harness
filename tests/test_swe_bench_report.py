import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.swe_bench_lite.__main__ import main
from evals.swe_bench_lite.report import analyze_run, compare_runs


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, events):
        root = self.root / name
        directory = root / "tasks" / "instance" / "rollout"
        directory.mkdir(parents=True)
        (directory / "events.jsonl").write_text("".join(
            json.dumps({"run_id": "run", "event_type": kind, "data": data}) + "\n"
            for kind, data in events), encoding="utf-8")
        return root

    def test_metrics_retry_child_usage_and_safe_attribution(self):
        request = {"turn": 1, "purpose": "main", "context_tokens": 120,
                   "context_attribution": {"categories": {"tool_schemas": {"estimated_tokens": 20}}}}
        response = {"turn": 1, "prompt_tokens": 100, "completion_tokens": 10}
        root = self.write("a", [
            ("model_requested", request), ("model_requested", request),
            ("model_responded", response),
            ("tool_called", {"turn": 1}), ("tool_called", {"turn": 1}),
            ("tool_result", {"turn": 1, "tool_name": "edit_file", "outcome": "permission_denied"}),
            ("model_requested", {**request, "parent_tool_call_id": "child"}),
            ("model_responded", {**response, "parent_tool_call_id": "child"}),
            ("tool_called", {"turn": 1, "parent_tool_call_id": "child"}),
            ("tool_result", {"turn": 1, "tool_name": "edit_file", "outcome": "returned"}),
            ("context_compacted", {"reason": "working", "turn": 2,
                                   "strategy": "historical_result_clearing",
                                   "before_tokens": 200, "after_tokens": 120}),
        ])
        report = analyze_run(root)
        self.assertEqual(report["turns"], 2)
        self.assertEqual(report["total_tokens"], 220)
        self.assertEqual(report["tool_calls"], 3)
        self.assertEqual(report["tool_call_turns"], 2)
        self.assertEqual(report["multi_tool_turns"], 1)
        self.assertEqual(report["multi_tool_rate"], .5)
        self.assertEqual(report["calls_per_tool_turn"], 1.5)
        self.assertEqual(report["peak_request_context_tokens"], 120)
        self.assertEqual(report["peak_pre_prune_context_tokens"], 200)
        self.assertNotIn("peak_context", report)
        self.assertEqual(report["working_context_prune_events"], 1)
        self.assertEqual(report["prunes"][0]["strategy"], "historical_result_clearing")
        self.assertEqual(report["prunes"][0]["after_tokens"], 120)
        self.assertIsNone(report["first_workspace_mutation_turn"])
        self.assertEqual(report["first_returned_file_mutation_tool_turns"][0]["turn"], 1)
        self.assertEqual(report["context_attribution_summary"]["categories"]["tool_schemas"],
                         {"sum_estimated_tokens": 60, "peak_estimated_tokens": 20})
        with patch("evals.swe_bench_lite.__main__.ChatCompletionsProvider") as provider:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["report", str(root)]), 0)
            self.assertEqual(json.loads(output.getvalue())["turns"], 2)
            provider.assert_not_called()

    def test_comparison_unknown_usage_and_zero_baseline(self):
        a = self.write("a", [])
        b = self.write("b", [("model_requested", {"turn": 1}),
                              ("model_responded", {"prompt_tokens": 20}),
                              ("model_responded", {})])
        self.assertIsNone(analyze_run(b)["prompt_tokens"])
        result = compare_runs(a, b)["metrics"]
        self.assertEqual(result["turns"], {"run_a": 0, "run_b": 1, "delta": 1, "percent_change": None})
        self.assertEqual(result["tool_calls"]["percent_change"], 0)
        self.assertIsNone(result["total_tokens"]["delta"])
        c = self.write("c", [("model_requested", {"turn": 1}), ("model_requested", {"turn": 2})])
        self.assertEqual(compare_runs(b, c)["metrics"]["turns"]["percent_change"], 100)
        with patch("evals.swe_bench_lite.__main__.ChatCompletionsProvider") as provider:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["compare", str(a), str(b)]), 0)
            provider.assert_not_called()

    def test_missing_and_corrupt_artifacts_are_explicit_errors(self):
        with self.assertRaises(ValueError):
            analyze_run(self.root)
        root = self.write("bad", [])
        next(root.rglob("events.jsonl")).write_text('{"data":', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "events.jsonl:1"):
            analyze_run(root)

    def test_peak_requests_prunes_and_provider_usage_are_separate(self):
        root = self.write("separate", [
            ("context_compacted", {"reason": "working", "before_tokens": 30000, "after_tokens": 14000}),
            ("model_requested", {"context_tokens": 14000}),
            ("model_requested", {"context_attribution": {
                "estimated_tokens": 15000, "calibration_adjustment_tokens": 500,
            }}),
            ("model_responded", {"prompt_tokens": 90000}),
            ("context_compacted", {"reason": "automatic", "before_tokens": 110000, "after_tokens": 50000}),
        ])
        report = analyze_run(root)
        self.assertEqual(report["peak_request_context_tokens"], 15500)
        self.assertEqual(report["peak_pre_prune_context_tokens"], 30000)
        self.assertEqual(report["prompt_tokens"], 90000)
        empty = self.write("empty", [("model_responded", {"prompt_tokens": 1000})])
        self.assertIsNone(analyze_run(empty)["peak_request_context_tokens"])
        self.assertIsNone(analyze_run(empty)["peak_pre_prune_context_tokens"])
        comparison = compare_runs(empty, root)["metrics"]
        self.assertIsNone(comparison["peak_pre_prune_context_tokens"]["delta"])
        self.assertEqual(comparison["peak_request_context_tokens"]["run_b"], 15500)
        self.assertNotIn("peak_context", comparison)
