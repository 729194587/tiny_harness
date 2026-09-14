import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.swe_bench_lite.__main__ import main
from evals.swe_bench_lite.evaluator import OfficialEvaluationResult
from evals.swe_bench_lite.finalizer import finalize_run, locate_run, render_summary


class FinalizerTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "run with spaces"
        self.root.mkdir()
        self.write(self.root / "predictions.jsonl", {"instance_id": "one", "model_patch": "patch"})
        self.write(self.root / "data" / "tasks.jsonl", {"instance_id": "one"})
        self.write(self.root / "metadata.json", {"selected_tasks_path": "data\\tasks.jsonl", "tasks": [{"instance_id": "one"}]})
        self.events([])

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def events(self, rows):
        path = self.root / "tasks" / "one" / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps({"event_type": kind, "data": data}) + "\n"
                                for kind, data in rows), encoding="utf-8")

    def evaluator(self, report, returncode=0):
        def evaluate(predictions, dataset, results_root, **kwargs):
            self.assertEqual(predictions, self.root / "predictions.jsonl")
            self.assertEqual(dataset, self.root / "data" / "tasks.jsonl")
            self.assertEqual(kwargs["instance_ids"], ("one",))
            output = results_root / "official_evaluation" / "eval"
            self.write(output / "metadata.json", {"predictions_path": str(predictions), "returncode": returncode})
            self.write(output / "logs" / "report.json", report)
            return OfficialEvaluationResult("eval", output, returncode)
        return evaluate

    def test_locate_paths_and_reject_multiple_tasks(self):
        predictions, dataset, instance = locate_run(self.root)
        self.assertEqual(instance, "one")
        self.assertTrue(predictions.is_file())
        self.assertEqual(dataset, self.root / "data" / "tasks.jsonl")
        self.write(self.root / "tasks" / "two" / "metadata.json", {"instance_id": "two"})
        with self.assertRaises(ValueError):
            locate_run(self.root)

    def test_metrics_checkpoint_and_console_consistency(self):
        self.events([
            ("model_requested", {"turn": 1}), ("model_requested", {"turn": 1}),
            ("model_responded", {"prompt_tokens": 100, "completion_tokens": 10,
                                 "prompt_cache_hit_tokens": 20, "prompt_cache_miss_tokens": 80}),
            ("tool_called", {"turn": 1}), ("tool_started", {"turn": 1}),
            ("workspace_observed", {"turn": 1, "root_turn": 1, "workspace_changed": True,
                                    "observation_status": "complete"}),
            ("model_requested", {"purpose": "summary"}),
            ("model_responded", {"purpose": "summary", "prompt_tokens": 300, "completion_tokens": 20,
                                 "prompt_cache_hit_tokens": 180, "prompt_cache_miss_tokens": 120}),
            ("context_compacted", {"reason": "working", "strategy": "historical_result_clearing"}),
            ("context_compacted", {"reason": "working", "strategy": "llm_task_state_checkpoint",
                                   "turn": 2, "before_tokens": 21000, "after_tokens": 4000}),
        ])
        with patch("evals.swe_bench_lite.finalizer.run_official_evaluation",
                   side_effect=self.evaluator({"one": {"resolved": True}})) as evaluate:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["finalize", str(self.root)]), 0)
            result = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(output.getvalue(), render_summary(result))
            self.assertEqual(output.getvalue(), (self.root / "summary.md").read_text(encoding="utf-8"))
            self.assertEqual(result["official_status"], "Resolved")
            for key, value in {"turns": 1, "tool_calls": 1, "first_workspace_mutation_turn": 1,
                               "prompt_tokens": 400, "completion_tokens": 30,
                               "total_cache_hit_tokens": 200, "total_cache_miss_tokens": 200,
                               "overall_cache_hit_rate": .5, "summary_cache_hit_rate": .6,
                               "checkpoint_count": 1}.items():
                self.assertEqual(result[key], value, key)
            checkpoint = result["checkpoints"][0]
            self.assertEqual((checkpoint["turn"], checkpoint["before_tokens"], checkpoint["after_tokens"]),
                             (2, 21000, 4000))
            self.assertTrue(finalize_run(self.root)["evaluation_reused"])
            evaluate.assert_called_once()

    def test_official_statuses_and_missing_cache(self):
        for report, code, expected in [
            ({"resolved_ids": ["one"]}, 0, "Resolved"),
            ({"unresolved_ids": ["one"]}, 0, "Unresolved"),
            ({"one": {"resolved": False}}, 0, "Unresolved"),
            ({"error_ids": ["one"], "unresolved_ids": ["one"]}, 0, "Evaluation Error"),
            ({}, 0, "Evaluation Error"),
            ({"one": {"resolved": False}}, 1, "Evaluation Error"),
        ]:
            with self.subTest(report=report, code=code):
                with patch("evals.swe_bench_lite.finalizer._existing", return_value=None), patch(
                    "evals.swe_bench_lite.finalizer.run_official_evaluation", side_effect=self.evaluator(report, code)
                ):
                    result = finalize_run(self.root)
                self.assertEqual(result["official_status"], expected)
                self.assertIsNone(result["overall_cache_hit_rate"])
                self.assertIsNone(result["summary_cache_hit_rate"])
                self.assertEqual(result["checkpoint_count"], 0)

    def test_exception_is_persisted_and_cli_returns_error(self):
        with patch("evals.swe_bench_lite.finalizer.run_official_evaluation", side_effect=RuntimeError("docker failed")):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["finalize", str(self.root)]), 2)
        result = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(result["official_status"], "Evaluation Error")
        self.assertIn("docker failed", result["evaluation_error"])

    def test_legacy_evaluation_reuse_and_changed_prediction(self):
        self.evaluator({"one": {"resolved": True}})(
            self.root / "predictions.jsonl", self.root / "data" / "tasks.jsonl", self.root.parent,
            instance_ids=("one",))
        with patch("evals.swe_bench_lite.finalizer.run_official_evaluation") as evaluate:
            self.assertTrue(finalize_run(self.root)["evaluation_reused"])
            evaluate.assert_not_called()
        # Persisted digest must also prevent reuse even when timestamps coincide.
        with patch("evals.swe_bench_lite.finalizer._existing", return_value=None), patch(
            "evals.swe_bench_lite.finalizer.run_official_evaluation", side_effect=self.evaluator({"resolved_ids": ["one"]})
        ):
            finalize_run(self.root)
        self.write(self.root / "predictions.jsonl", {"instance_id": "one", "model_patch": "new"})
        with patch("evals.swe_bench_lite.finalizer.run_official_evaluation", side_effect=RuntimeError("new evaluation")) as evaluate:
            self.assertEqual(finalize_run(self.root)["official_status"], "Evaluation Error")
            evaluate.assert_called_once()
