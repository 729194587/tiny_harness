import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from evals.swe_bench_lite import verified as v
from evals.swe_bench_lite.calibration import CalibrationResult
from evals.swe_bench_lite.data import _records, load_agent_tasks, load_evaluation_bundles


class VerifiedTests(unittest.TestCase):
    def test_fixed_selection_integrity_and_exclusions(self):
        manifest = json.loads(v.DEFAULT_SELECTION.with_name("manifest.json").read_text())
        rows = _records(v.DEFAULT_SELECTION)
        self.assertEqual(v.sha256(v.DEFAULT_SELECTION), manifest["selection_sha256"])
        self.assertEqual(len(rows), 30)
        self.assertEqual([r["instance_id"] for r in rows], manifest["instance_ids"])
        self.assertFalse(set(manifest["instance_ids"]) & set(manifest["excluded_instance_ids"]))
        for item in manifest["exclusions"]:
            self.assertEqual(v.text_sha256(v.ROOT / item["file"]), item["sha256"])
        self.assertEqual(v.select_rows(rows, set()), rows)

    def test_selection_ignores_answers_order_and_performance(self):
        rows = [{"instance_id": str(i), "patch": "answer"} for i in range(40)]
        expected = [r["instance_id"] for r in v.select_rows(rows, {"1"})]
        changed = [{**r, "patch": "different", "resolved": True} for r in reversed(rows)]
        self.assertEqual(expected, [r["instance_id"] for r in v.select_rows(changed, {"1"})])
        self.assertNotIn("1", expected)
        self.assertNotEqual(expected, [r["instance_id"] for r in v.select_rows(rows, {"1"}, seed=1)])
        with self.assertRaises(ValueError):
            v.select_rows(rows + [rows[0]], set())

    def test_agent_projection_does_not_contain_oracles(self):
        tasks = load_agent_tasks(v.DEFAULT_SELECTION)
        bundles = load_evaluation_bundles(v.DEFAULT_SELECTION)
        self.assertEqual(len(tasks), len(bundles))
        self.assertEqual(set(vars(tasks[0])), {"instance_id", "repo", "base_commit", "problem_statement", "image"})

    def test_wrong_source_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                v.prepare(v.DEFAULT_SELECTION, Path(tmp) / "out", [])
            self.assertFalse((Path(tmp) / "out").exists())

    def test_selection_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy(v.DEFAULT_SELECTION, root)
            shutil.copy(v.DEFAULT_SELECTION.with_name("manifest.json"), root)
            selection = root / "selected_tasks.jsonl"
            selection.write_bytes(selection.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                v.qualify(selection, root / "out")

    def qualify(self, environment_error=None, calibrated=True, official="PASSED", smoke=True):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(v, "DockerTaskEnvironment") as environment, \
                patch.object(v.subprocess, "run") as run, \
                patch.object(v, "calibrate_task") as calibrate, \
                patch.object(v, "grade_gold", return_value=official) as grade, \
                patch.object(v, "version", return_value="5.0.2"), \
                patch("openai.OpenAI", side_effect=AssertionError("Model API forbidden")):
            run.return_value = MagicMock(stdout='["image@sha256:digest"]')
            environment.return_value.__enter__.side_effect = environment_error
            calibrate.side_effect = lambda task, *a, **kw: CalibrationResult(
                task.instance_id, "CALIBRATED" if calibrated else "CALIBRATION_FAILED",
                calibrated, calibrated, calibrated, calibrated)
            result = v.qualify(v.DEFAULT_SELECTION, Path(tmp) / "out", smoke=smoke)
            self.assertEqual(len(result["results"]), 30)
            self.assertEqual(result["model_api_calls"], 0)
            return result, calibrate.call_count, grade.call_count

    def test_environment_failures_retained_no_replacement(self):
        result, calibrations, grades = self.qualify(environment_error=RuntimeError("docker unavailable"))
        self.assertEqual([r["status"] for r in result["results"]], ["ENVIRONMENT_FAILED"] * 3 + ["NOT_RUN"] * 27)
        self.assertEqual((calibrations, grades), (0, 0))

    def test_calibration_failures_are_distinct(self):
        result, calibrations, grades = self.qualify(calibrated=False)
        self.assertEqual([r["status"] for r in result["results"][:3]], ["CALIBRATION_FAILED"] * 3)
        self.assertEqual((calibrations, grades), (3, 0))

    def test_smoke_and_full_qualification(self):
        result, calibrations, grades = self.qualify()
        self.assertEqual((calibrations, grades), (3, 3))
        self.assertEqual(result["results"][0]["official_cli"], "PASSED")
        result, calibrations, grades = self.qualify(smoke=False)
        self.assertEqual((calibrations, grades), (30, 0))
        self.assertTrue(all(r["status"] == "RUNNABLE" for r in result["results"]))

    def test_cli_zero_exit_without_report_is_not_success(self):
        task = load_agent_tasks(v.DEFAULT_SELECTION)[0]
        bundle = load_evaluation_bundles(v.DEFAULT_SELECTION)[0]
        with tempfile.TemporaryDirectory() as tmp, patch.object(v, "run_official_evaluation") as evaluate:
            root = Path(tmp)
            evaluate.return_value = MagicMock(output_dir=root, returncode=0)
            self.assertEqual(v.grade_gold(task, bundle, v.DEFAULT_SELECTION, root), "FAILED")
            (root / "report.json").write_text(json.dumps({task.instance_id: {"resolved": True}}))
            self.assertEqual(v.grade_gold(task, bundle, v.DEFAULT_SELECTION, root), "PASSED")

    def test_official_installed_spec_and_grader_smoke(self):
        try:
            from swebench.harness.utils import make_test_spec, load_swebench_dataset
            from swebench.harness.grading import get_eval_report
        except ImportError:
            self.skipTest("swebench optional dependency is not installed")
        records = load_swebench_dataset(str(v.DEFAULT_SELECTION), split="test")
        self.assertEqual(len(records), 30)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "empty.log"
            log.write_text("")
            for row in records:
                spec = make_test_spec(row)
                self.assertEqual(spec.instance_id, row["instance_id"])
            # Missing test output must never become success; this is a parser
            # smoke check, not an actual benchmark/environment result.
            for row in records[:3]:
                report = get_eval_report(make_test_spec(row), {
                    "instance_id": row["instance_id"], "model_patch": row["patch"],
                    "model_name_or_path": "reference-gold"}, str(log), True)
                self.assertIs(report[row["instance_id"]]["resolved"], False)


if __name__ == "__main__":
    unittest.main()
