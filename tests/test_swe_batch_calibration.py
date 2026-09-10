import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evals.swe_bench_lite.__main__ import _parser, main
from evals.swe_bench_lite.calibration import CalibrationResult


MODULE = "evals.swe_bench_lite.__main__"


class BatchCalibrationTest(unittest.TestCase):
    def run_batch(self, outcomes, selection=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [SimpleNamespace(instance_id=f"task-{i}") for i in range(len(outcomes))]
            output = io.StringIO()
            with patch(f"{MODULE}.load_agent_tasks", return_value=tasks), \
                 patch(f"{MODULE}.load_evaluation_bundles", return_value=tasks), \
                 patch(f"{MODULE}.calibrate_task", side_effect=outcomes) as calibrate, \
                 patch(f"{MODULE}.ChatCompletionsProvider") as provider, \
                 patch(f"{MODULE}.run_selected_smoke") as rollout, \
                 redirect_stdout(output):
                code = main([
                    "calibrate", "--results-root", str(root), "--run-id", "batch",
                    *(selection if selection is not None else ["--all-selected"]),
                ])
            provider.assert_not_called()
            rollout.assert_not_called()
            for call, task in zip(calibrate.call_args_list, tasks):
                self.assertEqual(call.args, (
                    task, task, root / "batch" / "tasks" / task.instance_id / "calibration",
                ))
            self.assertEqual(calibrate.call_count, len(tasks))
            summary = json.loads((root / "batch" / "summary.json").read_text(encoding="utf-8"))
            metadata = json.loads((root / "batch" / "metadata.json").read_text(encoding="utf-8"))
            return code, summary, metadata, output.getvalue()

    def test_sequential_batch_isolates_errors_and_aggregates_all_outcomes(self):
        outcomes = [
            CalibrationResult("task-0", "CALIBRATED", True, True, True, True),
            PermissionError("cannot create task output"),
            CalibrationResult("task-2", "CALIBRATION_FAILED", False, True, True, True),
            CalibrationResult("task-3", "CALIBRATION_FAILED", False, False, False, False, "TimeoutExpired"),
            CalibrationResult("task-4", "CALIBRATED", True, True, True, True),
        ]
        code, summary, metadata, output = self.run_batch(outcomes)
        self.assertEqual(code, 2)
        self.assertEqual(summary["status"], "ERROR")
        self.assertEqual(summary["counts"], {"CALIBRATED": 2, "CALIBRATION_FAILED": 1, "ERROR": 2})
        rows = summary["results"]
        self.assertEqual([row["instance_id"] for row in rows], [f"task-{i}" for i in range(5)])
        self.assertEqual(rows[1]["error_type"], "PermissionError")
        self.assertEqual(rows[3]["status"], "ERROR")
        self.assertEqual(metadata["results"][3]["status"], "CALIBRATION_FAILED")
        self.assertEqual(
            [rows[2][key] for key in ("baseline_ftp_failed", "baseline_ptp_passed", "gold_ftp_passed", "gold_ptp_passed")],
            [False, True, True, True],
        )
        self.assertEqual(output.splitlines()[0].split("\t"), list(rows[0]))
        for row in rows:
            self.assertIn(row["instance_id"] + "\t" + row["status"], output)

    def test_exit_status_for_success_mismatch_and_empty_selection(self):
        for outcomes, expected in (
            ([CalibrationResult("task-0", "CALIBRATED", True, True, True, True)], 0),
            ([CalibrationResult("task-0", "CALIBRATION_FAILED", False, True, True, True)], 1),
            ([], 1),
        ):
            with self.subTest(expected=expected, outcomes=outcomes):
                code, summary, _, _ = self.run_batch(outcomes, selection=[])
                self.assertEqual(code, expected)
                self.assertEqual(summary["status"], "CALIBRATED" if code == 0 else "CALIBRATION_FAILED")

    def test_selection_options_are_mutually_exclusive(self):
        with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                _parser().parse_args(["calibrate", "--all-selected", "--instance-id", "task-0"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
