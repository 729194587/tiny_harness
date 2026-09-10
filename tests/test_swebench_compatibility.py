import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

from evals.swe_bench_lite.data import load_agent_tasks, load_evaluation_bundles


SWE_BENCH_AVAILABLE = importlib.util.find_spec("swebench") is not None
SELECTED_TASKS = (
    Path(__file__).resolve().parents[1]
    / "evals"
    / "swe_bench_lite"
    / "selected_tasks.jsonl"
)


@unittest.skipUnless(
    SWE_BENCH_AVAILABLE,
    "Install the swe-bench optional dependency to run compatibility smoke tests",
)
class SweBenchCompatibilityTest(unittest.TestCase):
    def test_installed_official_api_record_and_cli_are_compatible(self):
        import swebench
        from swebench.harness.grading import get_eval_report
        from swebench.harness.utils import make_test_spec

        self.assertEqual(swebench.__version__, "5.0.2")
        self.assertTrue(callable(get_eval_report))
        self.assertTrue(callable(make_test_spec))

        task = load_agent_tasks(SELECTED_TASKS)[0]
        bundle = load_evaluation_bundles(SELECTED_TASKS)[0]
        record = bundle.official_record()
        record["image"] = task.image
        self.assertEqual(
            set(record),
            {
                "instance_id",
                "image",
                "repo",
                "version",
                "FAIL_TO_PASS",
                "PASS_TO_PASS",
                "log_parser",
                "eval_type",
                "eval_script",
            },
        )
        test_spec = make_test_spec(record)

        self.assertEqual(test_spec.instance_id, task.instance_id)
        self.assertEqual(test_spec.repo, task.repo)
        self.assertEqual(test_spec.log_parser, bundle.log_parser)
        self.assertEqual(test_spec.eval_type, bundle.eval_type)

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "swebench.harness.run_evaluation",
                "--help",
            ],
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        help_text = completed.stdout + completed.stderr
        for option in (
            "--dataset_name",
            "--split",
            "--instance_ids",
            "--predictions_path",
            "--max_workers",
            "--run_id",
            "--report_dir",
        ):
            self.assertIn(option, help_text)
        self.assertNotIn("--clean", help_text)


if __name__ == "__main__":
    unittest.main()
