import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evals.swe_bench_lite.__main__ import _parser, main
from evals.swe_bench_lite.calibration import (
    CALIBRATED,
    CALIBRATION_FAILED,
    calibrate_task,
)
from evals.swe_bench_lite.data import (
    SweEvaluationBundle,
    SweTask,
    load_agent_tasks,
    load_evaluation_bundles,
)
from evals.swe_bench_lite.docker_workspace import DockerTaskEnvironment
from evals.swe_bench_lite.evaluator import run_official_evaluation
from evals.swe_bench_lite.pipeline import rollout_task


def task():
    return SweTask("owner__repo-1", "owner/repo", "abcdef1", "PUBLIC ISSUE", "image:latest")


def bundle():
    return SweEvaluationBundle(
        instance_id="owner__repo-1",
        repo="owner/repo",
        version="1.0",
        base_commit="abcdef1",
        patch="GOLD",
        test_patch="TESTS",
        fail_to_pass=("ftp",),
        pass_to_pass=("ptp",),
        eval_script="echo eval",
        log_parser="parse_log_pytest",
        eval_type="pass_and_fail",
    )


class FakeEnvironment:
    instances = []

    def __init__(self, value, *, network_mode=None):
        self.task = value
        self.network_mode = network_mode
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.shell_runner = object()
        self.commands = []
        self.closed = False
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def exec(self, command, check=True):
        self.commands.append((command, check))
        return SimpleNamespace(stdout=">>>>> Start Test Output\nlog\n", stderr="", returncode=0)

    def collect_patch(self):
        return "MODEL PATCH"

    def __exit__(self, *args):
        self.closed = True
        self.temporary.cleanup()


class SweDataBoundaryTest(unittest.TestCase):
    def test_agent_loader_has_no_evaluator_fields(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "tasks.jsonl"
        record = {
            "instance_id": "i", "repo": "r", "base_commit": "abc",
            "problem_statement": "public", "image": "img", "version": "1",
            "patch": "secret gold", "test_patch": "secret tests",
            "FAIL_TO_PASS": ["ftp"], "PASS_TO_PASS": ["ptp"],
            "hints_text": "secret hint", "eval_script": "secret script",
            "log_parser": "secret parser", "eval_type": "secret eval type",
        }
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        agent_task = load_agent_tasks(path)[0]
        self.assertEqual(
            set(agent_task.__dataclass_fields__),
            {"instance_id", "repo", "base_commit", "problem_statement", "image"},
        )
        serialized = repr(agent_task)
        for secret in (
            "secret gold", "secret tests", "ftp", "ptp", "secret hint",
            "secret script", "secret parser", "secret eval type",
        ):
            self.assertNotIn(secret, serialized)
        self.assertEqual(load_evaluation_bundles(path)[0].patch, "secret gold")

    def test_rollout_api_does_not_load_evaluator_bundle_or_leak_prompt(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / "out"
        observed = {}

        def agent(provider, workspace, messages, **options):
            observed.update(messages=messages, options=options, workspace=workspace)
            return "done"

        with patch(
            "evals.swe_bench_lite.data.load_evaluation_bundles",
            side_effect=AssertionError("must not load evaluator data"),
        ):
            result = rollout_task(
                task(), object(), output, model_name_or_path="model",
                environment_factory=FakeEnvironment, agent_entrypoint=agent,
            )
        self.assertEqual(result.model_patch, "MODEL PATCH")
        self.assertEqual(FakeEnvironment.instances[-1].network_mode, "none")
        self.assertEqual(observed["options"]["max_context_tokens"], 125_000)
        prompt = json.dumps(observed["messages"])
        self.assertIn("PUBLIC ISSUE", prompt)
        self.assertIn("Use relative paths with all tools", prompt)
        self.assertNotIn("/testbed", prompt)
        for secret in ("GOLD", "TESTS", "FAIL_TO_PASS", "PASS_TO_PASS", "hints_text"):
            self.assertNotIn(secret, prompt)
        self.assertTrue(FakeEnvironment.instances[-1].closed)

    def test_rollout_can_disable_context_compaction(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        observed = {}

        def agent(provider, workspace, messages, **options):
            del provider, workspace, messages
            observed.update(options)
            return "done"

        rollout_task(
            task(),
            object(),
            Path(temporary.name) / "out",
            model_name_or_path="model",
            max_context_tokens=None,
            environment_factory=FakeEnvironment,
            agent_entrypoint=agent,
        )

        self.assertIsNone(observed["max_context_tokens"])


class SweCliContextBudgetTest(unittest.TestCase):
    def test_default_matches_tinyharness_cli(self):
        args = _parser().parse_args(["run"])
        self.assertEqual(args.max_context_tokens, 125_000)

    def test_explicit_context_budget(self):
        args = _parser().parse_args(
            ["run", "--max-context-tokens", "64000"]
        )
        self.assertEqual(args.max_context_tokens, 64_000)

    def test_context_compaction_can_be_disabled(self):
        args = _parser().parse_args(["run", "--no-context-compaction"])
        self.assertIsNone(args.max_context_tokens)

    def test_disabled_context_compaction_reaches_pipeline(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        expected_run_dir = Path(temporary.name) / "run"
        with (
            patch.dict(os.environ, {"TINYHARNESS_API_KEY": "test-key"}),
            patch(
                "evals.swe_bench_lite.__main__.ChatCompletionsProvider",
                return_value=object(),
            ),
            patch(
                "evals.swe_bench_lite.__main__.run_selected_smoke",
                return_value=expected_run_dir,
            ) as run_selected,
        ):
            result = main(["run", "--no-context-compaction"])

        self.assertEqual(result, 0)
        self.assertIsNone(run_selected.call_args.kwargs["max_context_tokens"])


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        FakeEnvironment.instances = []
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def _grader(self, values):
        def grade(task_value, bundle_value, log_path, label):
            del task_value, bundle_value, log_path
            return {"tests_status": values[label]}
        return grade

    def test_only_expected_baseline_and_gold_states_calibrate(self):
        values = {
            "baseline": {
                "FAIL_TO_PASS": {"success": [], "failure": ["ftp"]},
                "PASS_TO_PASS": {"success": ["ptp"], "failure": []},
            },
            "gold": {
                "FAIL_TO_PASS": {"success": ["ftp"], "failure": []},
                "PASS_TO_PASS": {"success": ["ptp"], "failure": []},
            },
        }
        result = calibrate_task(
            task(), bundle(), Path(self.temporary.name) / "cal",
            environment_factory=FakeEnvironment, grader=self._grader(values),
        )
        self.assertEqual(result.status, CALIBRATED)
        self.assertEqual(len(FakeEnvironment.instances), 2)
        self.assertTrue(
            all(item.network_mode is None for item in FakeEnvironment.instances)
        )
        self.assertTrue(all(item.closed for item in FakeEnvironment.instances))
        self.assertTrue(any("git apply" in cmd for cmd, _ in FakeEnvironment.instances[1].commands))

    def test_any_oracle_mismatch_fails_calibration(self):
        values = {
            "baseline": {
                "FAIL_TO_PASS": {"success": ["ftp"], "failure": []},
                "PASS_TO_PASS": {"success": ["ptp"], "failure": []},
            },
            "gold": {
                "FAIL_TO_PASS": {"success": ["ftp"], "failure": []},
                "PASS_TO_PASS": {"success": ["ptp"], "failure": []},
            },
        }
        result = calibrate_task(
            task(), bundle(), Path(self.temporary.name) / "cal",
            environment_factory=FakeEnvironment, grader=self._grader(values),
        )
        self.assertEqual(result.status, CALIBRATION_FAILED)

    def test_exception_still_closes_environment(self):
        def broken(*args):
            raise RuntimeError("grader failed")
        result = calibrate_task(
            task(), bundle(), Path(self.temporary.name) / "cal",
            environment_factory=FakeEnvironment, grader=broken,
        )
        self.assertEqual(result.status, CALIBRATION_FAILED)
        self.assertTrue(FakeEnvironment.instances[0].closed)


class PatchCollectionTest(unittest.TestCase):
    def test_patch_includes_modified_deleted_untracked_binary_and_excludes_runtime(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        workspace = Path(temporary.name)
        subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
        (workspace / "modified.txt").write_text("before\n", encoding="utf-8")
        (workspace / "deleted.txt").write_text("delete\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=workspace, check=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "base"],
            cwd=workspace, check=True,
        )
        base_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        with (workspace / ".git" / "info" / "exclude").open("a", encoding="utf-8") as stream:
            stream.write("\n.tinyharness/\n")
        (workspace / "modified.txt").write_text("after\n", encoding="utf-8")
        (workspace / "deleted.txt").unlink()
        (workspace / "new.txt").write_text("new\n", encoding="utf-8")
        (workspace / "new.bin").write_bytes(b"\x00\x01\x02\xff")
        (workspace / ".tinyharness").mkdir()
        (workspace / ".tinyharness" / "events.jsonl").write_text("secret", encoding="utf-8")

        class LocalEnvironment:
            def exec(self, command, check=True):
                return subprocess.run(
                    command, shell=True, cwd=workspace, check=check,
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                )

        environment = LocalEnvironment()
        environment.task = SweTask(
            "owner__repo-1", "owner/repo", base_commit,
            "PUBLIC ISSUE", "image:latest",
        )
        patch_text = DockerTaskEnvironment.collect_patch(environment)
        self.assertIn("modified.txt", patch_text)
        self.assertIn("deleted.txt", patch_text)
        self.assertIn("new.txt", patch_text)
        self.assertIn("new.bin", patch_text)
        self.assertIn("GIT binary patch", patch_text)
        self.assertNotIn(".tinyharness", patch_text)

    def test_patch_uses_base_commit_and_combines_committed_and_uncommitted_changes(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        workspace = Path(temporary.name)
        subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
        (workspace / "committed.txt").write_text("base\n", encoding="utf-8")
        (workspace / "working.txt").write_text("base\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=workspace, check=True)
        subprocess.run(
            [
                "git", "-c", "user.name=Test", "-c",
                "user.email=test@example.com", "commit", "-qm", "base",
            ],
            cwd=workspace,
            check=True,
        )
        base_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        (workspace / "committed.txt").write_text("committed change\n", encoding="utf-8")
        subprocess.run(["git", "add", "committed.txt"], cwd=workspace, check=True)
        subprocess.run(
            [
                "git", "-c", "user.name=Test", "-c",
                "user.email=test@example.com", "commit", "-qm", "agent commit",
            ],
            cwd=workspace,
            check=True,
        )
        (workspace / "working.txt").write_text("uncommitted change\n", encoding="utf-8")

        class LocalEnvironment:
            task = SweTask(
                "owner__repo-1", "owner/repo", base_commit,
                "PUBLIC ISSUE", "image:latest",
            )

            def exec(self, command, check=True):
                return subprocess.run(
                    command,
                    shell=True,
                    cwd=workspace,
                    check=check,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

        patch_text = DockerTaskEnvironment.collect_patch(LocalEnvironment())
        self.assertIn("committed.txt", patch_text)
        self.assertIn("+committed change", patch_text)
        self.assertIn("working.txt", patch_text)
        self.assertIn("+uncommitted change", patch_text)


class DockerNetworkPolicyTest(unittest.TestCase):
    def _run_calls(self, network_mode):
        environment = DockerTaskEnvironment(task(), network_mode=network_mode)
        calls = []

        def fake_run(argv, **kwargs):
            del kwargs
            calls.append(argv)
            if argv[1] == "cp":
                assert environment.workspace is not None
                (environment.workspace / ".git" / "info").mkdir(
                    parents=True, exist_ok=True
                )
            return subprocess.CompletedProcess(argv, 0, "", "")

        with patch(
            "evals.swe_bench_lite.docker_workspace._run", side_effect=fake_run
        ):
            with environment:
                pass
        return calls

    def test_agent_network_none_is_in_docker_run_argv(self):
        calls = self._run_calls("none")
        argv = next(argv for argv in calls if argv[1] == "run")
        self.assertEqual(argv[argv.index("--network") + 1], "none")

    def test_default_environment_leaves_docker_network_unspecified(self):
        calls = self._run_calls(None)
        argv = next(argv for argv in calls if argv[1] == "run")
        self.assertNotIn("--network", argv)

    def test_workspace_disables_file_mode_tracking_locally(self):
        calls = self._run_calls(None)
        setup_command = next(
            argv[-1]
            for argv in calls
            if argv[1] == "exec" and "git reset --hard" in argv[-1]
        )
        self.assertIn("git config --local core.fileMode false", setup_command)


class LifecycleAndEvaluatorTest(unittest.TestCase):
    def test_agent_exception_still_closes_task_environment(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        FakeEnvironment.instances = []

        def broken_agent(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("model failed")

        with self.assertRaisesRegex(RuntimeError, "model failed"):
            rollout_task(
                task(),
                object(),
                Path(temporary.name) / "out",
                model_name_or_path="model",
                environment_factory=FakeEnvironment,
                agent_entrypoint=broken_agent,
            )
        self.assertTrue(FakeEnvironment.instances[-1].closed)
        metadata = json.loads(
            (Path(temporary.name) / "out" / "metadata.json").read_text()
        )
        self.assertEqual(metadata["status"], "FAILED")
        self.assertEqual(metadata["error_type"], "RuntimeError")

    def test_setup_failure_attempts_container_cleanup(self):
        environment = DockerTaskEnvironment(task())
        calls = []

        def fail_create(argv, **kwargs):
            calls.append(argv)
            if argv[1] == "create":
                raise subprocess.CalledProcessError(1, argv)
            return subprocess.CompletedProcess(argv, 0, "", "")

        with patch("evals.swe_bench_lite.docker_workspace._run", side_effect=fail_create):
            with self.assertRaises(subprocess.CalledProcessError):
                environment.__enter__()
        removed = [argv for argv in calls if argv[1:3] == ["rm", "--force"]]
        self.assertEqual(len(removed), 2)
        self.assertIsNone(environment.workspace)

    def test_official_evaluator_uses_serial_cli_and_unique_output(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        predictions = root / "predictions.jsonl"
        dataset = root / "selected.jsonl"
        predictions.write_text("{}\n", encoding="utf-8")
        dataset.write_text("{}\n", encoding="utf-8")
        completed = subprocess.CompletedProcess([], 0, "official output", "")
        with patch("evals.swe_bench_lite.evaluator.subprocess.run", return_value=completed) as run:
            result = run_official_evaluation(
                predictions, dataset, root, instance_ids=("one",), evaluation_run_id="eval-1"
            )
        argv = run.call_args.args[0]
        self.assertIn("swebench.harness.run_evaluation", argv)
        self.assertEqual(argv[argv.index("--max_workers") + 1], "1")
        self.assertEqual(argv[argv.index("--run_id") + 1], "eval-1")
        self.assertNotIn("--clean", argv)
        self.assertEqual(argv[-2:], ["--instance_ids", "one"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("official output", (result.output_dir / "harness.log").read_text())


if __name__ == "__main__":
    unittest.main()
