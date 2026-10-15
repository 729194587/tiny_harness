"""Offline acceptance tests; all Docker/model/grading execution is mocked."""
import copy
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch, Mock

from evals.swe_bench_lite import benchmark as b
from evals.swe_bench_lite.benchmark_report import report, official_status
from evals.swe_bench_lite.calibration import CalibrationResult, CALIBRATED, CALIBRATION_FAILED
from evals.swe_bench_lite.pipeline import RolloutResult, AGENT_SYSTEM_PROMPT
from evals.swe_bench_lite.evaluator import OfficialEvaluationResult, run_official_evaluation
from tiny_harness.models.base import ModelProviderError, ModelErrorKind

CONFIG = {"model": "offline-model", "model_version": "immutable-test-version",
          "base_url": "https://example.invalid/v1", "budgets": {
              "max_turns": 40, "subagent_max_turns": 10, "max_context_tokens": 125000}}
IMAGE = {"id": "sha256:" + "a" * 64, "repo_digests": ["example@sha256:" + "b" * 64],
         "architecture": "amd64", "os": "linux", "layers": ["sha256:" + "c" * 64]}


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.campaign = self.root / "campaign"
        # Preserve actual source inventory; avoid optional installed package dependency in unit fixtures.
        self.fp = {"source": {k: b.sha(v) for k, v in b.source_files().items()},
                   "grader": {"version": "fixture", "python_files_sha256": "fixture"},
                   "capabilities": {"tools": [], "skills": {}}, "git": {}}
        self.patcher = patch.object(b, "current_fingerprint", return_value=self.fp)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        b.prepare(self.campaign, copy.deepcopy(CONFIG))
        self.tasks = b.verify_bundle()[0]
        self.instance = self.tasks[0].instance_id
        self.provider = Mock(return_value=object())
        self.inspect = Mock(return_value=copy.deepcopy(IMAGE))

    def calibrated(self, task, bundle, directory, **kwargs):
        self.assertEqual(task.instance_id, bundle.instance_id)
        self.assertTrue(kwargs["official_script"])
        return CalibrationResult(task.instance_id, CALIBRATED, True, True, True, True)

    def rollout(self, task, provider, directory, **kwargs):
        self.assertEqual(set(task.__dataclass_fields__), set(b.AGENT_FIELDS))
        self.assertFalse(hasattr(task, "patch"))
        self.assertEqual(task.image, IMAGE["id"])
        self.assertEqual(kwargs["max_turns"], CONFIG["budgets"]["max_turns"])
        self.assertNotIn("profile", kwargs)
        directory.mkdir(parents=True)
        events = [
            ("model_requested", {"turn": 1}),
            ("model_request_failed", {"turn": 1, "error_kind": "server_unavailable"}),
            ("model_requested", {"turn": 1}),
            ("model_responded", {"turn": 1, "prompt_tokens": 100, "completion_tokens": 10}),
            ("tool_called", {"turn": 1, "tool_name": "code_outline"}),
            ("tool_started", {"turn": 1, "tool_name": "code_outline"}),
            ("workspace_observed", {"turn": 1, "root_turn": 1,
                "workspace_changed": True, "observation_status": "complete"}),
        ]
        (directory / "events.jsonl").write_text("".join(json.dumps({
            "event_type": kind, "data": data, "run_id": "fixture"
        }) + "\n" for kind, data in events))
        return RolloutResult(task.instance_id, "offline answer", "fixture patch")

    def run_one(self, **kwargs):
        return b.run_attempt(self.campaign, self.instance, self.provider,
            rollout=kwargs.pop("rollout", self.rollout),
            calibrator=kwargs.pop("calibrator", self.calibrated), image_inspector=self.inspect, **kwargs)

    def evaluator(self, predictions, dataset, results, **kwargs):
        self.assertEqual(kwargs["split"], "test")
        self.assertTrue(kwargs["local_images_only"])
        self.assertEqual(len(predictions.read_text().splitlines()), 1)
        self.assertEqual(json.loads(dataset.read_text())["image"], IMAGE["id"])
        output = results / "official_evaluation" / "fixture"
        output.mkdir(parents=True)
        b.write_new(output / "report.json", {self.instance: {"resolved": True}})
        return OfficialEvaluationResult("fixture", output, 0)

    def test_fixed_denominator_pending_and_per_repo(self):
        value = report(self.campaign)
        self.assertEqual(value["denominator"], 60)
        self.assertEqual(value["statuses"], {"pending_or_not_run": 60})
        self.assertEqual(value["resolved_rate"], 0)
        self.assertEqual(sum(r["tasks"] for r in value["per_repository"].values()), 60)
        self.provider.assert_not_called()

    def test_persist_grade_reuse_and_tamper_rejection(self):
        self.assertEqual(self.run_one()["status"], "completed")
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.run_one()
        grader = Mock(side_effect=self.evaluator)
        first = b.grade_attempt(self.campaign, self.instance, 1, evaluator=grader, image_inspector=self.inspect)
        second = b.grade_attempt(self.campaign, self.instance, 1, evaluator=grader, image_inspector=self.inspect)
        self.assertEqual(first, second)
        self.assertEqual(grader.call_count, 1)
        value = report(self.campaign)
        self.assertEqual(value["resolved_rate"], 1 / 60)
        self.assertEqual(value["physical_requests"], 2)
        self.assertEqual(value["service_failure_rate"], .5)
        self.assertIsNone(value["totals_all_attempts"]["prompt_tokens"])
        self.assertEqual(value["tasks"][0]["metrics"]["first_workspace_mutation_turn"], 1)
        attempt = self.campaign / "tasks" / self.instance / "attempts/0001"
        (attempt / "grading/official_evaluation/fixture/report.json").write_text("{}")
        with self.assertRaises(ValueError):
            report(self.campaign)

    def test_service_failure_manual_retry_only_and_costs_retained(self):
        fail = Mock(side_effect=ModelProviderError(ModelErrorKind.SERVER_UNAVAILABLE))
        self.assertEqual(self.run_one(rollout=fail)["status"], "infrastructure_failure")
        with self.assertRaises(FileNotFoundError):
            self.run_one()
        b.approve_retry(self.campaign, self.instance, "human", "Confirmed service outage")
        self.run_one()
        self.assertEqual(len(report(self.campaign)["attempts"]), 2)
        self.assertEqual(len(b.attempts(self.campaign / "tasks" / self.instance)), 2)

    def test_completed_or_agent_failed_cannot_be_retried(self):
        self.run_one(rollout=Mock(side_effect=ValueError("protocol failure")))
        with self.assertRaises(ValueError):
            b.approve_retry(self.campaign, self.instance, "human", "try again")
        self.assertEqual(report(self.campaign)["statuses"]["harness_or_protocol_failure"], 1)

    def test_missing_image_does_not_start_formal_attempt_or_api(self):
        self.inspect.side_effect = subprocess.CalledProcessError(1, ["docker", "image", "inspect"])
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_one()
        self.provider.assert_not_called()
        self.assertFalse(b.attempts(self.campaign / "tasks" / self.instance))

    def test_official_error_is_separate_and_no_implicit_regrade(self):
        self.run_one()
        grader = Mock(side_effect=RuntimeError("fixture grading failure"))
        value = b.grade_attempt(self.campaign, self.instance, 1, evaluator=grader, image_inspector=self.inspect)
        self.assertEqual(value["status"], "official_evaluation_error")
        b.grade_attempt(self.campaign, self.instance, 1, evaluator=grader, image_inspector=self.inspect)
        self.assertEqual(grader.call_count, 1)
        self.assertEqual(report(self.campaign)["statuses"]["official_evaluation_error"], 1)

    def test_sigkill_receipt_requires_manual_review_and_separate_approval(self):
        directory = self.campaign / "tasks" / self.instance
        attempt = directory / "attempts/0001"
        b.write_new(attempt / "receipt.json", {"instance_id": self.instance, "attempt": 1,
            "campaign_sha256": b.sha((self.campaign / "campaign.json").read_bytes()), "image": IMAGE})
        b.write_new(directory / "controller.lock/owner.json", {"pid": 123, "host": "fixture"})
        with self.assertRaises(FileExistsError):
            self.run_one()
        b.recover(self.campaign, self.instance, "human", "Host outage, worker stopped", confirm_stopped=True)
        with self.assertRaises(FileNotFoundError):
            self.run_one()
        b.approve_retry(self.campaign, self.instance, "human", "Infrastructure repaired")
        self.run_one()
        self.assertEqual(len(report(self.campaign)["attempts"]), 2)

    def test_interrupted_attempt_is_not_replayed_and_review_is_preserved(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_one(rollout=Mock(side_effect=KeyboardInterrupt))
        path = self.campaign / "tasks" / self.instance / "attempts/0001/result.json"
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            b.recover(self.campaign, self.instance, "human", "outage")
        b.recover(self.campaign, self.instance, "human", "Confirmed process stopped during outage", confirm_stopped=True)
        self.assertEqual(path.read_bytes(), original)
        b.approve_retry(self.campaign, self.instance, "human", "service recovered")
        self.run_one()
        self.assertEqual(len(report(self.campaign)["attempts"]), 2)

    def test_calibration_failure_never_constructs_provider(self):
        def fail(task, *args, **kwargs):
            return CalibrationResult(task.instance_id, CALIBRATION_FAILED, False, False, False, False)
        self.run_one(calibrator=fail)
        self.provider.assert_not_called()
        self.assertEqual(report(self.campaign)["statuses"]["infrastructure_failure"], 1)

    def test_configuration_source_and_image_changes_fail_closed(self):
        self.fp["changed"] = True
        # Stored fingerprint differs from the actual current fingerprint.
        with self.assertRaises(ValueError):
            self.run_one()
        self.fp.pop("changed")
        self.run_one(rollout=Mock(side_effect=ModelProviderError(ModelErrorKind.CONNECTION)))
        b.approve_retry(self.campaign, self.instance, "human", "connection failed")
        self.inspect.return_value["id"] = "sha256:" + "d" * 64
        with self.assertRaises(ValueError):
            self.run_one()

    def test_full_context_and_prompt_are_not_filtered(self):
        from tiny_harness.agent.context import create_run_context
        with tempfile.TemporaryDirectory() as tmp:
            context = create_run_context(b.NoModel(), Path(tmp), **CONFIG["budgets"])
            names = {t["function"]["name"] for t in context.tools}
            self.assertTrue({"code_outline", "grep", "glob", "search_code", "task"} <= names)
        def check(task, provider, directory, **kwargs):
            messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT},
                        {"role": "user", "content": task.problem_statement}]
            fake = Mock(tools=[], skill_catalog=Mock(manifests={}))
            with patch("tiny_harness.agent.context.create_run_context", return_value=fake), patch(
                "tiny_harness.agent.loop.agent_loop", return_value="answer"
            ) as loop:
                kwargs["agent_entrypoint"](provider, directory, messages, **CONFIG["budgets"])
                self.assertEqual(loop.call_args.args[0], messages)
                self.assertEqual(loop.call_args.args[2], task.problem_statement)
            return RolloutResult(task.instance_id, "answer", "patch")
        self.assertEqual(self.run_one(rollout=check)["status"], "completed")

    def test_capacity_and_manual_cleanup_have_no_mutations(self):
        with patch.object(b, "inspect_image", return_value=IMAGE), patch(
            "shutil.disk_usage", return_value=Mock(free=100)
        ):
            result = b.capacity(self.campaign, self.root, [self.instance], 80, 30)
            self.assertFalse(result["capacity_ok"])
        with patch.object(b, "inspect_image", return_value=IMAGE), patch(
            "subprocess.run", return_value=Mock(stdout="")
        ) as command:
            value = b.cleanup_plan(self.campaign, ["reference"])
            self.assertTrue(value["candidates"][0]["protected"])
            self.assertIsNone(value["candidates"][0]["command"])
            self.assertEqual(command.call_args.args[0], ["docker", "ps", "-aq"])

    def test_never_mutated_is_distinct_from_missing_observations(self):
        from evals.swe_bench_lite.benchmark_report import attempt_metrics
        self.run_one()
        attempt = self.campaign / "tasks" / self.instance / "attempts/0001"
        events = attempt / "rollout/events.jsonl"
        events.write_text(json.dumps({"event_type": "run_finished", "data": {}}) + "\n")
        self.assertEqual(attempt_metrics(attempt)["first_mutation_state"], "never_mutated")
        events.write_text(json.dumps({"event_type": "tool_started", "data": {"turn": 1}}) + "\n")
        self.assertEqual(attempt_metrics(attempt)["first_mutation_state"], "unknown")

    def test_completed_graded_batch_can_be_manually_planned_for_cleanup(self):
        self.run_one()
        b.grade_attempt(self.campaign, self.instance, 1, evaluator=self.evaluator, image_inspector=self.inspect)
        def inspector(reference):
            return IMAGE if reference == "finished-test-image" else {**IMAGE, "id": "sha256:" + "f" * 64}
        with patch.object(b, "inspect_image", side_effect=inspector), patch(
            "subprocess.run", return_value=Mock(stdout="")
        ) as command:
            plan = b.cleanup_plan(self.campaign, ["finished-test-image"])
            self.assertFalse(plan["candidates"][0]["protected"])
            self.assertEqual(plan["candidates"][0]["command"], ["docker", "image", "rm", IMAGE["id"]])
            self.assertEqual(command.call_count, 1)  # container inventory only, never remove


class BoundariesTest(unittest.TestCase):
    def test_frozen_integrity_and_allowlist(self):
        b.verify_bundle()
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            (p / "SHA256SUMS").write_text("tampered")
            with self.assertRaises(ValueError):
                b.verify_bundle(p)

    def test_official_split_and_local_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.run", return_value=Mock(
            returncode=0, stdout="", stderr=""
        )) as command:
            p = Path(tmp)
            run_official_evaluation(p / "predictions.jsonl", p / "dataset.jsonl", p,
                                    split="test", local_images_only=True)
            argv = command.call_args.args[0]
            self.assertEqual(argv[argv.index("--split") + 1], "test")
            self.assertIn("evals.swe_bench_lite.official_local", argv)
            self.assertIn("PYTHONPATH", command.call_args.kwargs["env"])

    def test_official_empty_conflicting_and_infra_verdicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "report.json"
            for data, expected in [({"empty_patch_ids": ["one"]}, "unresolved"),
                    ({"resolved_ids": ["one"], "error_ids": ["one"]}, "official_evaluation_error"),
                    ({"unresolved_ids": ["one"], "infra_failure_ids": ["one"]}, "infrastructure_failure"),
                    ({}, "official_evaluation_error")]:
                p.write_text(json.dumps(data))
                self.assertEqual(official_status(p.parent, "one")[0], expected)

    def test_execution_requires_explicit_flag(self):
        with patch.object(b, "run_attempt") as run, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            b.main(["run", "unused", "--instance-id", "one"])
        run.assert_not_called()

    def test_official_guard_denies_pull_and_restores_sdk(self):
        from docker.models.images import ImageCollection
        from evals.swe_bench_lite.official_local import main
        original = ImageCollection.pull
        def fake(*args, **kwargs):
            with self.assertRaises(RuntimeError):
                ImageCollection.pull(None, "never-pull-this")
        with patch("runpy.run_module", side_effect=fake):
            main()
        self.assertIs(ImageCollection.pull, original)

    def test_local_environment_sets_never_pull_flags(self):
        from evals.swe_bench_lite.docker_workspace import DockerTaskEnvironment
        task = b.verify_bundle()[0][0]
        environment = DockerTaskEnvironment(task, require_local_image=True)
        calls = []
        def fake(argv, **kwargs):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, "", "")
        with patch("evals.swe_bench_lite.docker_workspace._run", side_effect=fake):
            with environment:
                pass
        for argv in calls:
            if argv[1] in {"create", "run"}:
                self.assertEqual(argv[argv.index("--pull") + 1], "never")

    @unittest.skipUnless(importlib.util.find_spec("swebench"), "Requires official script projection")
    def test_calibration_uses_official_script_without_running_tests(self):
        from evals.swe_bench_lite.calibration import _run_phase
        from swebench.harness.utils import make_test_spec
        task = b.verify_bundle()[0][0]
        bundle = b.verify_bundle()[1][0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            class Fake:
                workspace = root / "workspace"
                def __enter__(self):
                    self.workspace.mkdir()
                    return self
                def __exit__(self, *args):
                    pass
                def exec(self, command, **kwargs):
                    return subprocess.CompletedProcess(command, 0, "fixture log", "")
            _run_phase(task, bundle, root, "baseline", lambda task: Fake(), lambda *args: {}, True)
            record = bundle.official_record()
            record["image"] = task.image
            self.assertEqual((root / "workspace/.tinyharness/eval.sh").read_bytes(),
                             make_test_spec(record).eval_script.encode())


@unittest.skipUnless(importlib.util.find_spec("swebench"), "Requires installed evaluation dependency")
class ActualPreparationTest(unittest.TestCase):
    def test_actual_dependencies_source_and_tools_are_frozen_without_execution(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            b, "docker_json", side_effect=AssertionError("No Docker access")
        ), patch("tiny_harness.models.chat_completions.OpenAI") as client:
            campaign = b.prepare(Path(temporary) / "campaign", copy.deepcopy(CONFIG))
            meta, tasks, _, _ = b.load_campaign(campaign, execution=True)
            client.assert_not_called()
            self.assertEqual(len(tasks), 60)
            names = {x["function"]["name"] for x in meta["fingerprint"]["capabilities"]["tools"]}
            self.assertTrue({"code_outline", "grep", "glob", "search_code", "task"} <= names)
            self.assertEqual(meta["fingerprint"]["prompt_sha256"], b.sha(AGENT_SYSTEM_PROMPT.encode()))
            self.assertTrue((campaign / "source/tiny_harness/agent/loop.py").is_file())


if __name__ == "__main__":
    unittest.main()
