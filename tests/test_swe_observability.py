import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evals.swe_bench_lite.observability import WorkspaceMutationLogger, source_metadata, workspace_snapshot
from evals.swe_bench_lite.pipeline import rollout_task, run_selected_smoke
from evals.swe_bench_lite.report import analyze_run
from tiny_harness.runtime.events import EventType, JsonlEventLogger
from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.messages import ToolCall
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.tools.registry import dispatch
from test_swe_bench_pipeline import FakeEnvironment, task


class WorkspaceObservationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        subprocess.run(["git", "init", "-q", str(self.workspace)], check=True, capture_output=True)
        self.logger = WorkspaceMutationLogger(JsonlEventLogger(self.root / "events.jsonl"), self.workspace)

    def interval(self, turn, action, *, parent=None):
        data = {"turn": turn, "tool_call_id": str(turn), "tool_name": "bash"}
        if parent is not None:
            data["parent_tool_call_id"] = parent
        self.logger.emit(EventType.TOOL_STARTED, data)
        action()
        self.logger.emit(EventType.TOOL_RESULT, {**data, "outcome": "returned"})

    def observations(self):
        return analyze_run(self.root)

    def test_indirect_untracked_content_changes_deletion_and_same_bytes(self):
        file = self.workspace / "ignored.txt"
        self.interval(1, lambda: None)
        self.interval(2, lambda: subprocess.run(
            [sys.executable, "-c", "from pathlib import Path; Path('ignored.txt').write_text('abc')"],
            cwd=self.workspace, check=True, capture_output=True))
        timestamp = file.stat().st_mtime_ns
        self.interval(3, lambda: file.write_text("abc"))
        def overwrite():
            file.write_text("xyz")
            os.utime(file, ns=(timestamp, timestamp))
        self.interval(4, overwrite)
        self.interval(5, file.unlink)
        report = self.observations()
        self.assertEqual(report["first_workspace_mutation_turn"], 2)
        self.assertEqual([o["workspace_changed"] for o in report["workspace_observations"]],
                         [False, True, False, True, True])
        self.assertNotIn("ignored.txt", json.dumps(report))

    def test_real_write_edit_dispatch_preserves_results(self):
        policy = Mock()
        policy.decide.return_value = PermissionDecision.ALLOW
        context = create_run_context(Mock(), self.workspace, allow_subagent=False)
        for turn, name, args in (
            (1, "write_file", {"path": "file.txt", "content": "before"}),
            (2, "edit_file", {"path": "file.txt", "old_text": "before", "new_text": "after"}),
        ):
            result = dispatch(context.tool_registry, ToolCall(str(turn), name, json.dumps(args)),
                              permission_policy=policy, event_logger=self.logger, turn=turn)
            self.assertNotIn("workspace_observed", result.content)
        self.assertEqual((self.workspace / "file.txt").read_text(), "after")
        self.assertEqual(self.observations()["first_workspace_mutation_turn"], 1)

    def test_internal_artifacts_excluded_and_unreadable_snapshot_is_unknown(self):
        def artifacts():
            for name in (".git", ".tinyharness"):
                directory = self.workspace / name
                directory.mkdir(exist_ok=True)
                (directory / "state").write_text("internal")
        self.interval(1, artifacts)
        self.assertEqual(workspace_snapshot(self.workspace), {})
        with patch("evals.swe_bench_lite.observability.workspace_snapshot", side_effect=OSError("private")):
            self.interval(2, lambda: None)
        self.interval(3, lambda: (self.workspace / "file").write_text("change"))
        report = self.observations()
        self.assertIsNone(report["first_workspace_mutation_turn"])
        self.assertNotIn("private", json.dumps(report))

    def test_git_ignored_outputs_do_not_count_but_tracked_and_untracked_content_do(self):
        (self.workspace / ".gitignore").write_text("__pycache__/\n.pytest_cache/\nbuild/\n")
        build = self.workspace / "build"
        build.mkdir()
        tracked = build / "tracked.txt"
        tracked.write_text("tracked")
        subprocess.run(["git", "-C", str(self.workspace), "add", "-f", "build/tracked.txt"],
                       check=True, capture_output=True)

        def generate():
            for directory in ("__pycache__", ".pytest_cache", "build", "pkg/__pycache__"):
                output = self.workspace / directory
                output.mkdir(parents=True, exist_ok=True)
                (output / "generated").write_text("generated content")

        self.interval(1, generate)
        self.interval(2, lambda: tracked.write_text("changed tracked content"))
        source = self.workspace / "source with spaces.py"
        self.interval(3, lambda: source.write_text("new source"))
        self.interval(4, tracked.unlink)
        observations = self.observations()
        self.assertEqual([o["workspace_changed"] for o in observations["workspace_observations"]],
                         [False, True, True, True])
        self.assertEqual(observations["first_workspace_mutation_turn"], 2)
        snapshot = workspace_snapshot(self.workspace)
        self.assertEqual(set(snapshot), {".gitignore", "source with spaces.py"})

    def test_git_inventory_failure_is_unknown_not_a_full_filesystem_scan(self):
        with patch("evals.swe_bench_lite.observability.subprocess.run", side_effect=OSError("git unavailable")):
            self.interval(1, lambda: (self.workspace / "unobserved").write_text("x"))
        observation = self.observations()["workspace_observations"][0]
        self.assertEqual(observation["observation_status"], "unknown")
        self.assertIsNone(observation["workspace_changed"])

    def test_child_change_maps_to_parent_turn_and_failed_tool_is_observed(self):
        parent = {"turn": 8, "tool_call_id": "task", "tool_name": "task"}
        self.logger.emit(EventType.TOOL_STARTED, parent)
        self.interval(1, lambda: (self.workspace / "child").write_text("x"), parent="task")
        self.logger.emit(EventType.TOOL_RESULT, {**parent, "outcome": "error"})
        self.assertEqual(self.observations()["first_workspace_mutation_turn"], 8)

    def test_fatal_run_failure_finishes_pending_observation(self):
        self.logger.emit(EventType.TOOL_STARTED, {"turn": 2, "tool_call_id": "failed", "tool_name": "bash"})
        (self.workspace / "partial").write_text("partial write")
        self.logger.emit(EventType.RUN_FAILED, {"turn": 2})
        self.assertEqual(self.observations()["first_workspace_mutation_turn"], 2)
        self.assertEqual(self.logger.pending, {})

    def test_symlinks_are_not_followed(self):
        outside = self.root / "outside"
        outside.write_text("secret")
        link = self.workspace / "link"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("symlink privilege unavailable")
        before = workspace_snapshot(self.workspace)
        outside.write_text("new secret")
        self.assertEqual(workspace_snapshot(self.workspace), before)
        self.assertEqual(before["link"][0], "link")


class ProvenanceTest(unittest.TestCase):
    def test_run_metadata_survives_calibration_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("evals.swe_bench_lite.pipeline.source_metadata", return_value={
                "tinyharness_git_commit": "abc", "tinyharness_git_dirty": False,
            }):
                with self.assertRaises(RuntimeError):
                    run_selected_smoke(Mock(), model_name_or_path="model", results_root=root, run_id="run",
                                       calibrator=Mock(side_effect=RuntimeError("calibration failed")))
            metadata = json.loads((root / "run" / "metadata.json").read_text())
            self.assertEqual(metadata["tinyharness_git_commit"], "abc")
            self.assertEqual(metadata["model_name_or_path"], "model")

    def test_git_metadata_in_clean_dirty_and_unavailable_checkout(self):
        with patch("evals.swe_bench_lite.observability.subprocess.run") as run:
            run.side_effect = [Mock(stdout="abc\n"), Mock(stdout=""), Mock(stdout="abc\n"), Mock(stdout="?? new\n")]
            self.assertEqual(source_metadata(), {"tinyharness_git_commit": "abc", "tinyharness_git_dirty": False})
            self.assertTrue(source_metadata()["tinyharness_git_dirty"])
            run.side_effect = OSError("git unavailable")
            self.assertIsNone(source_metadata()["tinyharness_git_commit"])

    def test_rollout_metadata_is_present_before_execution_and_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            def agent(*args, **kwargs):
                metadata = json.loads((output / "metadata.json").read_text())
                self.assertEqual(metadata["status"], "RUNNING")
                self.assertEqual(metadata["tinyharness_git_commit"], "abc")
                raise RuntimeError("failed")
            with patch("evals.swe_bench_lite.pipeline.source_metadata", return_value={
                "tinyharness_git_commit": "abc", "tinyharness_git_dirty": True,
            }):
                with self.assertRaises(RuntimeError):
                    rollout_task(task(), Mock(), output, model_name_or_path="model",
                                 environment_factory=FakeEnvironment, agent_entrypoint=agent)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "FAILED")
            for key in ("model_name_or_path", "max_turns", "max_context_tokens",
                        "working_context_trigger_tokens", "working_context_target_tokens",
                        "keep_recent_tool_batches", "working_memory_enabled", "progress_enabled",
                        "coding_environment_enabled", "tinyharness_git_dirty"):
                self.assertIn(key, metadata)
