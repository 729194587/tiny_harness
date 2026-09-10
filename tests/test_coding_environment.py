import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tiny_harness.agent.context import create_run_context, initialize_run_state
from tiny_harness.agent.environment import ENVIRONMENT_CONTEXT_MARKER
from tiny_harness.agent.messages import ToolCall
from tiny_harness.environments import CodingEnvironmentAdapter
from tiny_harness.environments.coding import MAX_CONTEXT_CHARS, MAX_GIT_OUTPUT_CHARS
from tiny_harness.runtime.hooks import HookBlock, ToolHooks
from tiny_harness.runtime.permissions import DEFAULT_PERMISSION_POLICY, PermissionDecision
from tiny_harness.runtime.shell_runner import SubprocessShellRunner
from tiny_harness.tools.registry import dispatch


class GitPolicy:
    def decide(self, name, arguments):
        if name in {"git_status", "git_diff"} and not arguments:
            return PermissionDecision.ALLOW
        return DEFAULT_PERMISSION_POLICY.decide(name, arguments)


class RecordingRunner:
    def __init__(self, report="Exit code: 0"):
        self.report = report
        self.calls = []

    def run(self, workspace, command):
        self.calls.append((workspace, command))
        return self.report


class CodingEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = Path(self.directory.name)
        self.adapter = CodingEnvironmentAdapter()

    def context(self, runner=None, **kwargs):
        return create_run_context(None, self.workspace, environment_adapter=self.adapter,
                                  shell_runner=runner or RecordingRunner(), **kwargs)

    def test_context_is_shallow_bounded_and_does_not_read_contents(self):
        (self.workspace / "README.md").write_text("SECRET_CONTENT", encoding="utf-8")
        (self.workspace / "pyproject.toml").touch()
        (self.workspace / "src").mkdir()
        (self.workspace / "src" / "nested-secret.py").touch()
        (self.workspace / "node_modules").mkdir()
        context = self.context()
        text = context.environment_context
        self.assertIn('"README.md"', text)
        self.assertIn("filename evidence only", text)
        self.assertNotIn("SECRET_CONTENT", text)
        self.assertNotIn("nested-secret", text)
        self.assertNotIn('"node_modules/"', text)
        self.assertIn("run_tests is not configured", text)
        messages = [{"role": "user", "content": "task"}]
        initialize_run_state(messages, context, "task")
        self.assertEqual(next(m["content"] for m in messages
                              if m.get("name") == ENVIRONMENT_CONTEXT_MARKER), text)
        for i in range(80):
            (self.workspace / (f"{i:03d}" + "x" * 90)).touch()
        text = self.adapter.initial_context(context)
        self.assertLessEqual(len(text), MAX_CONTEXT_CHARS)
        self.assertIn("truncated", text)

    def test_outside_symlink_is_omitted(self):
        with tempfile.TemporaryDirectory() as outside:
            try:
                (self.workspace / "outside-link").symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("Symlink creation unavailable")
            self.assertNotIn('"outside-link/"', self.context().environment_context)

    def test_test_runner_is_only_reported_when_supplied(self):
        context = self.context(test_runner=SimpleNamespace(run=lambda workspace: "tests"))
        self.assertIn("run_tests is configured", context.environment_context)
        self.assertIn("run_tests", [d.name for d in context.tool_registry.list()])

    def test_default_has_no_coding_tools_and_adapter_only_appends(self):
        baseline = create_run_context(None, self.workspace)
        context = self.context()
        self.assertEqual(context.tools[:-2], baseline.tools)
        self.assertEqual([d.name for d in context.tool_registry.list()][-2:],
                         ["git_status", "git_diff"])

    def test_policy_and_hooks_gate_injected_runner(self):
        runner = RecordingRunner()
        context = self.context(runner)
        call = ToolCall("git-1", "git_status", "{}")
        denied = dispatch(context.tool_registry, call)
        self.assertIn("Permission denied", denied.content)
        self.assertEqual(runner.calls, [])
        hooks = ToolHooks()
        hooks.register_pre(lambda context: HookBlock("blocked"))
        dispatch(context.tool_registry, call, permission_policy=GitPolicy(), tool_hooks=hooks)
        self.assertEqual(runner.calls, [])
        observed = []
        hooks = ToolHooks()
        hooks.register_post(lambda context, result: observed.append(result.content))
        result = dispatch(context.tool_registry, call, permission_policy=GitPolicy(), tool_hooks=hooks)
        self.assertEqual(observed, [result.content])
        self.assertEqual(runner.calls[0][0], self.workspace)
        self.assertIn("--no-optional-locks", runner.calls[0][1])

    def test_arguments_rejected_before_runner_and_reports_bounded(self):
        runner = RecordingRunner("Exit code: 128\nfatal: not a git repository\n" + "x" * 20000)
        context = self.context(runner)
        definition = context.tool_registry.lookup("git_diff")
        with self.assertRaises(TypeError):
            definition.execute(ToolCall("1", "git_diff", "{}"), {"command": "anything"})
        self.assertEqual(runner.calls, [])
        result = dispatch(context.tool_registry, ToolCall("2", "git_diff", "{}"),
                          permission_policy=GitPolicy())
        self.assertLessEqual(len(result.content), MAX_GIT_OUTPUT_CHARS)
        self.assertIn("Exit code: 128", result.content)
        self.assertIn("truncated", result.content)
        self.assertIn("--no-ext-diff --no-textconv", runner.calls[0][1])

    def test_runner_exception_uses_standard_error_result(self):
        class BrokenRunner:
            def run(self, workspace, command):
                raise OSError("runner unavailable")
        context = self.context(BrokenRunner())
        result = dispatch(context.tool_registry, ToolCall("1", "git_status", "{}"),
                          permission_policy=GitPolicy())
        self.assertIn("Error: OSError", result.content)

    def test_docker_runner_routes_git_to_container(self):
        from evals.swe_bench_lite.docker_workspace import DockerShellRunner

        runner = DockerShellRunner("coding-test", self.workspace)
        context = self.context(runner)
        with patch("evals.swe_bench_lite.docker_workspace._run") as execute:
            execute.return_value = SimpleNamespace(returncode=0, stdout="? code.py", stderr="")
            result = dispatch(context.tool_registry, ToolCall("1", "git_status", "{}"),
                              permission_policy=GitPolicy())
        argv = execute.call_args.args[0]
        self.assertEqual(argv[:7], ["docker", "exec", "--workdir", "/testbed",
                                    "coding-test", "/bin/bash", "-lc"])
        self.assertIn("status --porcelain", argv[7])
        self.assertIn("? code.py", result.content)

    @unittest.skipUnless(shutil.which("git"), "Git unavailable")
    def test_real_git_staged_unstaged_untracked_and_index_unchanged(self):
        def git(*args):
            return subprocess.run(["git", *args], cwd=self.workspace, check=True,
                                  capture_output=True, text=True).stdout
        git("init")
        for name in ("staged.txt", "unstaged.txt"):
            (self.workspace / name).write_text("before\n", encoding="utf-8")
        git("add", ".")
        git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "base")
        (self.workspace / "staged.txt").write_text("staged change\n", encoding="utf-8")
        git("add", "staged.txt")
        (self.workspace / "unstaged.txt").write_text("unstaged change\n", encoding="utf-8")
        (self.workspace / "new.txt").write_text("untracked secret\n", encoding="utf-8")
        index = (self.workspace / ".git" / "index").read_bytes()
        context = self.context(SubprocessShellRunner())
        results = {}
        for name in ("git_status", "git_diff"):
            results[name] = dispatch(context.tool_registry, ToolCall(name, name, "{}"),
                                     permission_policy=GitPolicy()).content
        self.assertIn("? new.txt", results["git_status"])
        self.assertIn("1 M.", results["git_status"])
        self.assertIn("1 .M", results["git_status"])
        self.assertIn("+staged change", results["git_diff"])
        self.assertIn("+unstaged change", results["git_diff"])
        self.assertNotIn("untracked secret", results["git_diff"])
        self.assertEqual(index, (self.workspace / ".git" / "index").read_bytes())


if __name__ == "__main__":
    unittest.main()
