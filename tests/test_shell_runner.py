import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.swe_bench_lite.docker_workspace import DockerShellRunner
from tiny_harness.agent.loop import run_agent
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.runtime.shell_runner import SubprocessShellRunner


class RecordingShellRunner:
    def __init__(self):
        self.calls = []

    def run(self, workspace, command):
        self.calls.append((workspace, command))
        return "Exit code: 0\ncontainer output"


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append(copy.deepcopy(messages))
        return self.responses.pop(0)


class ShellRunnerTest(unittest.TestCase):
    def test_default_subprocess_runner_preserves_host_shell_behavior(self):
        workspace = Path("workspace").resolve()
        completed = subprocess.CompletedProcess("command", 2, "out", "err")
        with patch(
            "tiny_harness.runtime.shell_runner.subprocess.run",
            return_value=completed,
        ) as run:
            result = SubprocessShellRunner(timeout_seconds=7).run(workspace, "command")
        self.assertEqual(result, "Exit code: 2\nouterr")
        self.assertEqual(run.call_args.args[0], "command")
        self.assertTrue(run.call_args.kwargs["shell"])
        self.assertEqual(run.call_args.kwargs["cwd"], workspace)
        self.assertEqual(run.call_args.kwargs["timeout"], 7)

    def test_docker_runner_constructs_exec_and_validates_workspace_mapping(self):
        workspace = Path("workspace").resolve()
        runner = DockerShellRunner("task-container", workspace, timeout_seconds=9)
        completed = subprocess.CompletedProcess([], 0, "ok", "")
        with patch(
            "evals.swe_bench_lite.docker_workspace._run",
            return_value=completed,
        ) as execute:
            result = runner.run(workspace, "pytest -q")
        self.assertEqual(result, "Exit code: 0\nok")
        self.assertEqual(
            execute.call_args.args[0],
            [
                "docker", "exec", "--workdir", "/testbed", "task-container",
                "/bin/bash", "-lc", "pytest -q",
            ],
        )
        self.assertFalse(execute.call_args.kwargs["check"])
        with self.assertRaisesRegex(ValueError, "bind mount"):
            runner.command_argv(workspace / "other", "pwd")

    def test_shell_runner_propagates_to_subagent(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        workspace = Path(temporary.name)
        shell_runner = RecordingShellRunner()
        provider = ScriptedProvider(
            [
                ModelResponse(None, None, [ToolCall("task", "task", '{"prompt":"check"}')], "tool_calls"),
                ModelResponse(None, None, [ToolCall("bash", "bash", '{"command":"pwd"}')], "tool_calls"),
                ModelResponse("child done", None, [], "stop"),
                ModelResponse("parent done", None, [], "stop"),
            ]
        )
        answer = run_agent(
            provider,
            workspace,
            [{"role": "user", "content": "delegate"}],
            shell_runner=shell_runner,
        )
        self.assertEqual(answer, "parent done")
        self.assertEqual(shell_runner.calls, [(workspace, "pwd")])
        self.assertEqual(provider.calls[2][-1]["content"], "Exit code: 0\ncontainer output")


if __name__ == "__main__":
    unittest.main()
