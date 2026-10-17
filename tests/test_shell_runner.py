import copy
import os
import sys
import time
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evals.swe_bench_lite.docker_workspace import DockerShellRunner
from tiny_harness.agent.loop import run_agent
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.runtime.shell_runner import SubprocessShellRunner, _terminate_process_tree


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
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / "output.py"
            script.write_text(
                "import sys; sys.stdout.buffer.write(b'out\\r\\nline\\r'); "
                "sys.stderr.buffer.write(b'err'); sys.exit(2)",
                encoding="utf-8",
            )
            result = SubprocessShellRunner(7).run(Path(folder), f'"{sys.executable}" "{script}"')
        self.assertEqual(result, "Exit code: 2\nout\nline\nerr")

    def test_standard_input_is_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / "input.py"
            script.write_text("import sys; print(repr(sys.stdin.read()))", encoding="utf-8")
            result = SubprocessShellRunner(3).run(Path(folder), f'"{sys.executable}" "{script}"')
        self.assertEqual(result, "Exit code: 0\n''")

    @unittest.skipUnless(os.name == "nt", "Windows process-tree cleanup")
    def test_windows_timeout_terminates_descendant_with_inherited_output(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            child = workspace / "child.py"
            child.write_text("import time; print('child', flush=True); time.sleep(60)", encoding="utf-8")
            parent = workspace / "parent.py"
            parent.write_text(
                "import subprocess, sys, time\nfrom pathlib import Path\n"
                "child = subprocess.Popen([sys.executable, 'child.py'])\n"
                "Path('child.pid').write_text(str(child.pid))\ntime.sleep(60)\n", encoding="utf-8",
            )
            command = f'"{sys.executable}" "{parent}"'
            # An outer watchdog keeps a regression from hanging the test suite.
            probe = (
                "from pathlib import Path\n"
                "from tiny_harness.runtime.shell_runner import SubprocessShellRunner\n"
                "try:\n"
                f"    SubprocessShellRunner(1).run(Path({str(workspace)!r}), {command!r})\n"
                "except TimeoutError as error:\n    print(error)\n"
                "else:\n    raise AssertionError('expected timeout')\n"
            )
            started = time.monotonic()
            try:
                result = subprocess.run(
                    [sys.executable, "-c", probe], stdin=subprocess.DEVNULL,
                    capture_output=True, text=True, timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("timed out after 1 seconds", result.stdout)
                self.assertIn("Process tree terminated", result.stdout)
                self.assertLess(time.monotonic() - started, 12)
                self.assertTrue((workspace / "child.pid").exists())
                import ctypes
                from ctypes import wintypes
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                kernel.OpenProcess.restype = wintypes.HANDLE
                kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
                kernel.WaitForSingleObject.restype = wintypes.DWORD
                kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                handle = kernel.OpenProcess(0x00100000, False, int((workspace / "child.pid").read_text()))
                if handle:
                    try:
                        self.assertEqual(kernel.WaitForSingleObject(handle, 1000), 0)
                    finally:
                        kernel.CloseHandle(handle)
                else:
                    self.assertEqual(ctypes.get_last_error(), 87)  # PID no longer exists.
            finally:
                if (workspace / "child.pid").exists():
                    subprocess.run(
                        ["taskkill", "/PID", (workspace / "child.pid").read_text(), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL, timeout=5,
                    )

    @unittest.skipUnless(os.name == "nt", "Windows cleanup fallback")
    def test_failed_tree_cleanup_has_only_bounded_waits(self):
        process = Mock(pid=123)
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired("command", 1)
        with patch("tiny_harness.runtime.shell_runner.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("taskkill", 5)) as kill:
            self.assertFalse(_terminate_process_tree(process))
        self.assertEqual(kill.call_args.kwargs["timeout"], 5)
        self.assertIs(kill.call_args.kwargs["stdout"], subprocess.DEVNULL)
        process.kill.assert_called_once()
        process.wait.assert_called_once_with(timeout=1)

    def test_timeout_is_tool_error_and_session_continues(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            script = workspace / "sleep.py"
            script.write_text("import time; time.sleep(60)", encoding="utf-8")
            command = f'"{sys.executable}" "{script}"'
            provider = ScriptedProvider([
                ModelResponse(None, None, [ToolCall("slow", "bash", json.dumps({"command": command}))], "tool_calls"),
                ModelResponse("continue after timeout", None, [], "stop"),
            ])
            answer = run_agent(provider, workspace, [], shell_runner=SubprocessShellRunner(.2))
        self.assertEqual(answer, "continue after timeout")
        self.assertIn("Error: TimeoutError: Shell command timed out", provider.calls[1][-1]["content"])

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
                "/bin/bash", "-lc", "umask 000\npytest -q",
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
