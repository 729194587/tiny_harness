import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tiny_harness.agent.loop import run_agent
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.runtime.test_runner import SubprocessTestRunner
from tiny_harness.runtime.skills import discover_skills
from tiny_harness.runtime.todos import TodoManager
from tiny_harness.tools.discovery import discover_tools
from tiny_harness.tools.registry import dispatch


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
            }
        )
        return self.responses.pop(0)


class RecordingRunner:
    def __init__(self) -> None:
        self.workspaces = []

    def run(self, workspace: Path) -> str:
        self.workspaces.append(workspace)
        return "Exit code: 0\nRan 1 test\n\nOK"


class RecordingEventLogger:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event_type, data=None) -> None:
        self.events.append(
            {"event_type": event_type.value, "data": dict(data or {})}
        )


class SubprocessTestRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_executes_canonical_suite_without_bytecode_artifacts(self) -> None:
        tests = self.workspace / "tests"
        tests.mkdir()
        (tests / "test_environment.py").write_text(
            "import os\n"
            "import unittest\n"
            "from pathlib import Path\n\n"
            "class EnvironmentTest(unittest.TestCase):\n"
            "    def test_runtime_environment(self):\n"
            "        self.assertEqual(\n"
            "            os.environ.get('PYTHONDONTWRITEBYTECODE'), '1'\n"
            "        )\n"
            "        self.assertEqual(\n"
            "            Path.cwd(), Path(__file__).resolve().parents[1]\n"
            "        )\n",
            encoding="utf-8",
        )
        runner = SubprocessTestRunner(
            (
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-v",
            ),
            timeout_seconds=30,
        )

        result = runner.run(self.workspace)

        self.assertTrue(result.startswith("Exit code: 0\n"))
        self.assertIn("Ran 1 test", result)
        self.assertIn("OK", result)
        self.assertEqual(list(self.workspace.rglob("__pycache__")), [])
        self.assertEqual(list(self.workspace.rglob("*.pyc")), [])

    def test_uses_fixed_argv_workspace_shell_false_timeout_and_environment(self):
        runner = SubprocessTestRunner(
            ("python", "-m", "unittest"),
            timeout_seconds=7,
        )
        completed = subprocess.CompletedProcess(
            ["python", "-m", "unittest"],
            1,
            stdout="stdout",
            stderr="stderr",
        )

        with patch(
            "tiny_harness.runtime.test_runner.subprocess.run",
            return_value=completed,
        ) as run:
            result = runner.run(self.workspace)

        self.assertEqual(result, "Exit code: 1\nstdoutstderr")
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["python", "-m", "unittest"])
        self.assertFalse(kwargs["shell"])
        self.assertEqual(kwargs["cwd"], self.workspace.resolve())
        self.assertEqual(kwargs["timeout"], 7)
        self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")

    def test_timeout_becomes_an_ordinary_tool_error(self) -> None:
        runner = SubprocessTestRunner(
            ("python", "-m", "unittest"),
            timeout_seconds=3,
        )
        with patch(
            "tiny_harness.runtime.test_runner.subprocess.run",
            side_effect=subprocess.TimeoutExpired(runner.argv, 3),
        ):
            registry = discover_tools(
                SimpleNamespace(
                    workspace=self.workspace,
                    todo_manager=TodoManager(),
                    subagent_runner=None,
                    skill_catalog=discover_skills(self.workspace),
                    compaction_request=None,
                    test_runner=runner,
                )
            )
            result = dispatch(
                registry,
                ToolCall("tests-timeout", "run_tests", "{}"),
            )

        self.assertIn("Error: TimeoutExpired", result.content)


class RunTestsRuntimeSemanticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_root_agent_calls_run_tests_then_must_propose_final_answer(self) -> None:
        runner = RecordingRunner()
        provider = ScriptedProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("tests-1", "run_tests", "{}")],
                    "tool_calls",
                ),
                ModelResponse("finished", None, [], "stop"),
            ]
        )

        answer = run_agent(
            provider,
            self.workspace,
            [{"role": "user", "content": "task"}],
            test_runner=runner,
        )

        self.assertEqual(answer, "finished")
        self.assertEqual(runner.workspaces, [self.workspace])
        self.assertIn(
            "run_tests",
            [tool["function"]["name"] for tool in provider.calls[0]["tools"]],
        )
        self.assertEqual(
            provider.calls[1]["messages"][-1]["content"],
            "Exit code: 0\nRan 1 test\n\nOK",
        )

    def test_passing_tests_are_an_ordinary_tool_result(self) -> None:
        runner = RecordingRunner()
        logger = RecordingEventLogger()
        provider = ScriptedProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("tests-1", "run_tests", "{}")],
                    "tool_calls",
                ),
                ModelResponse("tests passed", None, [], "stop"),
            ]
        )

        answer = run_agent(
            provider,
            self.workspace,
            [{"role": "user", "content": "task"}],
            max_turns=2,
            test_runner=runner,
            event_logger=logger,
        )

        event_names = [event["event_type"] for event in logger.events]
        self.assertEqual(answer, "tests passed")
        self.assertIn("run_finished", event_names)
        self.assertEqual(runner.workspaces, [self.workspace])


if __name__ == "__main__":
    unittest.main()
