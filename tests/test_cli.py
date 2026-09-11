import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tiny_harness.__main__ import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MODEL,
    _ask_permission,
    main,
)
from tiny_harness.agent.context import DEFAULT_SUBAGENT_MAX_TURNS
from tiny_harness.runtime.events import CompositeEventLogger, JsonlEventLogger
from tiny_harness.runtime.console import ConsoleEventLogger
from tiny_harness.runtime.recovery import RecoveryPolicy
from tiny_harness.runtime.task_state import TaskStateConfig


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @patch("tiny_harness.__main__.AgentSession")
    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_wires_one_shot_cli_to_provider_and_session(
        self, provider_class, session_class
    ) -> None:
        provider = provider_class.return_value
        session_class.return_value.submit.return_value = "final answer"
        stdout = io.StringIO()

        with patch.dict(os.environ, {"TINYHARNESS_API_KEY": "secret"}, clear=True):
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "create a file",
                        "--workspace",
                        str(self.workspace),
                        "--max-turns",
                        "7",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "final answer\n")
        provider_class.assert_called_once_with(
            api_key="secret",
            model=DEFAULT_MODEL,
            base_url=DEFAULT_BASE_URL,
        )
        positional = session_class.call_args.args
        self.assertIs(positional[0], provider)
        self.assertEqual(positional[1], self.workspace.resolve())
        self.assertIn(str(self.workspace.resolve()), positional[2])
        self.assertIn("todo_write", positional[2])
        self.assertIn("compact", positional[2])
        self.assertEqual(session_class.call_args.kwargs["max_turns"], 7)
        self.assertIs(
            session_class.call_args.kwargs["permission_prompt"], _ask_permission
        )
        self.assertEqual(
            session_class.call_args.kwargs["max_context_tokens"],
            DEFAULT_MAX_CONTEXT_TOKENS,
        )
        self.assertEqual(
            session_class.call_args.kwargs["subagent_max_turns"],
            DEFAULT_SUBAGENT_MAX_TURNS,
        )
        self.assertEqual(
            session_class.call_args.kwargs["recovery_policy"],
            RecoveryPolicy(max_retries=2),
        )
        self.assertFalse(session_class.call_args.kwargs["memory_enabled"])
        self.assertEqual(
            session_class.call_args.kwargs["task_state_config"], TaskStateConfig()
        )
        session_class.return_value.submit.assert_called_once_with("create a file")

    @patch("tiny_harness.__main__.AgentSession")
    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_task_state_options_are_independent(self, _, session_class) -> None:
        cases = [
            (["--task-state"], TaskStateConfig(enabled=True)),
            (["--task-state", "--task-state-reflection"],
             TaskStateConfig(enabled=True, reflection_enabled=True)),
            (["--task-state", "--task-state-reflection", "--task-state-reflection-interval", "5"],
             TaskStateConfig(enabled=True, reflection_enabled=True, reflection_interval=5)),
            (["--task-state", "--task-state-reflection-interval", "3"],
             TaskStateConfig(enabled=True, reflection_interval=3)),
            (["--task-state-reflection"], TaskStateConfig(reflection_enabled=True)),
            (["--task-state-reflection-interval", "4"], TaskStateConfig(reflection_interval=4)),
            (["--task-state-reflection-interval", "0"], TaskStateConfig()),
        ]
        session_class.return_value.submit.return_value = "done"
        for options, expected in cases:
            with self.subTest(options=options):
                with patch.dict(os.environ, {"TINYHARNESS_API_KEY": "secret"}, clear=True):
                    with contextlib.redirect_stdout(io.StringIO()):
                        exit_code = main(["task", "--workspace", str(self.workspace), *options])
                self.assertEqual(exit_code, 0)
                self.assertEqual(session_class.call_args.kwargs["task_state_config"], expected)

    @patch("tiny_harness.__main__.AgentSession")
    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_uses_environment_model_and_base_url(
        self, provider_class, session_class
    ) -> None:
        session_class.return_value.submit.return_value = "done"
        environment = {
            "TINYHARNESS_API_KEY": "secret",
            "TINYHARNESS_MODEL": "custom-model",
            "TINYHARNESS_BASE_URL": "https://example.test",
        }

        with patch.dict(os.environ, environment, clear=True):
            with contextlib.redirect_stdout(io.StringIO()):
                main(["task", "--workspace", str(self.workspace)])

        provider_class.assert_called_once_with(
            api_key="secret",
            model="custom-model",
            base_url="https://example.test",
        )

    @patch("tiny_harness.__main__.AgentSession")
    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_event_log_factory_creates_a_fresh_logger_per_submit(
        self, _, session_class
    ) -> None:
        session_class.return_value.submit.return_value = "done"
        log_path = self.workspace / "logs" / "run.jsonl"

        with patch.dict(os.environ, {"TINYHARNESS_API_KEY": "secret"}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "task",
                        "--workspace",
                        str(self.workspace),
                        "--event-log",
                        str(log_path),
                    ]
                )

        factory = session_class.call_args.kwargs["event_logger_factory"]
        composite = factory()
        self.assertIsInstance(composite, CompositeEventLogger)
        self.assertIsInstance(composite.loggers[0], ConsoleEventLogger)
        first = composite.loggers[1]
        second = factory().loggers[1]
        self.assertIsInstance(first, JsonlEventLogger)
        self.assertIsInstance(second, JsonlEventLogger)
        self.assertEqual(first.path, log_path.resolve())
        self.assertNotEqual(first.run_id, second.run_id)

    @patch("tiny_harness.__main__.AgentSession")
    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_no_event_log_factory_returns_console_logger(self, _, session_class) -> None:
        session_class.return_value.submit.return_value = "done"
        with patch.dict(os.environ, {"TINYHARNESS_API_KEY": "secret"}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()):
                main(["task", "--workspace", str(self.workspace)])

        factory = session_class.call_args.kwargs["event_logger_factory"]
        self.assertIsInstance(factory(), ConsoleEventLogger)

    @patch("tiny_harness.__main__.AgentSession")
    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_passes_context_and_recovery_configuration(
        self, _, session_class
    ) -> None:
        session_class.return_value.submit.return_value = "done"
        with patch.dict(os.environ, {"TINYHARNESS_API_KEY": "secret"}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "task",
                        "--workspace",
                        str(self.workspace),
                        "--max-context-tokens",
                        "2250",
                        "--subagent-max-turns",
                        "4",
                        "--max-model-retries",
                        "5",
                    ]
                )

        self.assertEqual(session_class.call_args.kwargs["max_context_tokens"], 2250)
        self.assertEqual(session_class.call_args.kwargs["subagent_max_turns"], 4)
        self.assertEqual(
            session_class.call_args.kwargs["recovery_policy"],
            RecoveryPolicy(max_retries=5),
        )

    @patch("tiny_harness.__main__.AgentSession")
    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_can_disable_context_compaction(self, _, session_class) -> None:
        session_class.return_value.submit.return_value = "done"
        with patch.dict(os.environ, {"TINYHARNESS_API_KEY": "secret"}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "task",
                        "--workspace",
                        str(self.workspace),
                        "--no-context-compaction",
                    ]
                )

        self.assertIsNone(session_class.call_args.kwargs["max_context_tokens"])
        self.assertNotIn("Use compact", session_class.call_args.args[2])

    @patch("tiny_harness.__main__.AgentSession")
    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_can_explicitly_enable_persistent_memory(
        self, _, session_class
    ) -> None:
        session_class.return_value.submit.return_value = "done"
        with patch.dict(os.environ, {"TINYHARNESS_API_KEY": "secret"}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "remember my preference",
                        "--workspace",
                        str(self.workspace),
                        "--memory",
                    ]
                )

        self.assertTrue(session_class.call_args.kwargs["memory_enabled"])
        system_prompt = session_class.call_args.args[2]
        self.assertIn("Persistent Memory", system_prompt)
        self.assertIn("untrusted historical data", system_prompt)

    @patch("tiny_harness.__main__.AgentSession")
    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_no_task_starts_repl_and_reuses_one_session(self, _, session_class) -> None:
        session = session_class.return_value
        session.submit.side_effect = ["first answer", "second answer"]

        with patch.dict(os.environ, {"TINYHARNESS_API_KEY": "secret"}, clear=True):
            with patch("builtins.input", side_effect=["first", "second", "exit"]):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(["--workspace", str(self.workspace)])

        self.assertEqual(exit_code, 0)
        session_class.assert_called_once()
        self.assertEqual(
            [call.args for call in session.submit.call_args_list],
            [("first",), ("second",)],
        )
        output = stdout.getvalue()
        self.assertIn("TinyHarness 交互会话", output)
        self.assertIn("first answer", output)
        self.assertIn("second answer", output)

    @patch("tiny_harness.__main__.AgentSession")
    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_repl_commands_are_not_submitted(self, _, session_class) -> None:
        session = session_class.return_value
        with patch.dict(os.environ, {"TINYHARNESS_API_KEY": "secret"}, clear=True):
            with patch("builtins.input", side_effect=["", "/help", "/clear", "q"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main(["--workspace", str(self.workspace)])

        self.assertEqual(exit_code, 0)
        session.clear.assert_called_once_with()
        session.submit.assert_not_called()

    @patch("tiny_harness.__main__.AgentSession")
    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_repl_eof_exits_cleanly(self, _, session_class) -> None:
        with patch.dict(os.environ, {"TINYHARNESS_API_KEY": "secret"}, clear=True):
            with patch("builtins.input", side_effect=EOFError):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["--workspace", str(self.workspace)]), 0)
        session_class.return_value.submit.assert_not_called()

    def test_requires_api_key(self) -> None:
        stderr = io.StringIO()
        with patch.dict(os.environ, {}, clear=True):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaisesRegex(SystemExit, "2"):
                    main(["task", "--workspace", str(self.workspace)])
        self.assertIn("必须设置 TINYHARNESS_API_KEY", stderr.getvalue())

    def test_rejects_non_directory_workspace(self) -> None:
        missing = self.workspace / "missing"
        stderr = io.StringIO()
        with patch.dict(os.environ, {"TINYHARNESS_API_KEY": "secret"}, clear=True):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaisesRegex(SystemExit, "2"):
                    main(["task", "--workspace", str(missing)])
        self.assertIn("workspace 不是目录", stderr.getvalue())

    def test_rejects_invalid_numeric_arguments(self) -> None:
        cases = [
            (["task", "--max-turns", "0"], "必须大于或等于 1"),
            (["task", "--max-context-tokens", "0"], "必须大于或等于 1"),
            (["task", "--subagent-max-turns", "0"], "必须大于或等于 1"),
            (["task", "--max-model-retries", "-1"], "必须大于或等于 0"),
            (["task", "--task-state-reflection-interval", "-1"], "必须大于或等于 0"),
            (["task", "--task-state-reflection-interval", "1.5"], "invalid"),
            (["task", "--task-state-reflection-interval", "abc"], "invalid"),
        ]
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaisesRegex(SystemExit, "2"):
                        main(arguments)
                self.assertIn(expected, stderr.getvalue())

    def test_allows_zero_model_retries(self) -> None:
        with patch.dict(os.environ, {"TINYHARNESS_API_KEY": "secret"}, clear=True):
            with patch("tiny_harness.__main__.ChatCompletionsProvider"):
                with patch("tiny_harness.__main__.AgentSession") as session_class:
                    session_class.return_value.submit.return_value = "done"
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(
                            main(
                                [
                                    "task",
                                    "--workspace",
                                    str(self.workspace),
                                    "--max-model-retries",
                                    "0",
                                ]
                            ),
                            0,
                        )

    def test_context_options_are_mutually_exclusive(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(SystemExit, "2"):
                main(
                    [
                        "task",
                        "--max-context-tokens",
                        "2250",
                        "--no-context-compaction",
                    ]
                )
        self.assertIn("not allowed with argument", stderr.getvalue())

    def test_permission_prompt_accepts_yes_and_displays_arguments(self) -> None:
        stdout = io.StringIO()
        with patch("builtins.input", return_value="yes"):
            with contextlib.redirect_stdout(stdout):
                allowed = _ask_permission("bash", {"command": "echo hello"})
        self.assertTrue(allowed)
        self.assertIn("需要工具授权", stdout.getvalue())
        self.assertIn('"command": "echo hello"', stdout.getvalue())

    def test_permission_prompt_defaults_to_denial(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            with patch("builtins.input", return_value=""):
                self.assertFalse(_ask_permission("bash", {"command": "echo no"}))
            with patch("builtins.input", side_effect=EOFError):
                self.assertFalse(_ask_permission("bash", {"command": "echo no"}))


if __name__ == "__main__":
    unittest.main()
