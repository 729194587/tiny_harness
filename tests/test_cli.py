import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tiny_harness.__main__ import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MODEL,
    main,
)
from tiny_harness.agent.context import DEFAULT_SUBAGENT_MAX_TURNS
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.agent.turn import TOOL_USE_EFFICIENCY_GUIDANCE
from tiny_harness.runtime.events import CompositeEventLogger, JsonlEventLogger
from tiny_harness.runtime.console import ConsoleEventLogger
from tiny_harness.runtime.recovery import RecoveryPolicy


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)
        # Isolate CLI dotenv loading from any developer configuration.
        cwd_patch = patch("tiny_harness.__main__.Path.cwd", return_value=self.workspace)
        cwd_patch.start()
        self.addCleanup(cwd_patch.stop)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @patch("tiny_harness.__main__.AgentSession")
    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_dotenv_configuration_and_environment_precedence(self, provider_class, session_class):
        session_class.return_value.submit.return_value = "done"
        (self.workspace / ".env").write_text(
            'TINYHARNESS_API_KEY="file-secret"\n'
            'TINYHARNESS_MODEL=file-model\n'
            'TINYHARNESS_BASE_URL=https://file.example.test\n', encoding="utf-8")
        target = self.workspace / "target"
        target.mkdir()
        (target / ".env").write_text("TINYHARNESS_API_KEY=wrong-workspace-key\n", encoding="utf-8")
        for environment, expected in (
            ({}, {"api_key": "file-secret", "model": "file-model", "base_url": "https://file.example.test"}),
            ({"TINYHARNESS_API_KEY": "process-secret", "TINYHARNESS_MODEL": "process-model",
              "TINYHARNESS_BASE_URL": "https://process.example.test"},
             {"api_key": "process-secret", "model": "process-model", "base_url": "https://process.example.test"}),
        ):
            with self.subTest(environment=bool(environment)), patch.dict(os.environ, environment, clear=True), \
                    contextlib.redirect_stdout(io.StringIO()):
                provider_class.reset_mock()
                self.assertEqual(main(["task", "--workspace", str(target)]), 0)
                provider_class.assert_called_once_with(**expected)

    @patch("tiny_harness.__main__.AgentSession")
    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_dotenv_with_only_api_key_keeps_defaults(self, provider_class, session_class):
        session_class.return_value.submit.return_value = "done"
        (self.workspace / ".env").write_text("TINYHARNESS_API_KEY=file-secret\n", encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["task", "--workspace", str(self.workspace)]), 0)
        provider_class.assert_called_once_with(
            api_key="file-secret", model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL)

    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_dotenv_does_not_search_parent_or_workspace(self, provider_class):
        startup = self.workspace / "startup"
        startup.mkdir()
        target = self.workspace / "target"
        target.mkdir()
        for directory in (self.workspace, target):
            (directory / ".env").write_text("TINYHARNESS_API_KEY=must-not-load\n", encoding="utf-8")
        with patch("tiny_harness.__main__.Path.cwd", return_value=startup), \
                patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["task", "--workspace", str(target)])
        self.assertEqual(error.exception.code, 2)
        provider_class.assert_not_called()

    def test_cli_provider_request_includes_shared_efficiency_guidance(self):
        requests = []

        class Provider:
            def complete(self, messages, tools):
                requests.append(messages)
                return ModelResponse("done", None, [], "stop")

        stdout = io.StringIO()
        with patch("tiny_harness.__main__.ChatCompletionsProvider", return_value=Provider()), \
                patch("pathlib.Path.home", return_value=self.workspace), \
                patch.dict(os.environ, {"TINYHARNESS_API_KEY": "secret"}, clear=True), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            exit_code = main(["inspect", "--workspace", str(self.workspace)])

        self.assertEqual(exit_code, 0)
        self.assertTrue(stdout.getvalue().startswith("done\n"))
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].count(
            {"role": "system", "content": TOOL_USE_EFFICIENCY_GUIDANCE}), 1)
        self.assertIn("Prefer targeted investigation", requests[0][0]["content"])

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
        self.assertNotIn("Use compact", positional[2])
        self.assertIn("Treat TinyHarness context summaries", positional[2])
        self.assertEqual(session_class.call_args.kwargs["max_turns"], 7)
        self.assertNotIn("permission_prompt", session_class.call_args.kwargs)
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
        session_class.return_value.submit.assert_called_once_with("create a file")

    def test_removed_flags_are_rejected(self) -> None:
        for flag in ("--task-state", "--task-state-reflection", "--task-state-reflection-interval",
                     "--working-memory", "--progress", "--working-context-trigger-tokens",
                     "--working-context-target-tokens"):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["task", flag])
                self.assertEqual(error.exception.code, 2)

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

    def test_cli_executes_mutating_bash_without_permission_input(self) -> None:
        command = "echo ok > created.txt"
        responses = iter([
            ModelResponse(None, None, [
                ToolCall("write-shell", "bash", json.dumps({"command": command}))
            ], "tool_calls"),
            ModelResponse("done", None, [], "stop"),
        ])
        requests = []

        class Provider:
            def complete(self, messages, tools):
                requests.append(list(messages))
                return next(responses)

        stdout = io.StringIO()
        with patch("tiny_harness.__main__.ChatCompletionsProvider", return_value=Provider()), \
                patch("pathlib.Path.home", return_value=self.workspace), \
                patch.dict(os.environ, {"TINYHARNESS_API_KEY": "secret"}, clear=False), \
                patch("builtins.input", side_effect=AssertionError("unexpected prompt")) as user_input, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["create a file", "--workspace", str(self.workspace)]), 0)

        user_input.assert_not_called()
        self.assertTrue(any(message.get("role") == "tool" and
                            "Exit code: 0" in message["content"] for message in requests[1]))
        self.assertEqual((self.workspace / "created.txt").read_text().strip(), "ok")
        self.assertTrue(stdout.getvalue().startswith("done\n"))


if __name__ == "__main__":
    unittest.main()
