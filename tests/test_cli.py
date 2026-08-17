import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tiny_harness.__main__ import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    _ask_permission,
    main,
)
from tiny_harness.runtime.events import NULL_EVENT_LOGGER, JsonlEventLogger


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @patch("tiny_harness.__main__.agent_loop", return_value="final answer")
    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_wires_cli_to_provider_and_agent_loop(self, provider_class, loop) -> None:
        provider = provider_class.return_value
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
        loop.assert_called_once()
        positional = loop.call_args.args
        self.assertIs(positional[0], provider)
        self.assertEqual(positional[1], self.workspace.resolve())
        self.assertEqual(positional[2][0]["role"], "system")
        self.assertIn(str(self.workspace.resolve()), positional[2][0]["content"])
        self.assertIn("todo_write", positional[2][0]["content"])
        self.assertEqual(
            positional[2][1],
            {"role": "user", "content": "create a file"},
        )
        self.assertEqual(loop.call_args.kwargs["max_turns"], 7)
        self.assertIs(loop.call_args.kwargs["permission_prompt"], _ask_permission)
        self.assertIs(loop.call_args.kwargs["event_logger"], NULL_EVENT_LOGGER)
        self.assertIsNone(loop.call_args.kwargs["max_context_chars"])

    @patch("tiny_harness.__main__.agent_loop", return_value="done")
    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_uses_environment_model_and_base_url(self, provider_class, _) -> None:
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

    @patch("tiny_harness.__main__.agent_loop", return_value="done")
    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_passes_jsonl_logger_when_event_log_is_set(self, _, loop) -> None:
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

        logger = loop.call_args.kwargs["event_logger"]
        self.assertIsInstance(logger, JsonlEventLogger)
        self.assertEqual(logger.path, log_path.resolve())

    @patch("tiny_harness.__main__.agent_loop", return_value="done")
    @patch("tiny_harness.__main__.ChatCompletionsProvider")
    def test_passes_context_budget_when_configured(self, _, loop) -> None:
        with patch.dict(os.environ, {"TINYHARNESS_API_KEY": "secret"}, clear=True):
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "task",
                        "--workspace",
                        str(self.workspace),
                        "--max-context-chars",
                        "9000",
                    ]
                )

        self.assertEqual(loop.call_args.kwargs["max_context_chars"], 9000)

    def test_requires_api_key(self) -> None:
        stderr = io.StringIO()

        with patch.dict(os.environ, {}, clear=True):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaisesRegex(SystemExit, "2"):
                    main(["task", "--workspace", str(self.workspace)])

        self.assertIn("TINYHARNESS_API_KEY is required", stderr.getvalue())

    def test_rejects_non_directory_workspace(self) -> None:
        missing = self.workspace / "missing"
        stderr = io.StringIO()

        with patch.dict(os.environ, {"TINYHARNESS_API_KEY": "secret"}, clear=True):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaisesRegex(SystemExit, "2"):
                    main(["task", "--workspace", str(missing)])

        self.assertIn("workspace is not a directory", stderr.getvalue())

    def test_rejects_non_positive_max_turns(self) -> None:
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(SystemExit, "2"):
                main(["task", "--max-turns", "0"])

        self.assertIn("must be at least 1", stderr.getvalue())

    def test_rejects_non_positive_context_budget(self) -> None:
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(SystemExit, "2"):
                main(["task", "--max-context-chars", "0"])

        self.assertIn("must be at least 1", stderr.getvalue())

    def test_permission_prompt_accepts_yes_and_displays_arguments(self) -> None:
        stdout = io.StringIO()

        with patch("builtins.input", return_value="yes"):
            with contextlib.redirect_stdout(stdout):
                allowed = _ask_permission("bash", {"command": "echo hello"})

        self.assertTrue(allowed)
        self.assertIn("Permission required", stdout.getvalue())
        self.assertIn('"command": "echo hello"', stdout.getvalue())

    def test_permission_prompt_defaults_to_denial(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            with patch("builtins.input", return_value=""):
                self.assertFalse(_ask_permission("bash", {"command": "echo no"}))
            with patch("builtins.input", side_effect=EOFError):
                self.assertFalse(_ask_permission("bash", {"command": "echo no"}))


if __name__ == "__main__":
    unittest.main()
