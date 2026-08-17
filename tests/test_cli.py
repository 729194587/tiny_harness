import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tiny_harness.__main__ import DEFAULT_BASE_URL, DEFAULT_MODEL, main


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
        self.assertEqual(
            positional[2][1],
            {"role": "user", "content": "create a file"},
        )
        self.assertEqual(loop.call_args.kwargs, {"max_turns": 7})

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


if __name__ == "__main__":
    unittest.main()
