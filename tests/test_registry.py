import json
import sys
import tempfile
import unittest
from pathlib import Path

from tiny_harness.agent.messages import ToolCall
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.tools.registry import dispatch, tool_schemas


class ToolRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name) / "workspace"
        self.workspace.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def call(self, call_id: str, name: str, arguments: object, **dispatch_options):
        return dispatch(
            self.workspace,
            ToolCall(call_id, name, json.dumps(arguments)),
            **dispatch_options,
        )

    def test_schemas_contain_exactly_the_phase_one_tools(self) -> None:
        schemas = tool_schemas()

        self.assertEqual(
            [schema["function"]["name"] for schema in schemas],
            ["read_file", "write_file", "edit_file", "list_files", "bash"],
        )
        for schema in schemas:
            self.assertEqual(schema["type"], "function")
            self.assertEqual(schema["function"]["parameters"]["type"], "object")

    def test_dispatches_file_tools_and_preserves_call_id(self) -> None:
        write_result = self.call(
            "write-1",
            "write_file",
            {"path": "example.txt", "content": "hello"},
        )
        read_result = self.call("read-1", "read_file", {"path": "example.txt"})

        self.assertEqual(write_result.tool_call_id, "write-1")
        self.assertEqual(write_result.content, "Wrote 5 bytes to example.txt")
        self.assertEqual(read_result.tool_call_id, "read-1")
        self.assertEqual(read_result.content, "hello")

    def test_dispatches_bash_in_workspace(self) -> None:
        command = f'"{sys.executable}" -c "from pathlib import Path; print(Path.cwd().name)"'

        result = self.call(
            "bash-1",
            "bash",
            {"command": command},
            permission_prompt=lambda *_: True,
        )

        self.assertEqual(result.tool_call_id, "bash-1")
        self.assertEqual(result.content, "workspace")

    def test_denied_bash_is_not_executed_and_preserves_call_id(self) -> None:
        command = (
            f'"{sys.executable}" -c '
            '"from pathlib import Path; Path(\'blocked.txt\').write_text(\'bad\')"'
        )

        result = self.call(
            "bash-denied",
            "bash",
            {"command": command},
            permission_prompt=lambda *_: False,
        )

        self.assertEqual(result.tool_call_id, "bash-denied")
        self.assertEqual(result.content, "Error: Permission denied for tool bash")
        self.assertFalse((self.workspace / "blocked.txt").exists())

    def test_bash_without_prompt_is_denied(self) -> None:
        result = self.call("bash-no-prompt", "bash", {"command": "echo no"})

        self.assertEqual(result.content, "Error: Permission denied for tool bash")

    def test_explicit_deny_does_not_execute_handler(self) -> None:
        class DenyPolicy:
            def decide(self, tool_name, arguments):
                return PermissionDecision.DENY

        result = self.call(
            "write-denied",
            "write_file",
            {"path": "blocked.txt", "content": "bad"},
            permission_policy=DenyPolicy(),
        )

        self.assertEqual(result.tool_call_id, "write-denied")
        self.assertEqual(
            result.content,
            "Error: Permission denied for tool write_file",
        )
        self.assertFalse((self.workspace / "blocked.txt").exists())

    def test_file_tools_do_not_prompt(self) -> None:
        def unexpected_prompt(tool_name, arguments):
            raise AssertionError("file tool should not prompt")

        result = self.call(
            "write-allowed",
            "write_file",
            {"path": "allowed.txt", "content": "ok"},
            permission_prompt=unexpected_prompt,
        )

        self.assertEqual(result.content, "Wrote 2 bytes to allowed.txt")
        self.assertEqual(
            (self.workspace / "allowed.txt").read_text(encoding="utf-8"),
            "ok",
        )

    def test_invalid_json_becomes_tool_result(self) -> None:
        result = dispatch(
            self.workspace,
            ToolCall("bad-json", "read_file", "{"),
        )

        self.assertEqual(result.tool_call_id, "bad-json")
        self.assertTrue(result.content.startswith("Error: JSONDecodeError:"))

    def test_non_object_arguments_become_tool_result(self) -> None:
        result = self.call("bad-shape", "read_file", ["example.txt"])

        self.assertEqual(
            result.content,
            "Error: ValueError: Tool arguments must be a JSON object",
        )

    def test_unknown_tool_becomes_tool_result(self) -> None:
        result = self.call("unknown-1", "missing_tool", {})

        self.assertEqual(result.content, "Error: ValueError: Unknown tool: missing_tool")

    def test_handler_parameter_error_becomes_tool_result(self) -> None:
        result = self.call("bad-args", "read_file", {})

        self.assertTrue(result.content.startswith("Error: TypeError:"))

    def test_handler_exception_becomes_tool_result(self) -> None:
        result = self.call("missing-file", "read_file", {"path": "missing.txt"})

        self.assertTrue(result.content.startswith("Error: FileNotFoundError:"))

    def test_workspace_escape_becomes_tool_result(self) -> None:
        result = self.call("escape", "write_file", {"path": "../outside.txt", "content": "no"})

        self.assertTrue(result.content.startswith("Error: ValueError: Path escapes workspace:"))
        self.assertFalse((self.workspace.parent / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()
