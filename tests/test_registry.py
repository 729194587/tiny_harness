import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from tiny_harness.agent.messages import ToolCall
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.runtime.context import CompactionRequest
from tiny_harness.runtime.todos import TodoManager
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

    def test_schemas_contain_all_default_runtime_tools(self) -> None:
        schemas = tool_schemas()

        self.assertEqual(
            [schema["function"]["name"] for schema in schemas],
            [
                "read_file",
                "write_file",
                "edit_file",
                "list_files",
                "bash",
                "todo_write",
                "task",
            ],
        )
        for schema in schemas:
            self.assertEqual(schema["type"], "function")
            self.assertEqual(schema["function"]["parameters"]["type"], "object")

        todo_schema = next(
            schema
            for schema in schemas
            if schema["function"]["name"] == "todo_write"
        )
        todo_parameters = todo_schema["function"]["parameters"]
        self.assertEqual(todo_parameters["properties"]["todos"]["maxItems"], 20)

        child_names = [
            schema["function"]["name"]
            for schema in tool_schemas(include_task=False)
        ]
        self.assertNotIn("task", child_names)

        compact_names = [
            schema["function"]["name"]
            for schema in tool_schemas(include_compact=True)
        ]
        self.assertIn("compact", compact_names)

    def test_dispatches_compact_only_with_run_scoped_request(self) -> None:
        manager = CompactionRequest()

        result = self.call(
            "compact-1",
            "compact",
            {},
            compaction_request=manager,
        )

        self.assertEqual(manager.revision, 1)
        self.assertEqual(
            result.content,
            "Compaction requested after this tool batch.",
        )

        missing_manager = self.call("compact-2", "compact", {})
        self.assertEqual(
            missing_manager.content,
            "Error: ValueError: Unknown tool: compact",
        )

    def test_dispatches_task_through_injected_runner(self) -> None:
        observed = []

        result = self.call(
            "task-1",
            "task",
            {"prompt": " inspect the project "},
            subagent_runner=lambda prompt, call_id: (
                observed.append((prompt, call_id)) or "child summary"
            ),
        )

        self.assertEqual(observed, [("inspect the project", "task-1")])
        self.assertEqual(result.tool_call_id, "task-1")
        self.assertEqual(result.content, "child summary")

    def test_task_without_runner_is_unknown(self) -> None:
        result = self.call(
            "task-1",
            "task",
            {"prompt": "inspect"},
        )

        self.assertEqual(
            result.content,
            "Error: ValueError: Unknown tool: task",
        )

    def test_empty_task_prompt_does_not_call_runner(self) -> None:
        observed = []

        result = self.call(
            "task-1",
            "task",
            {"prompt": "   "},
            subagent_runner=lambda prompt, call_id: observed.append(
                (prompt, call_id)
            ),
        )

        self.assertEqual(observed, [])
        self.assertEqual(
            result.content,
            "Error: ValueError: prompt must be a non-empty string",
        )

    def test_dispatches_todo_write_with_run_scoped_manager(self) -> None:
        manager = TodoManager()

        with contextlib.redirect_stdout(io.StringIO()):
            result = self.call(
                "todo-1",
                "todo_write",
                {
                    "todos": [
                        {
                            "content": "Implement the feature",
                            "status": "in_progress",
                        }
                    ]
                },
                todo_manager=manager,
            )

        self.assertEqual(result.tool_call_id, "todo-1")
        self.assertIn("[>] Implement the feature", result.content)
        self.assertEqual(manager.revision, 1)

    def test_todo_write_without_manager_becomes_tool_error(self) -> None:
        result = self.call(
            "todo-1",
            "todo_write",
            {"todos": []},
        )

        self.assertEqual(result.tool_call_id, "todo-1")
        self.assertEqual(
            result.content,
            "Error: RuntimeError: todo_write requires a TodoManager",
        )

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
        self.assertEqual(result.content, "Exit code: 0\nworkspace")

    def test_bash_reports_nonzero_exit_code_as_evidence(self) -> None:
        command = f'"{sys.executable}" -c "raise SystemExit(7)"'

        result = self.call(
            "bash-failed",
            "bash",
            {"command": command},
            permission_prompt=lambda *_: True,
        )

        self.assertEqual(result.tool_call_id, "bash-failed")
        self.assertEqual(result.content, "Exit code: 7")

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
