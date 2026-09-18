import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tiny_harness.agent.messages import ToolCall
from tiny_harness.runtime.events import NULL_EVENT_LOGGER
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.runtime.context import CompactionRequest
from tiny_harness.runtime.skills import discover_skills
from tiny_harness.runtime.todos import TodoManager
from tiny_harness.tools.discovery import discover_tools
from tiny_harness.tools.registry import dispatch


class RecordingTestRunner:
    def __init__(self, output: str = "Exit code: 0\nOK") -> None:
        self.output = output
        self.workspaces = []

    def run(self, workspace: Path) -> str:
        self.workspaces.append(workspace)
        return self.output


class ToolRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name) / "workspace"
        self.workspace.mkdir()
        self.todo_manager = TodoManager()
        self.registry = self.make_registry()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_registry(self, **overrides):
        capabilities = {
            "workspace": self.workspace,
            "event_logger": NULL_EVENT_LOGGER,
            "todo_manager": self.todo_manager,
            "subagent_runner": lambda prompt, call_id: "child summary",
            "skill_catalog": discover_skills(self.workspace),
            "compaction_request": None,
            "test_runner": None,
        }
        capabilities.update(overrides)
        return discover_tools(SimpleNamespace(**capabilities))

    def call(
        self,
        call_id: str,
        name: str,
        arguments: object,
        *,
        registry=None,
        **dispatch_options,
    ):
        return dispatch(
            registry or self.registry,
            ToolCall(call_id, name, json.dumps(arguments)),
            **dispatch_options,
        )

    def test_schemas_contain_all_default_runtime_tools(self) -> None:
        schemas = self.registry.model_schemas()

        self.assertEqual(
            [schema["function"]["name"] for schema in schemas],
            [
                "read_file",
                "write_file",
                "edit_file",
                "list_files",
                "search_code",
                "glob",
                "grep",
                "bash",
                "load_skill",
                "task",
                "todo_write",
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
            for schema in self.make_registry(
                subagent_runner=None
            ).model_schemas()
        ]
        self.assertNotIn("task", child_names)

        compact_names = [
            schema["function"]["name"]
            for schema in self.make_registry(
                compaction_request=CompactionRequest()
            ).model_schemas()
        ]
        self.assertIn("compact", compact_names)

        manifest = (
            self.workspace / ".tinyharness" / "skills" / "review" / "SKILL.md"
        )
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            "---\nname: review\ndescription: Review code\n---\n\nBODY",
            encoding="utf-8",
        )
        skill_names = [
            schema["function"]["name"]
            for schema in self.make_registry(
                skill_catalog=discover_skills(self.workspace)
            ).model_schemas()
        ]
        self.assertIn("load_skill", skill_names)

        default_names = [
            schema["function"]["name"] for schema in self.registry.model_schemas()
        ]
        configured_registry = self.make_registry(test_runner=RecordingTestRunner())
        configured_names = [
            schema["function"]["name"]
            for schema in configured_registry.model_schemas()
        ]
        self.assertNotIn("run_tests", default_names)
        self.assertIn("run_tests", configured_names)
        run_tests_schema = next(
            schema
            for schema in configured_registry.model_schemas()
            if schema["function"]["name"] == "run_tests"
        )
        self.assertEqual(
            run_tests_schema["function"]["parameters"],
            {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )

    def test_run_tests_requires_capability_and_accepts_no_arguments(self) -> None:
        unavailable = self.call("tests-missing", "run_tests", {})
        self.assertEqual(
            unavailable.content,
            "Error: ValueError: Unknown tool: run_tests",
        )

        runner = RecordingTestRunner()
        configured = self.make_registry(test_runner=runner)
        result = self.call(
            "tests-1",
            "run_tests",
            {},
            registry=configured,
        )
        self.assertEqual(result.content, "Exit code: 0\nOK")
        self.assertEqual(runner.workspaces, [self.workspace])

        rejected = self.call(
            "tests-args",
            "run_tests",
            {"target": "tests.test_model"},
            registry=configured,
        )
        self.assertIn("run_tests does not accept arguments", rejected.content)
        self.assertEqual(runner.workspaces, [self.workspace])

    def test_dispatches_load_skill_only_with_injected_catalog(self) -> None:
        manifest = (
            self.workspace / ".tinyharness" / "skills" / "review" / "SKILL.md"
        )
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            "---\nname: review\ndescription: Review code\n---\n\nBODY",
            encoding="utf-8",
        )
        catalog = discover_skills(self.workspace)

        result = self.call(
            "skill-1",
            "load_skill",
            {"name": "review"},
            registry=self.make_registry(skill_catalog=catalog),
        )
        missing_catalog = self.call(
            "skill-2",
            "load_skill",
            {"name": "review"},
            registry=self.make_registry(
                skill_catalog=discover_skills(self.workspace, sources=())
            ),
        )

        self.assertIn("BEGIN UNTRUSTED SKILL CONTENT", result.content)
        self.assertIn("BODY", result.content)
        self.assertEqual(
            missing_catalog.content,
            "Error: ValueError: Unknown tool: load_skill",
        )

    def test_dispatches_compact_only_with_run_scoped_request(self) -> None:
        manager = CompactionRequest()

        result = self.call(
            "compact-1",
            "compact",
            {},
            registry=self.make_registry(compaction_request=manager),
        )

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
        registry = self.make_registry(
            subagent_runner=lambda prompt, call_id: (
                observed.append((prompt, call_id)) or "child summary"
            )
        )

        result = self.call(
            "task-1",
            "task",
            {"prompt": " inspect the project "},
            registry=registry,
        )

        self.assertEqual(observed, [("inspect the project", "task-1")])
        self.assertEqual(result.tool_call_id, "task-1")
        self.assertEqual(result.content, "child summary")

    def test_task_without_runner_is_unknown(self) -> None:
        result = self.call(
            "task-1",
            "task",
            {"prompt": "inspect"},
            registry=self.make_registry(subagent_runner=None),
        )

        self.assertEqual(
            result.content,
            "Error: ValueError: Unknown tool: task",
        )

    def test_empty_task_prompt_does_not_call_runner(self) -> None:
        observed = []
        registry = self.make_registry(
            subagent_runner=lambda prompt, call_id: observed.append(
                (prompt, call_id)
            )
        )

        result = self.call(
            "task-1",
            "task",
            {"prompt": "   "},
            registry=registry,
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
                registry=self.make_registry(todo_manager=manager),
            )

        self.assertEqual(result.tool_call_id, "todo-1")
        self.assertIn("[>] Implement the feature", result.content)

    def test_todo_write_without_manager_is_not_registered(self) -> None:
        result = self.call(
            "todo-1",
            "todo_write",
            {"todos": []},
            registry=self.make_registry(todo_manager=None),
        )

        self.assertEqual(result.tool_call_id, "todo-1")
        self.assertEqual(
            result.content,
            "Error: ValueError: Unknown tool: todo_write",
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
        self.assertEqual(read_result.content, "[lines 1-1 of 1 | example.txt]\n\nhello")

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
        self.assertIn("current shell command is not allowed", result.content)
        self.assertIn("Do not retry", result.content)
        self.assertIn("provide the final answer", result.content)
        self.assertFalse((self.workspace / "blocked.txt").exists())

    def test_read_only_bash_without_prompt_is_allowed(self) -> None:
        result = self.call("bash-no-prompt", "bash", {"command": "echo ok"})

        self.assertEqual(result.content, "Exit code: 0\nok")

    def test_ambiguous_bash_without_prompt_is_denied(self) -> None:
        result = self.call(
            "bash-no-prompt",
            "bash",
            {"command": 'python -c "print(\'no\')"'},
        )

        self.assertIn("current shell command is not allowed", result.content)

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
        self.assertIn("requested write_file operation is not allowed", result.content)
        self.assertIn("provide the final answer", result.content)
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
            self.registry,
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
