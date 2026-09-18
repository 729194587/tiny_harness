import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from tiny_harness.agent.messages import ToolCall
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.runtime.skills import discover_skills
from tiny_harness.runtime.todos import TodoManager
from tiny_harness.tools.definition import ToolDefinition
from tiny_harness.tools.discovery import discover_tools
from tiny_harness.tools.registry import ToolRegistry, dispatch


class AllowPolicy:
    def decide(self, tool_name, arguments):
        del tool_name, arguments
        return PermissionDecision.ALLOW


class ToolDefinitionRegistryTest(unittest.TestCase):
    def definition(self, name: str = "echo") -> ToolDefinition:
        return ToolDefinition(
            name=name,
            description="Echo one text value.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            execute=lambda call, arguments: arguments["text"],
        )

    def test_valid_definition_registers_and_is_listed(self) -> None:
        definition = self.definition()
        registry = ToolRegistry()

        registry.register(definition)

        self.assertIs(registry.lookup("echo"), definition)
        self.assertEqual(registry.list(), (definition,))

    def test_duplicate_tool_name_fails_explicitly(self) -> None:
        registry = ToolRegistry()
        registry.register(self.definition())

        with self.assertRaisesRegex(ValueError, "Duplicate tool name: echo"):
            registry.register(self.definition())

    def test_model_schema_is_projected_from_registered_definition(self) -> None:
        definition = self.definition()
        registry = ToolRegistry()
        registry.register(definition)

        self.assertEqual(
            registry.model_schemas(),
            [
                {
                    "type": "function",
                    "function": {
                        "name": definition.name,
                        "description": definition.description,
                        "parameters": definition.parameters,
                    },
                }
            ],
        )

    def test_definition_owns_an_immutable_copy_of_its_parameter_schema(self) -> None:
        parameters = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }
        definition = ToolDefinition(
            name="echo",
            description="Echo text.",
            parameters=parameters,
            execute=lambda call, arguments: arguments["text"],
        )

        parameters["properties"]["constructor_mutation"] = {"type": "string"}
        exposed = definition.parameters
        exposed["properties"]["consumer_mutation"] = {"type": "string"}

        self.assertEqual(
            definition.model_schema()["function"]["parameters"],
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        )

    def test_dispatch_executes_the_definition_registered_for_the_call(self) -> None:
        observed = []
        definition = ToolDefinition(
            name="capture",
            description="Capture one value.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            execute=lambda call, arguments: (
                observed.append((call.id, arguments["value"])) or "captured"
            ),
        )
        registry = ToolRegistry()
        registry.register(definition)

        result = dispatch(
            registry,
            ToolCall("capture-1", "capture", json.dumps({"value": "VALUE"})),
            permission_policy=AllowPolicy(),
        )

        self.assertEqual(observed, [("capture-1", "VALUE")])
        self.assertEqual(result.content, "captured")


class ToolDiscoveryTest(unittest.TestCase):
    def test_discovers_a_module_that_provides_multiple_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            package_name = f"sample_tools_{uuid4().hex}"
            package = root / package_name
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "ignored.py").write_text(
                "VALUE = 'no tool factory here'\n",
                encoding="utf-8",
            )
            (package / "multiple.py").write_text(
                "from tiny_harness.tools.definition import ToolDefinition\n\n"
                "def build_tools(context):\n"
                "    context.factory_calls.append('multiple')\n"
                "    empty = {'type': 'object', 'properties': {}, "
                "'additionalProperties': False}\n"
                "    return (\n"
                "        ToolDefinition('first', 'First tool.', empty, "
                "lambda call, arguments: 'one'),\n"
                "        ToolDefinition('second', 'Second tool.', empty, "
                "lambda call, arguments: 'two'),\n"
                "    )\n",
                encoding="utf-8",
            )
            context = SimpleNamespace(factory_calls=[])
            sys.path.insert(0, str(root))
            try:
                registry = discover_tools(context, package_name=package_name)
            finally:
                sys.path.remove(str(root))
                for module_name in tuple(sys.modules):
                    if module_name == package_name or module_name.startswith(
                        f"{package_name}."
                    ):
                        sys.modules.pop(module_name, None)

        self.assertEqual(context.factory_calls, ["multiple"])
        self.assertEqual(
            [definition.name for definition in registry.list()],
            ["first", "second"],
        )

    def test_builtin_factories_omit_tools_with_missing_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            context = SimpleNamespace(
                workspace=workspace,
                todo_manager=TodoManager(),
                subagent_runner=None,
                skill_catalog=discover_skills(workspace, sources=()),
                test_runner=None,
            )

            registry = discover_tools(context)

        names = [definition.name for definition in registry.list()]
        self.assertEqual(
            names,
            [
                "read_file",
                "write_file",
                "edit_file",
                "list_files",
                "search_code",
                "glob",
                "grep",
                "bash",
                "todo_write",
            ],
        )
        self.assertNotIn("task", names)
        self.assertNotIn("load_skill", names)
        self.assertNotIn("compact", names)
        self.assertNotIn("run_tests", names)

    def test_missing_skill_catalog_capability_omits_load_skill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            context = SimpleNamespace(
                workspace=Path(temporary_directory),
                todo_manager=TodoManager(),
                subagent_runner=None,
                skill_catalog=None,
                test_runner=None,
            )

            registry = discover_tools(context)

        self.assertNotIn(
            "load_skill",
            [definition.name for definition in registry.list()],
        )


if __name__ == "__main__":
    unittest.main()
