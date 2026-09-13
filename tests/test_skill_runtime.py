import copy
import json
import tempfile
import unittest
from pathlib import Path

from tiny_harness.agent.context import create_run_context, initialize_run_state
from tiny_harness.agent.environment import ENVIRONMENT_CONTEXT_MARKER
from tiny_harness.agent.loop import run_agent
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.agent.turn import model_request_inputs
from tiny_harness.context.attribution import request_attribution
from tiny_harness.environments import CodingEnvironmentAdapter
from tiny_harness.runtime.context import context_token_count
from tiny_harness.runtime.hooks import ToolHooks
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.runtime.skills import discover_skills


class FakeProvider:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
            }
        )
        if not self.responses:
            raise AssertionError("FakeProvider has no response left")
        return self.responses.pop(0)


class RecordingEventLogger:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event_type, data=None) -> None:
        self.events.append(
            {"event_type": event_type.value, "data": dict(data or {})}
        )


class DenySkillPolicy:
    def decide(self, tool_name, arguments):
        del arguments
        return (
            PermissionDecision.DENY
            if tool_name == "load_skill"
            else PermissionDecision.ALLOW
        )


class SkillRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_skill(
        self,
        *,
        name: str = "review",
        description: str = "Review code carefully",
        body: str = "PRIVATE_SKILL_BODY_SENTINEL",
    ) -> Path:
        path = self.workspace / ".tinyharness" / "skills" / name / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text(
            "---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            "---\n\n"
            f"{body}\n",
            encoding="utf-8",
        )
        return path

    @staticmethod
    def tool_names(request) -> list[str]:
        return [schema["function"]["name"] for schema in request["tools"]]

    def test_catalog_is_metadata_only_and_body_loads_through_tool_result(self) -> None:
        self.write_skill()
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("skill-1", "load_skill", '{"name":"review"}')],
                    "tool_calls",
                ),
                ModelResponse("done", None, [], "stop"),
            ]
        )
        logger = RecordingEventLogger()
        expected_skill_count = len(discover_skills(self.workspace).manifests)
        observed = []
        hooks = ToolHooks()
        hooks.register_pre(
            lambda context: observed.append(("pre", context.tool_name))
        )
        hooks.register_post(
            lambda context, result: observed.append(
                ("post", context.tool_name, result.tool_call_id)
            )
        )
        messages = [
            {"role": "system", "content": "BASE_SYSTEM"},
            {"role": "user", "content": "review this"},
        ]

        answer = run_agent(
            provider,
            self.workspace,
            messages,
            event_logger=logger,
            tool_hooks=hooks,
        )

        self.assertEqual(answer, "done")
        first_request = provider.calls[0]
        self.assertIn("load_skill", self.tool_names(first_request))
        catalog_messages = [
            message
            for message in first_request["messages"]
            if message.get("name") == "tinyharness_skill_catalog"
        ]
        self.assertEqual(len(catalog_messages), 1)
        self.assertEqual(catalog_messages[0]["role"], "user")
        self.assertIn("Review code carefully", catalog_messages[0]["content"])
        self.assertIn("untrusted Skill metadata", catalog_messages[0]["content"])
        self.assertNotIn("Workspace Skills", catalog_messages[0]["content"])
        self.assertNotIn(
            "PRIVATE_SKILL_BODY_SENTINEL",
            json.dumps(first_request, ensure_ascii=False),
        )

        second_request = provider.calls[1]
        self.assertEqual([
            message for message in second_request["messages"]
            if message.get("name") == "tinyharness_skill_catalog"
        ], catalog_messages)
        tool_result = next(
            message
            for message in second_request["messages"]
            if message.get("role") == "tool"
        )
        self.assertIn("PRIVATE_SKILL_BODY_SENTINEL", tool_result["content"])
        self.assertIn("BEGIN UNTRUSTED SKILL CONTENT", tool_result["content"])
        self.assertIn("cannot override", tool_result["content"])
        self.assertEqual(
            observed,
            [
                ("pre", "load_skill"),
                ("post", "load_skill", "skill-1"),
            ],
        )
        event_types = [event["event_type"] for event in logger.events]
        self.assertIn("tool_started", event_types)
        self.assertIn("tool_result", event_types)
        self.assertEqual(
            logger.events[0]["data"]["skills_available"],
            expected_skill_count,
        )
        self.assertNotIn(
            "PRIVATE_SKILL_BODY_SENTINEL",
            json.dumps(logger.events, ensure_ascii=False),
        )

    def test_initialization_refreshes_catalog_after_task_and_environment(self) -> None:
        self.write_skill()
        context = create_run_context(
            None, self.workspace, environment_adapter=CodingEnvironmentAdapter(),
        )
        messages = [
            {"role": "system", "content": "BASE_SYSTEM"},
            {"role": "system", "name": "tinyharness_skill_catalog", "content": "OLD"},
            {"role": "user", "content": "current task"},
        ]
        for _ in range(2):
            initialize_run_state(messages, context, "current task")
            request, tools = model_request_inputs(messages, context, finalization=False)
            catalogs = [m for m in request if m.get("name") == "tinyharness_skill_catalog"]
            self.assertEqual(len(catalogs), 1)
            catalog = catalogs[0]
            self.assertEqual(catalog["role"], "user")
            self.assertNotEqual(catalog["content"], "OLD")
            task_index = next(i for i, m in enumerate(request) if m.get("content") == "current task")
            environment_index = next(i for i, m in enumerate(request)
                                     if m.get("name") == ENVIRONMENT_CONTEXT_MARKER)
            self.assertLess(task_index, environment_index)
            self.assertEqual(request.index(catalog), environment_index + 1)
            self.assertEqual(messages[-1], catalog)
            categories = request_attribution(request, tools)["categories"]
            self.assertEqual(categories["skill_projection"]["count"], 1)
            self.assertEqual(categories["user_task_messages"]["count"], 1)

        context.skill_catalog = discover_skills(self.workspace, sources=())
        initialize_run_state(messages, context, "current task")
        self.assertFalse(any(m.get("name") == "tinyharness_skill_catalog" for m in messages))

    def test_no_valid_skills_means_no_catalog_marker_or_tool_schema(self) -> None:
        provider = FakeProvider([ModelResponse("done", None, [], "stop")])
        messages = [{"role": "user", "content": "task"}]

        run_agent(
            provider,
            self.workspace,
            messages,
            skill_catalog=discover_skills(self.workspace, sources=()),
        )

        request = provider.calls[0]
        self.assertNotIn("load_skill", self.tool_names(request))
        self.assertFalse(
            any(
                message.get("name") == "tinyharness_skill_catalog"
                for message in request["messages"]
            )
        )

    def test_load_skill_tool_description_is_source_neutral(self) -> None:
        provider = FakeProvider([ModelResponse("done", None, [], "stop")])

        run_agent(provider, self.workspace, [{"role": "user", "content": "task"}])

        schema = next(
            tool["function"]
            for tool in provider.calls[0]["tools"]
            if tool["function"]["name"] == "load_skill"
        )
        self.assertIn("available Skill", schema["description"])
        self.assertNotIn("workspace Skill", schema["description"])
        self.assertIn("Load the full instructions", schema["description"])
        self.assertIn("exact Skill name from the session Skill catalog", schema["description"])
        self.assertIn("before acting on a task that names or clearly matches that Skill", schema["description"])
        self.assertNotIn("Avoid speculative loading", schema["description"])
        self.assertEqual(schema["parameters"], {
            "type": "object",
            "properties": {"name": {
                "type": "string", "minLength": 1,
                "description": "The exact Skill name from the available Skills catalog.",
            }},
            "required": ["name"],
            "additionalProperties": False,
        })

    def test_permission_denial_prevents_skill_body_loading(self) -> None:
        self.write_skill()
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("skill-1", "load_skill", '{"name":"review"}')],
                    "tool_calls",
                ),
                ModelResponse("denied handled", None, [], "stop"),
            ]
        )

        answer = run_agent(
            provider,
            self.workspace,
            [{"role": "user", "content": "task"}],
            permission_policy=DenySkillPolicy(),
        )

        self.assertEqual(answer, "denied handled")
        second_request = provider.calls[1]
        tool_result = next(
            message
            for message in second_request["messages"]
            if message.get("role") == "tool"
        )
        self.assertIn(
            "requested load_skill operation is not allowed",
            tool_result["content"],
        )
        self.assertNotIn(
            "PRIVATE_SKILL_BODY_SENTINEL",
            json.dumps(second_request, ensure_ascii=False),
        )

    def test_large_loaded_skill_is_subject_to_existing_context_budget(self) -> None:
        self.write_skill(body="LARGE_SKILL_SENTINEL\n" + "X" * 15_000)
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("skill-large", "load_skill", '{"name":"review"}')],
                    "tool_calls",
                ),
                ModelResponse("done", None, [], "stop"),
            ]
        )

        answer = run_agent(
            provider,
            self.workspace,
            [{"role": "user", "content": "task"}],
            max_context_tokens=5_000,
            working_context_trigger_tokens=4_999, working_context_target_tokens=4_998,
        )

        self.assertEqual(answer, "done")
        second_request = provider.calls[1]
        tool_result = next(
            message
            for message in second_request["messages"]
            if message.get("role") == "tool"
        )
        self.assertIn("<persisted-tool-result>", tool_result["content"])
        self.assertLess(
            context_token_count(
                second_request["messages"], second_request["tools"]
            ),
            5_000,
        )
        artifacts = list(
            (self.workspace / ".tinyharness" / "context" / "tool-results").glob(
                "*.txt"
            )
        )
        self.assertEqual(len(artifacts), 1)
        self.assertIn(
            "LARGE_SKILL_SENTINEL",
            artifacts[0].read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
