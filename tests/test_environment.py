import copy
import tempfile
import unittest
from pathlib import Path

from tiny_harness.agent.context import create_run_context, initialize_run_state
from tiny_harness.agent.environment import ENVIRONMENT_CONTEXT_MARKER
from tiny_harness.agent.loop import run_agent
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.agent.session import AgentSession
from tiny_harness.runtime.context import ContextLimitError, prepare_context
from tiny_harness.runtime.hooks import HookBlock, ToolHooks
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.tools.definition import ToolDefinition
from tiny_harness.tools.discovery import discover_tools


class Provider:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        return self.responses.pop(0)


def answer():
    return ModelResponse("done", None, [], "stop")


class Adapter:
    def __init__(self, name="environment_echo", text="Environment facts", events=None):
        self.name = name
        self.text = text
        self.events = events if events is not None else []
        self.contexts = []

    def build_tools(self, context):
        self.contexts.append(context)

        def execute(call, arguments):
            self.events.append("execute")
            return "environment result"

        return (ToolDefinition(self.name, "Environment tool", {
            "type": "object", "properties": {},
        }, execute),)

    def initial_context(self, context):
        return self.text


class EnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = Path(self.directory.name)

    def test_default_discovery_and_messages_are_unchanged(self):
        context = create_run_context(Provider(), self.workspace)
        self.assertEqual(context.tools, discover_tools(context).model_schemas())
        messages = [{"role": "user", "content": "task"}]
        initialize_run_state(messages, context, "task")
        self.assertFalse(any(m.get("name") == ENVIRONMENT_CONTEXT_MARKER for m in messages))

    def test_additive_tools_are_in_compactor_schemas(self):
        context = create_run_context(Provider(), self.workspace,
                                     environment_adapter=Adapter(), max_context_tokens=25000)
        baseline = discover_tools(context).model_schemas()
        self.assertEqual(context.tools[:-1], baseline)
        self.assertEqual(context.tools[-1]["function"]["name"], "environment_echo")
        self.assertEqual(context.compactor.tools, context.tools)

    def test_duplicate_builtin_is_rejected(self):
        context = create_run_context(Provider(), self.workspace)
        name = context.tool_registry.list()[0].name
        with self.assertRaisesRegex(ValueError, "Duplicate tool name"):
            create_run_context(Provider(), self.workspace, environment_adapter=Adapter(name=name))

    def test_session_context_reaches_model_without_accumulating(self):
        provider = Provider([answer(), answer(), answer()])
        session = AgentSession(provider, self.workspace, "system",
                               environment_adapter=Adapter(), max_context_tokens=25000)
        for task in ("first", "second"):
            self.assertEqual(session.submit(task), "done")
            markers = [m for m in provider.calls[-1][0] if m.get("name") == ENVIRONMENT_CONTEXT_MARKER]
            self.assertEqual(markers, [{"role": "user", "name": ENVIRONMENT_CONTEXT_MARKER,
                                        "content": "Environment facts"}])
            self.assertFalse(any(m.get("name") == ENVIRONMENT_CONTEXT_MARKER for m in session.messages))
        session.clear()
        self.assertEqual(session.messages, [{"role": "system", "content": "system"}])
        session.submit("third")

    def test_initialization_replaces_stale_context_and_budget_is_enforced(self):
        context = create_run_context(Provider(), self.workspace,
                                     environment_adapter=Adapter(text="large context " * 10000),
                                     max_context_tokens=1000)
        messages = [{"role": "user", "content": "task"}]
        initialize_run_state(messages, context, "task")
        initialize_run_state(messages, context, "task")
        self.assertEqual(sum(m.get("name") == ENVIRONMENT_CONTEXT_MARKER for m in messages), 1)
        with self.assertRaises(ContextLimitError):
            prepare_context(messages, context.compactor, "", "task")

    def test_tools_use_existing_permission_hooks_and_execution(self):
        for mode, expected in (("allow", ["pre", "permission", "execute", "post"]),
                               ("deny", ["pre", "permission"]),
                               ("block", ["pre"])):
            with self.subTest(mode=mode):
                events = []
                class Policy:
                    def decide(self, name, arguments):
                        events.append("permission")
                        return PermissionDecision.DENY if mode == "deny" else PermissionDecision.ALLOW
                hooks = ToolHooks()
                def pre(context):
                    events.append("pre")
                    return HookBlock("blocked") if mode == "block" else None
                hooks.register_pre(pre)
                hooks.register_post(lambda context, result: events.append("post"))
                provider = Provider([ModelResponse(None, None, [ToolCall("1", "environment_echo", "{}")],
                                                   "tool_calls"), answer()])
                messages = [{"role": "user", "content": "task"}]
                run_agent(provider, self.workspace, messages, max_turns=3,
                          environment_adapter=Adapter(events=events),
                          permission_policy=Policy(), tool_hooks=hooks)
                self.assertEqual(events, expected)
                result = next(m for m in messages if m.get("role") == "tool")
                self.assertEqual(result["tool_call_id"], "1")

    def test_default_permission_does_not_allow_adapter_tool(self):
        adapter = Adapter()
        provider = Provider([ModelResponse(None, None, [ToolCall("1", adapter.name, "{}")],
                                           "tool_calls"), answer()])
        run_agent(provider, self.workspace, [{"role": "user", "content": "task"}],
                  environment_adapter=adapter, max_turns=3)
        self.assertEqual(adapter.events, [])

    def test_child_rebuilds_adapter_with_fresh_run_state(self):
        adapter = Adapter()
        provider = Provider([answer()])
        parent = create_run_context(provider, self.workspace, environment_adapter=adapter)
        self.assertEqual(parent.subagent_runner("child task", "parent-call"), "done")
        self.assertEqual(len(adapter.contexts), 2)
        child = adapter.contexts[1]
        self.assertIsNot(child.todo_manager, parent.todo_manager)
        self.assertIsNone(child.subagent_runner)
        self.assertTrue(any(m.get("name") == ENVIRONMENT_CONTEXT_MARKER for m in provider.calls[0][0]))

    def test_invalid_context_poisoning_preserves_session_contract(self):
        session = AgentSession(Provider(), self.workspace, "system", environment_adapter=Adapter(text=None))
        with self.assertRaisesRegex(TypeError, "must be a string"):
            session.submit("task")
        self.assertTrue(session.failed)


if __name__ == "__main__":
    unittest.main()
