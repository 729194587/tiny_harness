import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from tiny_harness.agent.context import create_run_context, initialize_run_state
from tiny_harness.agent.loop import agent_loop
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.agent.session import AgentSession
from tiny_harness.agent.turn import model_request_inputs, call_model
from tiny_harness.runtime.context import prepare_context, ContextLimitError
from tiny_harness.runtime.events import EventType, EventLogError, hash_text
from tiny_harness.runtime.hooks import ToolHooks, HookBlock
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.runtime.working_memory import MAX_NOTE_CHARS, WORKING_MEMORY_MARKER, WorkingMemory
from tiny_harness.tools.registry import dispatch


def answer(text="done"):
    return ModelResponse(text, None, [], "stop")


def update(note, call_id="update"):
    return ToolCall(call_id, "update_working_memory", json.dumps({"note": note}))


def batch(*calls):
    return ModelResponse(None, None, list(calls), "tool_calls")


class Provider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def state(messages):
    projections = [m for m in messages if m.get("name") == WORKING_MEMORY_MARKER]
    assert len(projections) == 1
    return json.loads(projections[0]["content"].split("\n", 1)[1])


class WorkingMemoryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)

    def context(self, provider=None, **options):
        return create_run_context(provider or Provider([]), self.workspace,
                                  working_memory_enabled=True, **options)

    def test_objective_replacement_bound_and_clear(self):
        memory = WorkingMemory()
        memory.initialize("完整目标" * 1000)
        self.assertEqual(memory.objective, "完整目标" * 1000)
        memory.update("old")
        memory.update("新😀" * MAX_NOTE_CHARS)
        self.assertEqual(memory.note, ("新😀" * MAX_NOTE_CHARS)[:MAX_NOTE_CHARS])
        with self.assertRaises(ValueError):
            memory.update(None)
        self.assertEqual(len(memory.note), MAX_NOTE_CHARS)
        memory.update("")
        self.assertEqual(memory.note, "")

    def test_projection_is_passive_and_independent_of_objective_length(self):
        memory = WorkingMemory()
        memory.initialize("short")
        memory.update("bounded note")
        projection = memory.projection()
        memory.initialize("LONG OBJECTIVE " * 10000)
        memory.update("bounded note")
        self.assertEqual(memory.projection(), projection)
        self.assertEqual(state([projection]), {"note": "bounded note"})
        self.assertNotIn("update_working_memory", projection["content"])
        self.assertNotIn("objective", projection["content"])

    def test_disabled_omits_tool_projection_and_calls(self):
        provider = Provider([answer()])
        context = create_run_context(provider, self.workspace)
        self.assertIsNone(context.working_memory)
        self.assertNotIn("update_working_memory", [t.name for t in context.tool_registry.list()])
        messages = [{"role": "user", "content": "goal"}]
        agent_loop(messages, context, "goal")
        self.assertEqual(len(provider.calls), 1)
        self.assertFalse(any(m.get("name") == WORKING_MEMORY_MARKER for m in provider.calls[0][0]))

    def test_projection_leaves_history_and_system_prefix_unchanged(self):
        context = self.context()
        context.working_memory.initialize("goal")
        messages = [{"role": "system", "content": "stable system"},
                    {"role": "user", "content": "goal"}]
        original = copy.deepcopy(messages)
        for note in ("first", "latest", ""):
            context.working_memory.update(note)
            request, _ = model_request_inputs(messages, context, finalization=False)
            self.assertEqual(state(request), {"note": note})
            self.assertEqual(request[0], original[0])
            self.assertEqual(request[-1]["role"], "user")
            self.assertEqual(messages, original)
        request, tools = model_request_inputs(messages, context, finalization=True)
        self.assertEqual(state(request), {"note": ""})
        self.assertEqual(tools, [])

    def test_explicit_updates_survive_turns_without_extra_provider_calls(self):
        provider = Provider([
            batch(update("old", "one"), update("latest", "two")),
            batch(ToolCall("read", "list_files", "{}")),
            answer(),
        ])
        context = self.context(provider, max_turns=3)
        messages = [{"role": "user", "content": "goal"}]
        self.assertEqual(agent_loop(messages, context, "goal"), "done")
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual([state(m)["note"] for m, _ in provider.calls], ["", "latest", "latest"])
        self.assertFalse(any(m.get("name") == WORKING_MEMORY_MARKER for m in messages))
        self.assertEqual(context.working_memory.note, "latest")

    def test_tool_schema_validation_safe_events_and_default_permission(self):
        context = self.context()
        logger = Mock()
        context.event_logger = logger
        definition = context.tool_registry.lookup("update_working_memory")
        self.assertEqual(definition.parameters["properties"]["note"]["maxLength"], MAX_NOTE_CHARS)
        self.assertIn("changes materially", definition.description)
        self.assertIn("needed later", definition.description)
        self.assertIn("Do not merely repeat recent tool", definition.description)
        self.assertIn("Todo", definition.description)
        schema = definition.parameters
        schema["properties"].clear()
        self.assertIn("note", definition.parameters["properties"])
        note = "private content " * 200
        result = dispatch(context.tool_registry, update(note), event_logger=logger)
        self.assertNotIn("Error:", result.content)
        self.assertEqual(context.working_memory.note, note[:MAX_NOTE_CHARS])
        events = [(c.args[0], c.args[1]) for c in logger.emit.call_args_list]
        changed = next(data for kind, data in events if kind == EventType.WORKING_MEMORY_UPDATED)
        self.assertEqual(changed["note_length"], MAX_NOTE_CHARS)
        self.assertEqual(changed["note_hash"], hash_text(note[:MAX_NOTE_CHARS]))
        self.assertTrue(changed["truncated"])
        self.assertNotIn("private content", str(events))
        self.assertIn(EventType.TOOL_STARTED, [kind for kind, _ in events])
        for arguments in ({}, {"note": 5}, {"note": "x", "extra": 1}):
            result = dispatch(context.tool_registry, ToolCall("bad", "update_working_memory", json.dumps(arguments)))
            self.assertIn("Error:", result.content)
            self.assertEqual(context.working_memory.note, note[:MAX_NOTE_CHARS])

    def test_permission_and_pre_hook_prevent_mutation(self):
        context = self.context()
        context.working_memory.update("original")
        deny = Mock()
        deny.decide.return_value = PermissionDecision.DENY
        result = dispatch(context.tool_registry, update("denied"), permission_policy=deny)
        self.assertIn("Error:", result.content)
        hooks = ToolHooks()
        hooks.register_pre(lambda _: HookBlock("blocked"))
        result = dispatch(context.tool_registry, update("blocked"), tool_hooks=hooks)
        self.assertIn("Error:", result.content)
        self.assertEqual(context.working_memory.note, "original")

    def test_update_event_log_failure_remains_fatal(self):
        context = self.context()
        context.event_logger = Mock()
        context.event_logger.emit.side_effect = EventLogError("failed")
        with self.assertRaises(EventLogError):
            dispatch(context.tool_registry, update("new"))
        # State mutations, like other tool side effects, are not rolled back.
        self.assertEqual(context.working_memory.note, "new")

    def test_session_starts_each_submission_with_fresh_memory(self):
        provider = Provider([batch(update("first note")), answer(), answer()])
        session = AgentSession(provider, self.workspace, "system", working_memory_enabled=True)
        session.submit("first objective")
        session.submit("second objective")
        self.assertEqual(state(provider.calls[-1][0]), {"note": ""})
        self.assertFalse(any(m.get("name") == WORKING_MEMORY_MARKER for m in session.messages))
        self.assertFalse((self.workspace / ".tinyharness" / "memory").exists())

    def test_subagent_has_own_objective_and_note(self):
        provider = Provider([batch(update("child note")), answer(), answer()])
        parent = self.context(provider)
        parent.working_memory.initialize("parent objective")
        parent.working_memory.update("parent note")
        parent.subagent_runner("child objective", "parent-call")
        self.assertEqual(state(provider.calls[0][0]), {"note": ""})
        self.assertEqual(state(provider.calls[1][0])["note"], "child note")
        parent.subagent_runner("second child", "second-call")
        self.assertEqual(state(provider.calls[2][0]), {"note": ""})
        self.assertEqual(parent.working_memory.note, "parent note")
        self.assertEqual(parent.working_memory.objective, "parent objective")

    def test_compaction_preserves_memory_without_reflection_calls(self):
        for mode in ("automatic", "manual", "reactive"):
            with self.subTest(mode=mode):
                provider = Provider([answer("summary"), answer()])
                context = self.context(provider, max_context_tokens=12000)
                messages = [{"role": "system", "content": "system"},
                            {"role": "user", "content": "old " * 18000},
                            {"role": "assistant", "content": "old answer"},
                            {"role": "user", "content": "goal"}]
                initialize_run_state(messages, context, "goal")
                context.working_memory.update("retained state")
                original = copy.deepcopy(messages)
                if mode == "automatic":
                    prepared = prepare_context(messages, context.compactor, "", "goal")
                elif mode == "manual":
                    prepared = context.compactor.compact_history(messages, "", reason="manual")
                else:
                    prepared = context.compactor.reactive_compact(messages, "", failed_request_tokens=24000)
                self.assertEqual(messages, original)
                self.assertEqual(len(provider.calls), 1)  # Only the existing summary call.
                messages[:] = prepared.messages
                self.assertFalse(any(m.get("name") == WORKING_MEMORY_MARKER for m in messages))
                call_model(messages, context)
                self.assertEqual(len(provider.calls), 2)
                self.assertEqual(state(provider.calls[-1][0]), {"note": "retained state"})

    def test_projection_budget_is_reserved_without_history_injection(self):
        context = self.context(max_context_tokens=5000)
        messages = [{"role": "user", "content": "goal"}]
        initialize_run_state(messages, context, "goal")
        baseline = context.compactor.token_meter.estimate(messages, context.tools)
        context.working_memory.update("x" * MAX_NOTE_CHARS)
        self.assertGreater(context.compactor.token_meter.estimate(messages, context.tools), baseline)
        request, tools = model_request_inputs(messages, context, finalization=False)
        self.assertGreater(context.token_meter.estimate_request(messages, request, tools), baseline)
        context.compactor.max_tokens = 100
        with self.assertRaises(ContextLimitError):
            prepare_context(messages, context.compactor, "", "goal")
        self.assertEqual(context.working_memory.note, "x" * MAX_NOTE_CHARS)


if __name__ == "__main__":
    unittest.main()
