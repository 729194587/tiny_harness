import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.loop import agent_loop
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.agent.session import AgentSession
from tiny_harness.runtime.context import ContextCompactor, ContextLimitError, prepare_context
from tiny_harness.runtime.events import EventLogError, EventType
from tiny_harness.runtime.task_state import (
    MAX_ITEMS, TASK_STATE_MARKER, TaskState, TaskStateConfig, TaskStateManager,
)


class Provider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        return next(self.responses)


def reflection(focus="verify"):
    state = asdict(TaskState(current_focus=focus, facts=["observed"], next_steps=["test"]))
    del state["goal"]
    return ModelResponse(content=json.dumps(state), reasoning_content=None, finish_reason="stop", tool_calls=[])


class TaskStateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name)

    def manager(self, **kwargs):
        return TaskStateManager(TaskStateConfig(enabled=True, **kwargs), Mock(return_value=reflection()))

    def test_independent_defaults_and_bounded_event_updates(self):
        manager = self.manager()
        manager.initialize("goal")
        manager.emit(EventType.TOOL_CALLED, {"tool_name": "read_file"})
        self.assertEqual(manager.state.completed, [])
        for i in range(30):
            manager.emit(EventType.TOOL_RESULT, {
                "tool_name": "read_file", "tool_call_id": str(i), "outcome": "returned",
                "content": "private output must not be copied",
            })
        self.assertEqual(len(manager.state.completed), MAX_ITEMS)
        self.assertNotIn("private", str(manager.state))
        for outcome in ("error", "permission_denied", "hook_blocked"):
            manager.emit(EventType.TOOL_RESULT, {"outcome": outcome})
        self.assertEqual(len(manager.state.failed_attempts), 3)
        self.assertEqual(TaskState().completed, [])
        manager.initialize("new goal")
        self.assertEqual(manager.state.completed, [])

    def test_injection_replaces_marker_without_mutating_input(self):
        manager = self.manager()
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "goal"}]
        injected = manager.inject(manager.inject(messages))
        self.assertEqual(len(injected), 3)
        self.assertEqual(injected[1]["name"], TASK_STATE_MARKER)
        self.assertEqual(len(messages), 2)

    def test_interval_and_pre_compaction_reflection(self):
        manager = self.manager(reflection_enabled=True, reflection_interval=2)
        manager.initialize("original")
        manager.prepare_turn([], 1)
        manager.prepare_turn([], 2)
        manager.complete.assert_not_called()
        manager.prepare_turn([], 3)
        manager.complete.assert_called_once()
        self.assertEqual(manager.state.goal, "original")
        self.assertEqual(manager.state.facts, ["observed"])
        manager.before_compaction([])
        self.assertEqual(manager.complete.call_count, 2)
        self.assertEqual(manager.complete.call_args.args[1], [])

    def test_reflection_disabled_by_default(self):
        manager = self.manager(reflection_interval=1)
        manager.prepare_turn([], 2)
        manager.before_compaction([])
        manager.complete.assert_not_called()

    def test_invalid_reflection_is_atomic_and_event_log_errors_are_fatal(self):
        manager = self.manager(reflection_enabled=True)
        manager.initialize("original")
        original = copy.deepcopy(manager.state)
        for content in ('not json', '[]', '{"facts": [42]}'):
            manager.complete.return_value = ModelResponse(content=content, reasoning_content=None, finish_reason="stop", tool_calls=[])
            manager.reflect([])
            self.assertEqual(manager.state, original)
        manager.complete.side_effect = RuntimeError("offline")
        manager.reflect([])
        self.assertEqual(manager.state, original)
        manager.complete.side_effect = EventLogError("failed")
        with self.assertRaises(EventLogError):
            manager.reflect([])

    def test_config_validation(self):
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                TaskStateConfig(reflection_interval=value)

    def test_loop_observes_denied_tool_and_injects_state(self):
        provider = Provider([
            ModelResponse(content=None, reasoning_content=None, finish_reason="tool_calls", tool_calls=[ToolCall(id="bad", name="unknown", arguments_json="{}")]),
            reflection(),
            ModelResponse(content="done", reasoning_content=None, finish_reason="stop", tool_calls=[]),
        ])
        context = create_run_context(provider, self.workspace, max_turns=3,
            task_state_config=TaskStateConfig(enabled=True, reflection_enabled=True, reflection_interval=1))
        messages = [{"role": "user", "content": "goal"}]
        self.assertEqual(agent_loop(messages, context, "goal"), "done")
        # Reflection input sees deterministic dispatch evidence before replacing state.
        self.assertIn("unknown", str(provider.calls[1][0]))
        self.assertIn("failed_attempts", str(provider.calls[1][0]))
        self.assertEqual(provider.calls[1][1], [])
        self.assertIn('"current_focus": "verify"', str(provider.calls[2][0]))

    def test_disabled_has_no_marker_or_extra_model_calls(self):
        provider = Provider([ModelResponse(content="done", reasoning_content=None, finish_reason="stop", tool_calls=[])])
        context = create_run_context(provider, self.workspace)
        self.assertIsNone(context.task_state_manager)
        agent_loop([{"role": "user", "content": "goal"}], context, "goal")
        self.assertEqual(len(provider.calls), 1)
        self.assertNotIn(TASK_STATE_MARKER, str(provider.calls))

    def test_session_creates_fresh_state_and_removes_markers(self):
        provider = Provider([ModelResponse(content="done", reasoning_content=None, finish_reason="stop", tool_calls=[]) for _ in range(2)])
        session = AgentSession(provider, self.workspace, "system", task_state_config=TaskStateConfig(enabled=True))
        session.submit("first")
        self.assertNotIn(TASK_STATE_MARKER, str(session.messages))
        session.submit("second")
        marker = next(m for m in provider.calls[1][0] if m.get("name") == TASK_STATE_MARKER)
        self.assertIn('"goal": "second"', marker["content"])
        self.assertNotIn('"goal": "first"', marker["content"])
        self.assertFalse((self.workspace / ".tinyharness" / "memory").exists())

    def test_compaction_callbacks_run_before_loss_and_preserve_state(self):
        history = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old " * 3000},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "goal"},
        ]
        for mode in ("automatic", "manual", "reactive"):
            with self.subTest(mode=mode):
                provider = Provider([ModelResponse(content="summary", reasoning_content=None, finish_reason="stop", tool_calls=[])])
                compactor = ContextCompactor(self.workspace, provider, [], 2000)
                manager = self.manager(reflection_enabled=True)
                manager.initialize("goal")
                compactor.before_compaction = manager.before_compaction
                messages = manager.inject(history)
                original = copy.deepcopy(messages)
                if mode == "automatic":
                    prepared = prepare_context(messages, compactor, "", "goal")
                elif mode == "manual":
                    prepared = compactor.compact_history(messages, "", reason="manual")
                else:
                    prepared = compactor.reactive_compact(messages, "", failed_request_tokens=5000)
                manager.complete.assert_called_once()
                self.assertIn("old old old", str(manager.complete.call_args))
                self.assertEqual(messages, original)
                marker = next(m for m in prepared.messages if m.get("name") == TASK_STATE_MARKER)
                self.assertIn('"current_focus": "verify"', marker["content"])

    def test_state_counts_toward_context_budget(self):
        provider = Provider([])
        compactor = ContextCompactor(self.workspace, provider, [], 100)
        manager = self.manager()
        manager.initialize("goal")
        manager.state.facts = ["fact " * 100 for _ in range(20)]
        with self.assertRaises(ContextLimitError):
            prepare_context(manager.inject([{"role": "user", "content": "goal"}]), compactor, "", "goal")

    def test_child_has_independent_state_and_does_not_feed_parent_observer(self):
        provider = Provider([
            ModelResponse(content=None, reasoning_content=None, finish_reason="tool_calls",
                          tool_calls=[ToolCall("child-call", "unknown", "{}")]),
            ModelResponse(content="child done", reasoning_content=None, finish_reason="stop", tool_calls=[]),
        ])
        context = create_run_context(provider, self.workspace,
            task_state_config=TaskStateConfig(enabled=True))
        context.task_state_manager.initialize("parent goal")
        self.assertEqual(context.subagent_runner("child goal", "parent-call"), "child done")
        self.assertEqual(context.task_state_manager.state.failed_attempts, [])
        self.assertEqual(context.task_state_manager.state.goal, "parent goal")
        marker = next(m for m in provider.calls[1][0] if m.get("name") == TASK_STATE_MARKER)
        self.assertIn('"goal": "child goal"', marker["content"])
        self.assertIn("child-call", marker["content"])


if __name__ == "__main__":
    unittest.main()
