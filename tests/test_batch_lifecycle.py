import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.loop import agent_loop
from tiny_harness.agent.messages import ModelProtocolError, ModelResponse, ToolCall
from tiny_harness.agent.session import AgentSession, SessionFailedError
from tiny_harness.agent.tool_batch import execute_tool_batch
from tiny_harness.runtime.events import EventLogError, EventType
from tiny_harness.runtime.hooks import HookExecutionError, ToolHooks
from tiny_harness.runtime.skills import discover_skills


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        return next(self.responses)


class BatchLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = Path(self.directory.name)

    def write_call(self, call_id, path):
        return ToolCall(call_id, "write_file", json.dumps({"path": path, "content": "REAL"}))

    def test_invalid_batch_fails_before_any_handler_or_history_commit(self):
        for invalid_id in ("first", "", "   ", None, 17):
            with self.subTest(invalid_id=invalid_id):
                calls = [self.write_call("first", "first.txt"), self.write_call(invalid_id, "second.txt")]
                provider = ScriptedProvider([ModelResponse(None, None, calls, "tool_calls")])
                context = create_run_context(
                    provider, self.workspace, allow_subagent=False,
                    skill_catalog=discover_skills(self.workspace, sources=()),
                )
                messages = [{"role": "user", "content": "task"}]
                with patch("tiny_harness.tools.filesystem.write_file") as handler:
                    with self.assertRaises(ModelProtocolError):
                        agent_loop(messages, context, "task")
                    handler.assert_not_called()
                self.assertEqual(messages, [{"role": "user", "content": "task"}])
                self.assertEqual(provider.calls, 1)

    def test_direct_batch_executor_validates_before_dispatch(self):
        context = create_run_context(ScriptedProvider([]), self.workspace)
        calls = [self.write_call("same", "a.txt"), self.write_call("same", "b.txt")]
        with patch("tiny_harness.agent.tool_batch.dispatch") as dispatch:
            with self.assertRaises(ModelProtocolError):
                execute_tool_batch([], calls, context)
            dispatch.assert_not_called()

    def test_fatal_observer_preserves_history_and_requires_clear(self):
        for failure in ("post_hook", "event_log", "interrupt"):
            with self.subTest(failure=failure):
                calls = [self.write_call(str(i), f"{failure}-{i}.txt") for i in range(3)]
                provider = ScriptedProvider([
                    ModelResponse(None, None, calls, "tool_calls"),
                    ModelResponse("fresh answer", None, [], "stop"),
                ])
                hooks = ToolHooks()

                def post(context, result):
                    if context.tool_call_id == "1":
                        if failure == "post_hook":
                            raise ValueError("observer failed")
                        if failure == "interrupt":
                            raise KeyboardInterrupt()

                hooks.register_post(post)

                class Logger:
                    def emit(self, event_type, data=None):
                        if failure == "event_log" and event_type == EventType.TOOL_RESULT and data["tool_call_id"] == "1":
                            raise EventLogError("observer failed")

                def compose(*args, **kwargs):
                    return create_run_context(*args, **kwargs, tool_hooks=hooks)

                session = AgentSession(provider, self.workspace, "system", event_logger_factory=Logger)
                expected = {"post_hook": HookExecutionError, "event_log": EventLogError, "interrupt": KeyboardInterrupt}[failure]
                with patch("tiny_harness.agent.session.create_run_context", side_effect=compose):
                    with self.assertRaises(expected):
                        session.submit("write files")
                self.assertTrue(session.failed)
                self.assertEqual((self.workspace / f"{failure}-0.txt").read_text(), "REAL")
                self.assertEqual((self.workspace / f"{failure}-1.txt").read_text(), "REAL")
                self.assertFalse((self.workspace / f"{failure}-2.txt").exists())
                self.assertEqual([m["tool_call_id"] for m in session.messages if m["role"] == "tool"], ["0"])
                assistant = next(m for m in session.messages if m.get("tool_calls"))
                self.assertEqual([c["id"] for c in assistant["tool_calls"]], ["0", "1", "2"])
                retained = copy.deepcopy(session.messages)
                with self.assertRaisesRegex(SessionFailedError, r"clear\(\)"):
                    session.submit("try again")
                self.assertEqual(session.messages, retained)
                self.assertEqual(provider.calls, 1)
                session.clear()
                self.assertFalse(session.failed)
                self.assertEqual(session.messages, [{"role": "system", "content": "system"}])
                self.assertEqual((self.workspace / f"{failure}-1.txt").read_text(), "REAL")
                self.assertEqual(session.submit("fresh task"), "fresh answer")
                self.assertEqual(provider.calls, 2)

    def test_tool_argument_errors_do_not_poison_session(self):
        provider = ScriptedProvider([
            ModelResponse(None, None, [ToolCall("bad", "write_file", "{")], "tool_calls"),
            ModelResponse("done", None, [], "stop"),
            ModelResponse("next", None, [], "stop"),
        ])
        session = AgentSession(provider, self.workspace, "system")
        self.assertEqual(session.submit("task"), "done")
        self.assertFalse(session.failed)
        result = next(m for m in session.messages if m["role"] == "tool")
        self.assertIn("JSONDecodeError", result["content"])
        self.assertEqual(session.submit("next task"), "next")

    def test_invalid_input_does_not_poison_session(self):
        provider = ScriptedProvider([ModelResponse("done", None, [], "stop")])
        session = AgentSession(provider, self.workspace, "system")
        with self.assertRaises(ValueError):
            session.submit("   ")
        self.assertFalse(session.failed)
        self.assertEqual(provider.calls, 0)
        self.assertEqual(session.submit("task"), "done")


if __name__ == "__main__":
    unittest.main()
