import copy
import tempfile
import unittest
from pathlib import Path

from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.agent.session import AgentSession
from tiny_harness.agent.turn import model_request_inputs
from tiny_harness.runtime.events import EventType
from tiny_harness.runtime.progress import ExecutionState, ProgressTracker
from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.runtime.recovery import RecoveryPolicy
from tiny_harness.tools.registry import dispatch


class Provider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append(copy.deepcopy(messages))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def final():
    return ModelResponse("done", None, [], "stop")


def observation(messages):
    return [m["content"] for m in messages
            if str(m.get("content", "")).startswith("Execution state")]


class ProgressTests(unittest.TestCase):
    def test_outcomes_and_started_semantics(self):
        tracker = ProgressTracker()
        for outcome in ("returned", "error", "permission_denied", "hook_blocked"):
            tracker.emit(EventType.TOOL_CALLED)
            tracker.emit(EventType.TOOL_RESULT, {"outcome": outcome})
        tracker.emit(EventType.TOOL_DENIED)
        for name in ("bash", "git_status", "git_diff", "run_tests", "read_file"):
            tracker.emit(EventType.TOOL_STARTED, {"tool_name": name})
        for _ in range(3):
            tracker.emit(EventType.MODEL_REQUESTED, {"turn": 1})
            tracker.render(turn=1, max_turns=20)
        self.assertEqual(tracker.state, ExecutionState(4, 1, 1, 2, 3, 1))

    def test_dispatch_and_request_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            context = create_run_context(Provider([]), Path(directory), progress_enabled=True)
            dispatch(context.tool_registry, ToolCall("x", "bash", '{"command":"custom-command"}'),
                     permission_prompt=lambda *args: False, event_logger=context.event_logger)
            dispatch(context.tool_registry, ToolCall("y", "bash", 'invalid'),
                     event_logger=context.event_logger)
            state = context.progress_tracker.state
            self.assertEqual((state.tool_calls, state.tool_denied, state.tool_failed), (2, 1, 1))
            self.assertEqual(state.commands_started, 0)
            messages = [{"role": "user", "content": "task"}]
            original = copy.deepcopy(messages)
            context.current_turn = 8
            for _ in range(3):
                request, _ = model_request_inputs(messages, context, finalization=False)
                self.assertIn("Turn: 8/20", observation(request)[0])
                self.assertEqual(len(observation(request)), 1)
            self.assertEqual(messages, original)
            self.assertEqual(state.tool_calls, 2)
            disabled = create_run_context(Provider([]), Path(directory))
            self.assertIsNone(disabled.progress_tracker)
            self.assertFalse(observation(model_request_inputs(messages, disabled, finalization=False)[0]))

    def test_submissions_and_sessions_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = Provider([ModelResponse(None, None, [ToolCall("x", "unknown", "{}")], "tool_calls"), final(), final(), final()])
            session = AgentSession(provider, Path(directory), "system", progress_enabled=True)
            session.submit("first")
            self.assertIn("Tool calls: 1", observation(provider.requests[1])[0])
            session.submit("second")
            self.assertIn("Tool calls: 0", observation(provider.requests[2])[0])
            self.assertFalse(observation(session.messages))
            AgentSession(provider, Path(directory), "system", progress_enabled=True).submit("third")
            self.assertIn("Tool calls: 0", observation(provider.requests[3])[0])

    def test_child_events_do_not_reach_parent_tracker(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = Provider([ModelResponse(None, None, [ToolCall("x", "unknown", "{}")], "tool_calls"), final()])
            context = create_run_context(provider, Path(directory), progress_enabled=True)
            context.subagent_runner("child task", "parent-call")
            self.assertEqual(context.progress_tracker.state, ExecutionState())
            self.assertIn("Tool calls: 1", observation(provider.requests[1])[0])

    def test_retry_preserves_logical_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = Provider([ModelProviderError(ModelErrorKind.RATE_LIMIT), final()])
            session = AgentSession(
                provider, Path(directory), "system", progress_enabled=True,
                recovery_policy=RecoveryPolicy(base_delay_seconds=0, jitter_ratio=0),
            )
            session.submit("task")
            self.assertEqual(observation(provider.requests[0]), observation(provider.requests[1]))
            self.assertIn("Turn: 1/20", observation(provider.requests[1])[0])
            self.assertFalse(observation(session.messages))


if __name__ == "__main__":
    unittest.main()
