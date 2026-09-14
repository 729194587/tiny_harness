"""Todo updates remain append-only between compaction epochs (offline)."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.loop import agent_loop
from tiny_harness.agent.messages import ModelResponse, ToolCall, assistant_message_from_response
from tiny_harness.agent.tool_batch import execute_tool_batch
from tiny_harness.agent.turn import prepare_model_request_inputs
from tiny_harness.runtime.context import prepare_context, validate_active_request
from tiny_harness.runtime.events import EventType


def todo_response(index, status):
    todos = [] if status is None else [{"content": "Implement change", "status": status}]
    return ModelResponse(None, None, [ToolCall(str(index), "todo_write", json.dumps({"todos": todos}))], "tool_calls")


class TodoCacheTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.provider = Mock()
        self.provider.complete.return_value = ModelResponse("Task checkpoint", None, [], "stop")
        self.logger = Mock()
        self.context = create_run_context(self.provider, Path(temporary.name),
                                          allow_subagent=False, event_logger=self.logger,
                                          max_context_tokens=125_000)

    def request(self, messages):
        prepared = prepare_context(messages, self.context.compactor,
                                   self.context.todo_manager.render(), "task")
        messages[:] = prepared.messages
        return copy.deepcopy(prepare_model_request_inputs(messages, self.context, finalization=False))

    def update(self, messages, index, status):
        response = todo_response(index, status)
        messages.append(assistant_message_from_response(response))
        execute_tool_batch(messages, response.tool_calls, self.context)

    def test_loop_multiple_updates_and_clear_preserve_request_prefix(self):
        requests = []
        responses = iter([
            todo_response(1, "pending"), todo_response(2, "in_progress"),
            todo_response(3, "completed"),
            *[ModelResponse(None, None, [ToolCall(str(i), "list_files", "{}")], "tool_calls")
              for i in range(4, 7)],
            todo_response(7, None), ModelResponse("done", None, [], "stop"),
        ])

        def complete(messages, tools, **kwargs):
            requests.append(copy.deepcopy((messages, tools)))
            return next(responses)

        self.provider.complete.side_effect = complete
        messages = [{"role": "system", "content": "stable"}, {"role": "user", "content": "task"}]
        self.assertEqual(agent_loop(messages, self.context, "task"), "done")
        for previous, current in zip(requests, requests[1:]):
            self.assertEqual(current[0][:len(previous[0])], previous[0])
            self.assertEqual(current[1], previous[1])
        for index, state in ((1, "[ ] Implement change"), (2, "[>] Implement change"),
                             (3, "[x] Implement change"), (7, "No todos.")):
            self.assertIn(state, requests[index][0][-1]["content"])
        for index in range(4, 7):
            self.assertEqual(requests[index][0][-1]["content"], requests[4][0][-1]["content"])
        self.assertFalse(any(m.get("name") == "tinyharness_todo_state" for m in messages))
        self.assertEqual(self.context.todo_manager.render(), "No todos.")
        self.assertEqual(sum(c.args[0] == EventType.TODO_UPDATED for c in self.logger.emit.call_args_list), 4)

    def test_working_checkpoint_snapshots_todos_then_updates_remain_append_only(self):
        messages = [{"role": "user", "content": "task"}]
        self.update(messages, "before", "in_progress")
        # Balanced older evidence crosses the real working threshold.
        for index in range(8):
            call = ToolCall(f"evidence-{index}", "bash", "{}")
            messages.append(assistant_message_from_response(ModelResponse(None, "diagnosis", [call], "tool_calls")))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": "evidence " * 1600})
        first, tools = self.request(messages)
        self.provider.complete.assert_called_once()
        snapshot = next(m for m in first if m.get("name") == "tinyharness_todo_state")
        self.assertIn("[>] Implement change", snapshot["content"])
        self.assertTrue(any(m.get("name") == "tinyharness_context_summary" for m in first))
        for index, status in (("after", "completed"), ("clear", None)):
            self.update(messages, index, status)
            current, current_tools = self.request(messages)
            self.assertEqual(current[:len(first)], first)
            self.assertEqual(current_tools, tools)
            self.assertEqual(current[-1]["content"], self.context.todo_manager.render())
            self.assertIn(snapshot, current)
            validate_active_request(messages, "task")
            first = current
        self.provider.complete.assert_called_once()
