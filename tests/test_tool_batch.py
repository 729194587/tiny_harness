import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.messages import ToolCall
from tiny_harness.agent.tool_batch import execute_tool_batch
from tiny_harness.runtime.context import CompactionRequest


class RecordingEventLogger:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event_type, data=None) -> None:
        self.events.append(
            {"event_type": event_type.value, "data": dict(data or {})}
        )


class RecordingCompactor:
    def __init__(self) -> None:
        self.calls = []

    def compact_history(self, messages, todo_state, *, reason):
        self.calls.append(
            {
                "messages": list(messages),
                "todo_state": todo_state,
                "reason": reason,
            }
        )
        return SimpleNamespace(messages=list(messages))


class ExecuteToolBatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def context(self, *, event_logger=None):
        keyword_arguments = {"allow_subagent": False}
        if event_logger is not None:
            keyword_arguments["event_logger"] = event_logger
        return create_run_context(
            object(),
            self.workspace,
            **keyword_arguments,
        )

    def test_appends_multiple_results_in_model_order(self) -> None:
        context = self.context()
        context.current_turn = 1
        messages = []

        execute_tool_batch(
            messages,
            [
                ToolCall("write-1", "write_file", '{"path":"a.txt","content":"A"}'),
                ToolCall("read-1", "read_file", '{"path":"a.txt"}'),
            ],
            context,
        )

        self.assertEqual(
            [message["tool_call_id"] for message in messages],
            ["write-1", "read-1"],
        )
        self.assertEqual(messages[-1]["content"], "A")

    def test_adds_todo_reminder_after_three_unchanged_batches(self) -> None:
        logger = RecordingEventLogger()
        context = self.context(event_logger=logger)
        messages = []

        for turn in range(1, 4):
            context.current_turn = turn
            execute_tool_batch(
                messages,
                [ToolCall(f"list-{turn}", "list_files", "{}")],
                context,
            )

        self.assertIn("<todo-reminder>", messages[-1]["content"])
        reminder = next(
            event for event in logger.events
            if event["event_type"] == "todo_reminder"
        )
        self.assertEqual(reminder["data"]["turn"], 3)
        self.assertEqual(reminder["data"]["rounds_since_todo"], 3)

    def test_manual_compaction_runs_after_the_complete_batch(self) -> None:
        request = CompactionRequest()
        compactor = RecordingCompactor()
        context = self.context()
        context.compactor = compactor
        context.compaction_request = request
        context.current_turn = 1
        messages = []

        execute_tool_batch(
            messages,
            [
                ToolCall("write-1", "write_file", '{"path":"a.txt","content":"A"}'),
                ToolCall("compact-1", "compact", "{}"),
            ],
            context,
        )

        self.assertEqual(len(compactor.calls), 1)
        self.assertEqual(compactor.calls[0]["reason"], "manual")
        self.assertEqual(
            [
                message["tool_call_id"]
                for message in compactor.calls[0]["messages"]
            ],
            ["write-1", "compact-1"],
        )


if __name__ == "__main__":
    unittest.main()
