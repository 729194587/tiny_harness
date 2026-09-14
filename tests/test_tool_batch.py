import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.messages import ToolCall
from tiny_harness.agent.tool_batch import execute_tool_batch


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

    def context(self, *, max_context_tokens=None):
        keyword_arguments = {"allow_subagent": False}
        if max_context_tokens is not None:
            keyword_arguments["max_context_tokens"] = max_context_tokens
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

    def test_manual_compaction_runs_after_the_complete_batch(self) -> None:
        compactor = RecordingCompactor()
        context = self.context(max_context_tokens=25_000)
        context.compactor = compactor
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
