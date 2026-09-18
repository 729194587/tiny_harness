import tempfile
import unittest
from pathlib import Path

from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.messages import ToolCall
from tiny_harness.agent.tool_batch import execute_tool_batch


class ExecuteToolBatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def context(self):
        return create_run_context(
            object(),
            self.workspace,
            allow_subagent=False,
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
        self.assertEqual(messages[-1]["content"], "[lines 1-1 of 1 | a.txt]\n\nA")


if __name__ == "__main__":
    unittest.main()
