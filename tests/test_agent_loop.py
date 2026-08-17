import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

from tiny_harness.agent.loop import agent_loop
from tiny_harness.agent.messages import ModelResponse, ToolCall


class FakeProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
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


class AgentLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_returns_final_text_and_appends_assistant_message(self) -> None:
        provider = FakeProvider(
            [ModelResponse("finished", None, [], "stop")]
        )
        messages = [{"role": "user", "content": "do the task"}]

        answer = agent_loop(provider, self.workspace, messages)

        self.assertEqual(answer, "finished")
        self.assertEqual(
            messages[-1],
            {"role": "assistant", "content": "finished"},
        )
        self.assertEqual(len(provider.calls), 1)

    def test_executes_multiple_tool_calls_before_next_model_call(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    "I will create both files.",
                    [
                        ToolCall(
                            "call-1",
                            "write_file",
                            '{"path":"a.txt","content":"A"}',
                        ),
                        ToolCall(
                            "call-2",
                            "write_file",
                            '{"path":"b.txt","content":"B"}',
                        ),
                    ],
                    "tool_calls",
                ),
                ModelResponse("created", None, [], "stop"),
            ]
        )
        messages = [{"role": "user", "content": "create two files"}]

        answer = agent_loop(provider, self.workspace, messages)

        self.assertEqual(answer, "created")
        self.assertEqual((self.workspace / "a.txt").read_text(encoding="utf-8"), "A")
        self.assertEqual((self.workspace / "b.txt").read_text(encoding="utf-8"), "B")
        self.assertEqual(len(provider.calls), 2)

        second_request = provider.calls[1]["messages"]
        self.assertEqual(second_request[-3]["role"], "assistant")
        self.assertEqual(
            second_request[-3]["reasoning_content"],
            "I will create both files.",
        )
        self.assertEqual(
            [call["id"] for call in second_request[-3]["tool_calls"]],
            ["call-1", "call-2"],
        )
        self.assertEqual(
            [message["tool_call_id"] for message in second_request[-2:]],
            ["call-1", "call-2"],
        )

    def test_tool_error_is_fed_back_to_model(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("call-1", "read_file", '{"path":"missing.txt"}')],
                    "tool_calls",
                ),
                ModelResponse("handled the error", None, [], "stop"),
            ]
        )
        messages = [{"role": "user", "content": "read the file"}]

        answer = agent_loop(provider, self.workspace, messages)

        self.assertEqual(answer, "handled the error")
        tool_message = provider.calls[1]["messages"][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "call-1")
        self.assertTrue(tool_message["content"].startswith("Error: FileNotFoundError:"))

    def test_multiple_tools_receive_independent_permission_decisions(self) -> None:
        command = (
            f'"{sys.executable}" -c '
            '"from pathlib import Path; Path(\'blocked.txt\').write_text(\'bad\')"'
        )
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall(
                            "write-call",
                            "write_file",
                            '{"path":"allowed.txt","content":"ok"}',
                        ),
                        ToolCall(
                            "bash-call",
                            "bash",
                            json.dumps({"command": command}),
                        ),
                    ],
                    "tool_calls",
                ),
                ModelResponse("continued after denial", None, [], "stop"),
            ]
        )
        messages = [{"role": "user", "content": "run both"}]

        answer = agent_loop(
            provider,
            self.workspace,
            messages,
            permission_prompt=lambda *_: False,
        )

        self.assertEqual(answer, "continued after denial")
        self.assertEqual(
            (self.workspace / "allowed.txt").read_text(encoding="utf-8"),
            "ok",
        )
        self.assertFalse((self.workspace / "blocked.txt").exists())
        tool_messages = provider.calls[1]["messages"][-2:]
        self.assertEqual(
            [message["tool_call_id"] for message in tool_messages],
            ["write-call", "bash-call"],
        )
        self.assertEqual(
            tool_messages[1]["content"],
            "Error: Permission denied for tool bash",
        )

    def test_passes_all_registered_tool_schemas(self) -> None:
        provider = FakeProvider([ModelResponse("done", None, [], "stop")])

        agent_loop(provider, self.workspace, [])

        names = [
            schema["function"]["name"]
            for schema in provider.calls[0]["tools"]
        ]
        self.assertEqual(
            names,
            ["read_file", "write_file", "edit_file", "list_files", "bash"],
        )

    def test_rejects_non_positive_max_turns(self) -> None:
        provider = FakeProvider([])

        with self.assertRaisesRegex(ValueError, "at least 1"):
            agent_loop(provider, self.workspace, [], max_turns=0)

        self.assertEqual(provider.calls, [])

    def test_raises_after_maximum_model_turns(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("call-1", "write_file", '{"path":"a.txt","content":"A"}')],
                    "tool_calls",
                )
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "Maximum model turns reached: 1"):
            agent_loop(provider, self.workspace, [], max_turns=1)

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual((self.workspace / "a.txt").read_text(encoding="utf-8"), "A")

    def test_does_not_treat_length_stop_as_final_answer(self) -> None:
        provider = FakeProvider(
            [ModelResponse("partial", None, [], "length")]
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Model stopped without a final answer: length",
        ):
            agent_loop(provider, self.workspace, [])


if __name__ == "__main__":
    unittest.main()
