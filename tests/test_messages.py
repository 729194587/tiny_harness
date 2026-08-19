import unittest

from tiny_harness.agent.messages import (
    ModelResponse,
    ToolCall,
    assistant_message_from_response,
    validate_model_response,
)


class ModelResponseProtocolTest(unittest.TestCase):
    def test_accepts_consistent_stop_and_tool_call_responses(self) -> None:
        validate_model_response(ModelResponse("done", None, [], "stop"))
        validate_model_response(
            ModelResponse(
                None,
                None,
                [ToolCall("call-1", "list_files", "{}")],
                "tool_calls",
            )
        )

    def test_rejects_non_executable_responses(self) -> None:
        cases = [
            ModelResponse(
                None,
                None,
                [ToolCall("call-1", "list_files", "{}")],
                "stop",
            ),
            ModelResponse(None, None, [], "tool_calls"),
            ModelResponse("partial", None, [], "length"),
            ModelResponse(None, None, [], "content_filter"),
        ]

        for response in cases:
            with self.subTest(finish_reason=response.finish_reason):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Model response is not executable",
                ):
                    validate_model_response(response)

    def test_builds_assistant_message_with_reasoning_and_tool_calls(self) -> None:
        response = ModelResponse(
            "working",
            "private reasoning",
            [
                ToolCall("call-1", "read_file", '{"path":"a.py"}'),
                ToolCall("call-2", "list_files", "{}"),
            ],
            "tool_calls",
        )

        message = assistant_message_from_response(response)

        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "working")
        self.assertEqual(message["reasoning_content"], "private reasoning")
        self.assertEqual(
            message["tool_calls"],
            [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"a.py"}',
                    },
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "list_files",
                        "arguments": "{}",
                    },
                },
            ],
        )

    def test_omits_optional_fields_when_absent(self) -> None:
        message = assistant_message_from_response(
            ModelResponse("done", None, [], "stop")
        )

        self.assertEqual(message, {"role": "assistant", "content": "done"})


if __name__ == "__main__":
    unittest.main()
