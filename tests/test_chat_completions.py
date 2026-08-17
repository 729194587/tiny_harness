import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tiny_harness.models.chat_completions import ChatCompletionsProvider


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fake_client(response):
    completions = FakeCompletions(response)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    return client, completions


def fake_tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class ChatCompletionsProviderTest(unittest.TestCase):
    def test_constructs_configured_client(self) -> None:
        with patch("tiny_harness.models.chat_completions.OpenAI") as openai:
            ChatCompletionsProvider(
                api_key="secret",
                base_url="https://example.test",
                model="test-model",
            )

        openai.assert_called_once_with(
            api_key="secret",
            base_url="https://example.test",
        )

    def test_sends_messages_tools_and_model(self) -> None:
        api_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="done", tool_calls=None),
                )
            ]
        )
        client, completions = fake_client(api_response)
        provider = ChatCompletionsProvider(
            "secret", "https://example.test", "test-model", client=client
        )
        messages = [{"role": "user", "content": "hello"}]
        tools = [{"type": "function", "function": {"name": "read_file"}}]

        provider.complete(messages, tools)

        self.assertEqual(
            completions.calls,
            [{"model": "test-model", "messages": messages, "tools": tools}],
        )

    def test_normalizes_text_response_without_reasoning_extension(self) -> None:
        api_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="final answer", tool_calls=None),
                )
            ]
        )
        client, _ = fake_client(api_response)
        provider = ChatCompletionsProvider(
            "secret", "https://example.test", "test-model", client=client
        )

        result = provider.complete([], [])

        self.assertEqual(result.content, "final answer")
        self.assertIsNone(result.reasoning_content)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.finish_reason, "stop")

    def test_normalizes_multiple_tool_calls_and_reasoning_extension(self) -> None:
        api_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content=None,
                        reasoning_content="I should inspect both files.",
                        tool_calls=[
                            fake_tool_call("call-1", "read_file", '{"path":"a.py"}'),
                            fake_tool_call("call-2", "read_file", '{"path":"b.py"}'),
                        ],
                    ),
                )
            ]
        )
        client, _ = fake_client(api_response)
        provider = ChatCompletionsProvider(
            "secret", "https://example.test", "test-model", client=client
        )

        result = provider.complete([], [])

        self.assertIsNone(result.content)
        self.assertEqual(result.reasoning_content, "I should inspect both files.")
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(
            [(call.id, call.name, call.arguments_json) for call in result.tool_calls],
            [
                ("call-1", "read_file", '{"path":"a.py"}'),
                ("call-2", "read_file", '{"path":"b.py"}'),
            ],
        )

    def test_rejects_response_without_choices(self) -> None:
        client, _ = fake_client(SimpleNamespace(choices=[]))
        provider = ChatCompletionsProvider(
            "secret", "https://example.test", "test-model", client=client
        )

        with self.assertRaisesRegex(RuntimeError, "no choices"):
            provider.complete([], [])

    def test_rejects_response_without_finish_reason(self) -> None:
        api_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    message=SimpleNamespace(content="partial", tool_calls=None),
                )
            ]
        )
        client, _ = fake_client(api_response)
        provider = ChatCompletionsProvider(
            "secret", "https://example.test", "test-model", client=client
        )

        with self.assertRaisesRegex(RuntimeError, "no finish reason"):
            provider.complete([], [])


if __name__ == "__main__":
    unittest.main()
