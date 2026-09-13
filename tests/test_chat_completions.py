import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.models.chat_completions import ChatCompletionsProvider


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
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
            max_retries=0,
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

    def test_disables_thinking_for_deepseek_only(self) -> None:
        api_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="done", tool_calls=None),
                )
            ]
        )
        deepseek_client, deepseek_completions = fake_client(api_response)
        other_client, other_completions = fake_client(api_response)
        messages = [{"role": "user", "content": "hello"}]

        ChatCompletionsProvider(
            "secret",
            "https://api.deepseek.com/v1",
            "deepseek-v4-flash",
            client=deepseek_client,
        ).complete(messages, [])
        ChatCompletionsProvider(
            "secret",
            "https://example.test/v1",
            "test-model",
            client=other_client,
        ).complete(messages, [])

        self.assertEqual(
            deepseek_completions.calls[0]["extra_body"],
            {"thinking": {"type": "disabled"}},
        )
        self.assertNotIn("extra_body", other_completions.calls[0])

    def test_serializes_explicit_tool_choice(self) -> None:
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
        tools = [{"type": "function", "function": {"name": "read_file"}}]

        provider.complete([], tools, tool_choice="auto")
        provider.complete([], [], tool_choice="none")

        self.assertEqual(completions.calls[0]["tool_choice"], "auto")
        self.assertEqual(completions.calls[0]["tools"], tools)
        self.assertEqual(completions.calls[1]["tool_choice"], "none")
        self.assertEqual(completions.calls[1]["tools"], [])

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
        self.assertNotIn("tools", client.chat.completions.calls[0])
        self.assertNotIn("tool_choice", client.chat.completions.calls[0])

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

    def test_normalizes_rate_limit_and_retry_after(self) -> None:
        error = RuntimeError("rate limited")
        error.status_code = 429
        error.response = SimpleNamespace(headers={"retry-after": "2.5"})
        client, _ = fake_client(error)
        provider = ChatCompletionsProvider(
            "secret", "https://example.test", "test-model", client=client
        )

        with self.assertRaises(ModelProviderError) as raised:
            provider.complete([], [])

        self.assertEqual(raised.exception.kind, ModelErrorKind.RATE_LIMIT)
        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.retry_after_seconds, 2.5)

    def test_context_error_recognition_is_a_restricted_heuristic(self) -> None:
        context_error = RuntimeError("maximum context length exceeded")
        context_error.status_code = 400
        client, _ = fake_client(context_error)
        provider = ChatCompletionsProvider(
            "secret", "https://example.test", "test-model", client=client
        )

        with self.assertRaises(ModelProviderError) as raised:
            provider.complete([], [])

        self.assertEqual(raised.exception.kind, ModelErrorKind.CONTEXT_LENGTH)

        fatal_error = RuntimeError("maximum context length in an auth response")
        fatal_error.status_code = 401
        fatal_client, _ = fake_client(fatal_error)
        fatal_provider = ChatCompletionsProvider(
            "secret", "https://example.test", "test-model", client=fatal_client
        )
        with self.assertRaises(ModelProviderError) as fatal:
            fatal_provider.complete([], [])
        self.assertEqual(fatal.exception.kind, ModelErrorKind.FATAL)

    def test_normalizes_server_connection_and_unknown_failures(self) -> None:
        cases = [
            (SimpleNamespace(status_code=503), ModelErrorKind.SERVER_UNAVAILABLE),
            (TimeoutError("timed out"), ModelErrorKind.CONNECTION),
            (SimpleNamespace(status_code=422), ModelErrorKind.FATAL),
        ]
        for raw, expected in cases:
            with self.subTest(expected=expected):
                if not isinstance(raw, Exception):
                    error = RuntimeError("request failed")
                    error.status_code = raw.status_code
                else:
                    error = raw
                client, _ = fake_client(error)
                provider = ChatCompletionsProvider(
                    "secret",
                    "https://example.test",
                    "test-model",
                    client=client,
                )
                with self.assertRaises(ModelProviderError) as raised:
                    provider.complete([], [])
                self.assertEqual(raised.exception.kind, expected)

    def test_insufficient_system_resource_discards_choice_payload(self) -> None:
        api_response = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="insufficient_system_resource")]
        )
        client, _ = fake_client(api_response)
        provider = ChatCompletionsProvider(
            "secret", "https://example.test", "test-model", client=client
        )

        with self.assertRaises(ModelProviderError) as raised:
            provider.complete([], [])

        self.assertEqual(raised.exception.kind, ModelErrorKind.SERVER_UNAVAILABLE)

    def test_length_and_content_filter_discard_choice_payload(self) -> None:
        for finish_reason in ("length", "content_filter"):
            with self.subTest(finish_reason=finish_reason):
                api_response = SimpleNamespace(
                    choices=[SimpleNamespace(finish_reason=finish_reason)]
                )
                client, _ = fake_client(api_response)
                provider = ChatCompletionsProvider(
                    "secret",
                    "https://example.test",
                    "test-model",
                    client=client,
                )

                with self.assertRaises(ModelProviderError) as raised:
                    provider.complete([], [])

                self.assertEqual(raised.exception.kind, ModelErrorKind.FATAL)
                self.assertIn(finish_reason, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
