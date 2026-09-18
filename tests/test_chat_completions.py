import unittest
import tempfile
from pathlib import Path
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
    def test_summary_and_main_serialize_the_same_tool_options(self):
        from tiny_harness.agent.context import _completion_router
        from tiny_harness.runtime.context import ContextCompactor
        from tiny_harness.runtime.recovery import RecoveryExecutor

        client, completions = fake_client(SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="summary", tool_calls=None),
        )]))
        provider = ChatCompletionsProvider("secret", "https://example.test", "deepseek", client=client)
        tools = [{"type": "function", "function": {
            "name": name, "parameters": {"type": "object"},
        }} for name in ("read_file", "glob")]
        router = _completion_router(provider, RecoveryExecutor(), lambda: 1)
        router("main", [], tools)
        router("summary", [], tools)
        with tempfile.TemporaryDirectory() as directory:
            compactor = ContextCompactor(Path(directory), provider, tools, 10_000)
            compactor._summary_complete([], compactor._summary_tools())
        for request in completions.calls:
            self.assertEqual(request["tools"], tools)
            self.assertNotIn("tool_choice", request)

    def test_dsml_finalization_is_bounded_and_session_stays_usable(self) -> None:
        from tiny_harness.agent.session import AgentSession
        from tiny_harness.agent.turn import FINALIZATION_FAILURE, FINALIZATION_INSTRUCTION

        def completion(content):
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content, tool_calls=None),
            )])

        markup = '<｜DSML｜function_calls><｜DSML｜invoke name="read_file">'
        for recover in (True, False):
            with self.subTest(recover=recover), tempfile.TemporaryDirectory() as directory:
                client, completions = fake_client(None)
                provider = ChatCompletionsProvider(
                    "secret", "https://example.test", "deepseek", client=client,
                )
                session = AgentSession(provider, Path(directory), "system", max_turns=1)
                with patch.object(completions, "create", side_effect=[
                    completion(markup),
                    completion("verified answer" if recover else markup),
                    completion("next answer"),
                ]) as create:
                    self.assertEqual(session.submit("inspect"),
                                     "verified answer" if recover else FINALIZATION_FAILURE)
                    self.assertEqual(create.call_count, 2)
                    self.assertEqual(session.submit("next request"), "next answer")
                    for call in create.call_args_list:
                        self.assertEqual(call.kwargs["tools"], [])
                        self.assertEqual(call.kwargs["tool_choice"], "none")
                        self.assertNotIn(markup, str(call.kwargs))
                        self.assertEqual(sum(m.get("content") == FINALIZATION_INSTRUCTION
                                             for m in call.kwargs["messages"]), 1)

    def test_recognizes_only_explicit_dsml_tool_protocol(self) -> None:
        for content, expected in (
            ('<｜DSML｜function_calls><｜DSML｜invoke name="read_file">', True),
            ('<|DSML|invoke name="read_file">', True),
            ('</｜DSML｜function_calls>', True),
            ('DSML is a serialization format. Call read_file to inspect a file.', False),
            ('<parameter>ordinary XML</parameter>', False),
            ('A valid final answer.', False),
            (None, False),
        ):
            with self.subTest(content=content):
                client, _ = fake_client(SimpleNamespace(choices=[SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=content, tool_calls=None),
                )]))
                response = ChatCompletionsProvider(
                    "secret", "https://example.test", "deepseek", client=client,
                ).complete([], [], tool_choice="none")
                self.assertEqual(response.contains_tool_protocol, expected)
                self.assertEqual(response.content, content)

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

    def test_preserves_provider_defaults_and_reasoning_history(self) -> None:
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
        messages = [{"role": "user", "content": "hello"},
                    {"role": "assistant", "content": None,
                     "reasoning_content": "Inspect the file first.",
                     "tool_calls": [{"id": "call-1", "type": "function",
                                     "function": {"name": "read_file", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "call-1", "content": "file"}]

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

        self.assertNotIn("extra_body", deepseek_completions.calls[0])
        self.assertEqual(deepseek_completions.calls[0]["messages"], messages)
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
