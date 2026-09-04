import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.agent.turn import call_model
from tiny_harness.runtime.recovery import RecoveryPolicy


class FakeProvider:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class CallModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def context(self, provider, tools=None):
        context = create_run_context(
            provider,
            self.workspace,
            allow_subagent=False,
            recovery_policy=RecoveryPolicy(
                max_retries=0,
                base_delay_seconds=0,
                max_delay_seconds=0,
                jitter_ratio=0,
            ),
        )
        if tools is not None:
            context.tool_registry = SimpleNamespace(
                model_schemas=lambda: copy.deepcopy(tools)
            )
        return context

    def test_returns_protocol_valid_response_without_committing_it(self) -> None:
        response = ModelResponse("done", None, [], "stop")
        provider = FakeProvider([response])
        messages = [{"role": "user", "content": "task"}]

        context = self.context(provider)
        context.current_turn = 3

        result = call_model(messages, context)

        self.assertIs(result, response)
        self.assertEqual(messages, [{"role": "user", "content": "task"}])

    def test_rejects_invalid_response_without_committing_it(self) -> None:
        response = ModelResponse(
            None,
            None,
            [ToolCall("call-1", "write_file", "{}")],
            "length",
        )
        provider = FakeProvider([response])
        messages = [{"role": "user", "content": "task"}]

        with self.assertRaisesRegex(
            RuntimeError,
            "Model response is not executable: length",
        ):
            call_model(messages, self.context(provider))

        self.assertEqual(messages, [{"role": "user", "content": "task"}])

    def test_forwards_the_configured_tool_schemas(self) -> None:
        provider = FakeProvider([ModelResponse("done", None, [], "stop")])
        tools = [{"type": "function", "function": {"name": "read_file"}}]

        call_model([], self.context(provider, tools))

        self.assertEqual(provider.calls[0]["tools"], tools)


if __name__ == "__main__":
    unittest.main()
