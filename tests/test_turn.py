import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.agent.turn import call_model, model_request_inputs
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


class ToolChoiceProvider(FakeProvider):
    supports_tool_choice = True

    def complete(self, messages, tools, *, tool_choice=None):
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
                "tool_choice": tool_choice,
            }
        )
        return self.responses.pop(0)


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

    def test_runtime_state_is_current_and_not_committed_or_accumulated(self) -> None:
        provider = FakeProvider([])
        context = self.context(provider)
        context.max_turns = 20
        messages = [
            {"role": "system", "content": "stable"},
            {"role": "user", "content": "task"},
        ]

        context.current_turn = 8
        first, first_tools = model_request_inputs(
            messages, context, finalization=False
        )
        context.current_turn = 20
        final, final_tools = model_request_inputs(
            messages, context, finalization=True
        )

        first_states = [
            item["content"]
            for item in first
            if str(item.get("content", "")).startswith("TinyHarness runtime state:")
        ]
        final_states = [
            item["content"]
            for item in final
            if str(item.get("content", "")).startswith("TinyHarness runtime state:")
        ]
        self.assertEqual(len(first_states), 1)
        self.assertIn("current main-agent turn: 8 / 20", first_states[0])
        self.assertIn("remaining main-agent turns: 12", first_states[0])
        self.assertIn("finalization: false", first_states[0])
        self.assertIn("tools available: true", first_states[0])
        self.assertEqual(len(final_states), 1)
        self.assertIn("current main-agent turn: 20 / 20", final_states[0])
        self.assertIn("remaining main-agent turns: 0", final_states[0])
        self.assertIn("finalization: true", final_states[0])
        self.assertIn("tools available: false", final_states[0])
        self.assertEqual(first_tools, context.tools)
        self.assertEqual(final_tools, [])
        self.assertEqual(messages[0]["content"], "stable")
        self.assertEqual(len(messages), 2)

    def test_main_turns_send_auto_and_finalization_sends_none(self) -> None:
        provider = ToolChoiceProvider(
            [
                ModelResponse("continue", None, [], "stop"),
                ModelResponse("done", None, [], "stop"),
            ]
        )
        context = self.context(provider)

        context.current_turn = 1
        call_model([], context)
        context.current_turn = context.max_turns
        call_model([], context, finalization=True)

        self.assertEqual(
            [call["tool_choice"] for call in provider.calls],
            ["auto", "none"],
        )
        self.assertTrue(provider.calls[0]["tools"])
        self.assertEqual(provider.calls[1]["tools"], [])


if __name__ == "__main__":
    unittest.main()
