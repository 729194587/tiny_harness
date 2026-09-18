import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.agent.turn import (
    TOOL_USE_EFFICIENCY_GUIDANCE, call_model, model_request_inputs,
)
from tiny_harness.runtime.context import CompactionConfig, validate_active_request
from tiny_harness.agent.session import AgentSession
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

    def test_large_read_file_is_bounded_on_every_request_without_mutating_history(self):
        large = "HEAD\n" + "x" * 100_000 + "\nTAIL"
        threshold = CompactionConfig().large_result_chars
        items = [
            ("large", "read_file", large),
            ("small", "read_file", "short result"),
            ("boundary", "read_file", "b" * threshold),
            ("over", "read_file", "c" * (threshold + 1)),
            ("other", "bash", large),
        ]
        messages = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": call_id, "type": "function", "function": {
                    "name": name, "arguments": '{"path":"file.py"}',
                }} for call_id, name, _ in items
            ]},
            *[{"role": "tool", "tool_call_id": call_id, "content": content}
              for call_id, _, content in items],
        ]
        original = copy.deepcopy(messages)
        provider = FakeProvider([ModelResponse("done", None, [], "stop")] * 3)
        context = self.context(provider)
        self.assertIsNone(context.compactor)
        for finalization in (False, False, True):
            call_model(messages, context, finalization=finalization)
            request = provider.calls[-1]["messages"]
            validate_active_request(request, "task")
            results = [item for item in request if item["role"] == "tool"]
            self.assertEqual([item["tool_call_id"] for item in results],
                             [item[0] for item in items])
            self.assertLess(len(results[0]["content"]), 3_000)
            self.assertIn("HEAD", results[0]["content"])
            self.assertIn("TAIL", results[0]["content"])
            self.assertIn(str(len(large)), results[0]["content"])
            self.assertEqual(results[1]["content"], "short result")
            self.assertEqual(results[2]["content"], "b" * threshold)
            self.assertLess(len(results[3]["content"]), 3_000)
            self.assertEqual(results[4]["content"], large)
            self.assertEqual(messages, original)

    def test_session_preserves_bounded_read_navigation_across_submissions(self):
        content = "start\n" + "x" * 100_000 + "\nend"
        (self.workspace / "large.txt").write_text(content, encoding="utf-8")
        provider = FakeProvider([
            ModelResponse(None, None, [ToolCall(
                "read", "read_file", '{"path":"large.txt"}'
            )], "tool_calls"),
            ModelResponse("done", None, [], "stop"),
            ModelResponse("done again", None, [], "stop"),
        ])
        session = AgentSession(provider, self.workspace, "system")
        self.assertEqual(session.submit("read the file"), "done")
        self.assertEqual(session.submit("continue"), "done again")
        results = [item for item in session.messages if item["role"] == "tool"]
        artifacts = list(self.workspace.glob(".tinyharness/context/tool-results/*.txt"))
        self.assertEqual(artifacts, [])
        header = "[lines 1-1 of 3 | large.txt]"
        self.assertIn(header, results[0]["content"])
        self.assertIn("continue with start_line=2, start_column=1", results[0]["content"])
        self.assertEqual((self.workspace / "large.txt").read_text(encoding="utf-8"), content)
        for call in provider.calls[1:]:
            result = next(item for item in call["messages"] if item["role"] == "tool")
            self.assertEqual(result["content"], results[0]["content"])
            self.assertLessEqual(len(result["content"]), 30_000)

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
        self.assertEqual(first_states, [])
        self.assertEqual(final_states, [])
        later, later_tools = model_request_inputs(messages, context, finalization=False)
        self.assertEqual((first, first_tools), (later, later_tools))
        self.assertEqual(first_tools, context.tools)
        self.assertEqual(final_tools, [])
        self.assertEqual(messages[0]["content"], "stable")
        self.assertEqual(len(messages), 2)

    def test_efficiency_guidance_is_request_only_and_requires_available_tools(self):
        messages = [
            {"role": "system", "content": "task guidance"},
            {"role": "user", "content": "task"},
        ]
        original = copy.deepcopy(messages)
        guidance = {"role": "system", "content": TOOL_USE_EFFICIENCY_GUIDANCE}
        for has_tools, finalization in ((True, False), (False, False), (True, True)):
            with self.subTest(tools=has_tools, final=finalization):
                tools = [{"type": "function", "function": {"name": "read_file"}}] if has_tools else []
                context = self.context(FakeProvider([]), tools)
                for _ in range(2):
                    request, schemas = model_request_inputs(
                        messages, context, finalization=finalization,
                    )
                    self.assertEqual(request.count(guidance), int(has_tools and not finalization))
                    self.assertEqual(schemas, [] if finalization else tools)
                    self.assertEqual(messages, original)
                    self.assertEqual(request[0], original[0])
                    if finalization:
                        self.assertEqual(request[:-1], original)
                        self.assertEqual(request[-1]["role"], "system")
                        self.assertIn("The tool-use phase has ended", request[-1]["content"])
                    else:
                        self.assertEqual(request[-1], original[-1])

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
