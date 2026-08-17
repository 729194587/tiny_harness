import copy
import json
import tempfile
import unittest
from pathlib import Path

from tiny_harness.agent.loop import agent_loop
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.runtime.context import (
    ContextLimitError,
    ContextProtocolError,
    context_char_count,
    prepare_context,
)
from tiny_harness.tools.registry import tool_schemas


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "example",
            "parameters": {"type": "object"},
        },
    }
]


def tool_block(*call_ids: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "example", "arguments": "{}"},
                }
                for call_id in call_ids
            ],
        },
        *[
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"result-{call_id}",
            }
            for call_id in call_ids
        ],
    ]


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
        return self.responses.pop(0)


class RecordingEventLogger:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event_type, data=None) -> None:
        self.events.append(
            {"event_type": event_type.value, "data": dict(data or {})}
        )


class ContextPreparationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prefix = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "current task"},
        ]

    def test_character_count_uses_compact_json_for_messages_and_tools(self) -> None:
        messages = [{"role": "user", "content": "你好"}]
        expected = len(
            json.dumps(
                {"messages": messages, "tools": TOOLS},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

        self.assertEqual(context_char_count(messages, TOOLS), expected)
        self.assertGreater(
            context_char_count(messages, TOOLS),
            context_char_count(messages, []),
        )

    def test_under_budget_returns_an_independent_copy(self) -> None:
        messages = self.prefix + tool_block("call-1")
        budget = context_char_count(messages, TOOLS)

        prepared = prepare_context(messages, TOOLS, budget)

        self.assertEqual(prepared.messages, messages)
        self.assertIsNot(prepared.messages, messages)
        self.assertIsNot(prepared.messages[0], messages[0])
        self.assertEqual(prepared.before_chars, budget)
        self.assertEqual(prepared.after_chars, budget)
        self.assertEqual(prepared.dropped_blocks, 0)
        self.assertEqual(prepared.dropped_messages, 0)

    def test_drops_the_oldest_complete_block(self) -> None:
        oldest = tool_block("old")
        latest = tool_block("latest")
        messages = self.prefix + oldest + latest
        expected = self.prefix + latest
        budget = context_char_count(expected, TOOLS)

        prepared = prepare_context(messages, TOOLS, budget)

        self.assertEqual(prepared.messages, expected)
        self.assertEqual(prepared.dropped_blocks, 1)
        self.assertEqual(prepared.dropped_messages, len(oldest))
        self.assertLessEqual(prepared.after_chars, budget)

    def test_keeps_a_contiguous_suffix_of_recent_blocks(self) -> None:
        first = tool_block("first")
        second = tool_block("second")
        latest = tool_block("latest")
        messages = self.prefix + first + second + latest
        expected = self.prefix + second + latest
        budget = context_char_count(expected, TOOLS)

        prepared = prepare_context(messages, TOOLS, budget)

        self.assertEqual(prepared.messages, expected)
        self.assertEqual(prepared.dropped_blocks, 1)

    def test_multiple_tool_calls_and_results_are_never_split(self) -> None:
        multi_call = tool_block("call-1", "call-2")
        latest = tool_block("call-3")
        messages = self.prefix + multi_call + latest
        expected = self.prefix + latest
        budget = context_char_count(expected, TOOLS)

        prepared = prepare_context(messages, TOOLS, budget)

        self.assertEqual(prepared.messages, expected)
        self.assertEqual(prepared.dropped_messages, 3)
        self.assertNotIn("call-1", json.dumps(prepared.messages))
        self.assertNotIn("call-2", json.dumps(prepared.messages))

    def test_fails_when_pinned_task_and_latest_block_do_not_fit(self) -> None:
        messages = self.prefix + tool_block("latest")
        required_chars = context_char_count(messages, TOOLS)

        with self.assertRaisesRegex(ContextLimitError, "exceeds configured"):
            prepare_context(messages, TOOLS, required_chars - 1)

    def test_rejects_an_orphan_tool_result(self) -> None:
        messages = self.prefix + [
            {"role": "tool", "tool_call_id": "orphan", "content": "result"}
        ]

        with self.assertRaisesRegex(ContextProtocolError, "no preceding"):
            prepare_context(messages, TOOLS, 10_000)

    def test_rejects_incomplete_or_reordered_tool_results(self) -> None:
        incomplete = self.prefix + tool_block("call-1", "call-2")[:-1]
        reordered = self.prefix + tool_block("call-1", "call-2")
        reordered[-2], reordered[-1] = reordered[-1], reordered[-2]

        with self.assertRaisesRegex(ContextProtocolError, "must match in order"):
            prepare_context(incomplete, TOOLS, 10_000)
        with self.assertRaisesRegex(ContextProtocolError, "must match in order"):
            prepare_context(reordered, TOOLS, 10_000)

    def test_rejects_non_positive_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            prepare_context(self.prefix, TOOLS, 0)


class ContextAgentLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_agent_sends_bounded_context_and_records_trimming(self) -> None:
        prefix = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "current task"},
        ]
        oldest = tool_block("old")
        latest = tool_block("latest")
        messages = prefix + oldest + latest
        expected = prefix + latest
        budget = context_char_count(expected, tool_schemas())
        provider = FakeProvider([ModelResponse("done", None, [], "stop")])
        logger = RecordingEventLogger()

        answer = agent_loop(
            provider,
            self.workspace,
            messages,
            max_context_chars=budget,
            event_logger=logger,
        )

        self.assertEqual(answer, "done")
        self.assertEqual(provider.calls[0]["messages"], expected)
        event_names = [event["event_type"] for event in logger.events]
        self.assertEqual(
            event_names[:3],
            ["run_started", "context_trimmed", "model_requested"],
        )
        trimmed = logger.events[1]["data"]
        self.assertEqual(trimmed["turn"], 1)
        self.assertEqual(trimmed["dropped_blocks"], 1)
        self.assertEqual(trimmed["dropped_messages"], len(oldest))
        self.assertNotIn("current task", json.dumps(trimmed))

    def test_context_failure_happens_before_provider_call(self) -> None:
        messages = [{"role": "user", "content": "task"}]
        provider = FakeProvider([ModelResponse("should not run", None, [], "stop")])
        logger = RecordingEventLogger()

        with self.assertRaises(ContextLimitError):
            agent_loop(
                provider,
                self.workspace,
                messages,
                max_context_chars=1,
                event_logger=logger,
            )

        self.assertEqual(provider.calls, [])
        self.assertEqual(
            [event["event_type"] for event in logger.events],
            ["run_started", "run_failed"],
        )
        self.assertEqual(
            logger.events[-1]["data"]["error_type"],
            "ContextLimitError",
        )

    def test_protocol_failure_happens_before_provider_call(self) -> None:
        messages = [
            {"role": "user", "content": "task"},
            {"role": "tool", "tool_call_id": "orphan", "content": "result"},
        ]
        provider = FakeProvider([ModelResponse("should not run", None, [], "stop")])

        with self.assertRaises(ContextProtocolError):
            agent_loop(
                provider,
                self.workspace,
                messages,
                max_context_chars=10_000,
            )

        self.assertEqual(provider.calls, [])

    def test_unconfigured_budget_preserves_phase_three_behavior(self) -> None:
        messages = [{"role": "tool", "tool_call_id": "orphan", "content": "x"}]
        provider = FakeProvider([ModelResponse("done", None, [], "stop")])

        answer = agent_loop(provider, self.workspace, messages)

        self.assertEqual(answer, "done")
        self.assertEqual(provider.calls[0]["messages"], messages[:-1])

    def test_todo_reminder_over_budget_fails_before_next_provider_call(self) -> None:
        latest_without_reminder = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "list-3",
                        "type": "function",
                        "function": {
                            "name": "list_files",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "list-3",
                "content": "(no files)",
            },
        ]
        budget = context_char_count(latest_without_reminder, tool_schemas())
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall(f"list-{index}", "list_files", "{}")],
                    "tool_calls",
                )
                for index in range(1, 4)
            ]
            + [ModelResponse("must not be requested", None, [], "stop")]
        )

        with self.assertRaises(ContextLimitError):
            agent_loop(
                provider,
                self.workspace,
                [],
                max_context_chars=budget,
            )

        self.assertEqual(len(provider.calls), 3)


if __name__ == "__main__":
    unittest.main()
