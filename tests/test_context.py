import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tiny_harness.agent.loop import run_agent as agent_loop
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.runtime.context import (
    CompactionConfig,
    ContextArtifactError,
    ContextCompactor,
    ContextLimitError,
    ContextProtocolError,
    ContextSummaryError,
    context_token_count,
    prepare_context,
    trim_context_blocks,
    validate_active_request,
)
from tiny_harness.runtime.hooks import HookBlock, ToolHooks
from tiny_harness.runtime.recovery import RecoveryPolicy
from tiny_harness.runtime.skills import discover_skills


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "example",
            "parameters": {"type": "object"},
        },
    }
]


def tool_block(call_id: str, result: str = "result", assistant_text=None):
    return [
        {
            "role": "assistant",
            "content": assistant_text,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "example", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def multi_tool_block(items: list[tuple[str, str]]):
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
                for call_id, _ in items
            ],
        },
        *[
            {"role": "tool", "tool_call_id": call_id, "content": result}
            for call_id, result in items
        ],
    ]


class FakeProvider:
    def __init__(self, responses=()) -> None:
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append(
            {"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools)}
        )
        if not self.responses:
            raise AssertionError("FakeProvider has no response left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingEventLogger:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event_type, data=None) -> None:
        self.events.append(
            {"event_type": event_type.value, "data": dict(data or {})}
        )


class ContextPrimitiveTest(unittest.TestCase):
    def test_token_estimate_includes_messages_and_tool_schemas(self) -> None:
        messages = [{"role": "user", "content": "你好"}]
        expected = len(
            json.dumps(
                {"messages": messages, "tools": TOOLS},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        self.assertEqual(context_token_count(messages, TOOLS), (expected + 3) // 4)

    def test_context_sizing_uses_injected_token_meter(self) -> None:
        class RecordingMeter:
            def __init__(self) -> None:
                self.calls = []

            def estimate(self, messages, tools) -> int:
                self.calls.append((messages, tools))
                return 10

        meter = RecordingMeter()
        compactor = ContextCompactor(
            Path.cwd(), FakeProvider(), TOOLS, 100, token_meter=meter
        )

        prepared = prepare_context(
            [{"role": "user", "content": "task"}],
            compactor,
            "No todos.",
            "task",
        )

        self.assertEqual(prepared.after_tokens, 10)
        self.assertGreaterEqual(len(meter.calls), 2)

    def test_compatibility_helper_drops_complete_old_blocks(self) -> None:
        prefix = [{"role": "user", "content": "task"}]
        old = tool_block("old")
        latest = tool_block("latest")
        budget = context_token_count(prefix + latest, TOOLS)

        prepared = trim_context_blocks(prefix + old + latest, TOOLS, budget)

        self.assertEqual(prepared.messages, prefix + latest)
        self.assertEqual(prepared.dropped_blocks, 1)
        self.assertEqual(prepared.dropped_messages, 2)

    def test_protocol_validation_rejects_orphan_and_reordered_results(self) -> None:
        orphan = [{"role": "tool", "tool_call_id": "x", "content": "bad"}]
        reordered = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {}},
                    {"id": "b", "type": "function", "function": {}},
                ],
            },
            {"role": "tool", "tool_call_id": "b", "content": "b"},
            {"role": "tool", "tool_call_id": "a", "content": "a"},
        ]
        with self.assertRaises(ContextProtocolError):
            trim_context_blocks(orphan, TOOLS, 2_500)
        with self.assertRaises(ContextProtocolError):
            trim_context_blocks(reordered, TOOLS, 2_500)

    def test_active_request_must_match_latest_real_user_message(self) -> None:
        messages = [
            {"role": "user", "content": "current task"},
            {
                "role": "user",
                "name": "tinyharness_todo_state",
                "content": "control state",
            },
        ]

        validate_active_request(messages, "current task")
        with self.assertRaisesRegex(
            ContextProtocolError,
            "does not match the active request",
        ):
            validate_active_request(messages, "different task")


class ContextCompactorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)
        self.prefix = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "ORIGINAL_TASK"},
        ]

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def compactor(self, provider=None, *, max_tokens=2_500, **config):
        return ContextCompactor(
            self.workspace,
            provider or FakeProvider(),
            TOOLS,
            max_tokens,
            config=CompactionConfig(**config),
        )

    def prepare(self, compactor, messages, todo_state="No todos."):
        active_request = next(
            (
                str(message.get("content") or "")
                for message in reversed(messages)
                if message.get("role") == "user"
                and not str(message.get("name") or "").startswith(
                    "tinyharness_"
                )
            ),
            "",
        )
        return prepare_context(
            messages,
            compactor,
            todo_state,
            active_request,
        )

    def test_under_budget_short_history_is_unchanged_and_writes_no_artifacts(self):
        messages = self.prefix + tool_block("one", "short")
        prepared = self.prepare(self.compactor(), messages)

        self.assertEqual(prepared.messages, messages)
        self.assertFalse(prepared.changed)
        self.assertFalse((self.workspace / ".tinyharness").exists())

    def test_under_budget_does_not_shorten_tool_results(self):
        results = [f"RESULT_{index}_" + (str(index) * 500) for index in range(6)]
        messages = self.prefix + [
            message
            for index, result in enumerate(results)
            for message in tool_block(str(index), result)
        ]
        provider = FakeProvider()

        prepared = self.prepare(
            self.compactor(
                provider,
                max_tokens=2_500,
                micro_result_chars=120,
            ),
            messages,
        )

        visible_results = [
            message["content"]
            for message in prepared.messages
            if message.get("role") == "tool"
        ]
        self.assertEqual(visible_results, results)
        self.assertEqual(prepared.shortened_results, 0)
        self.assertEqual(prepared.archived_messages, 0)
        self.assertEqual(provider.calls, [])

    def test_new_tool_batch_does_not_evict_previous_batch_under_budget(self):
        reads = [(f"read-{index}", f"READ_{index}_" + "R" * 400) for index in range(3)]
        greps = [(f"grep-{index}", f"GREP_{index}_" + "G" * 400) for index in range(3)]
        messages = self.prefix + multi_tool_block(reads) + multi_tool_block(greps)

        prepared = self.prepare(
            self.compactor(max_tokens=2_500, micro_result_chars=120),
            messages,
        )

        visible = json.dumps(prepared.messages, ensure_ascii=False)
        for _, evidence in reads + greps:
            self.assertIn(evidence, visible)
        self.assertEqual(prepared.shortened_results, 0)
        self.assertEqual(prepared.archived_messages, 0)
        self.assertFalse(prepared.summarized)

    def test_compaction_starts_only_under_pressure(self):
        full_output = "PRESSURE_EVIDENCE_" + "A" * 2_000
        messages = self.prefix + tool_block("large", full_output)
        measured = context_token_count(messages, TOOLS)
        under_max = (measured * 5 + 3) // 4 + 10
        over_max = (measured * 5) // 4 - 10

        under = self.prepare(
            self.compactor(
                max_tokens=under_max,
                large_result_chars=100,
                result_preview_chars=40,
            ),
            messages,
        )
        over = self.prepare(
            self.compactor(
                max_tokens=over_max,
                large_result_chars=100,
                result_preview_chars=40,
            ),
            messages,
        )

        self.assertEqual(under.messages[-1]["content"], full_output)
        self.assertEqual(under.persisted_results, 0)
        self.assertEqual(over.persisted_results, 1)
        replacement = over.messages[-1]["content"]
        self.assertIn("Head:\nPRESSURE_EVIDENCE_", replacement)
        self.assertIn("middle omitted; full output persisted", replacement)
        self.assertIn("Tail:\n" + "A" * 20, replacement)
        relative = next(
            line.removeprefix("Full output: ")
            for line in replacement.splitlines()
            if line.startswith("Full output: ")
        )
        self.assertEqual(
            (self.workspace / relative).read_text(encoding="utf-8"),
            full_output,
        )
        self.assertLessEqual(over.after_tokens, int(over_max * 0.8))

    def test_compaction_event_identifies_persisted_tool_results(self):
        logger = RecordingEventLogger()
        full_output = "AUDITABLE_RESULT_" + "A" * 2_000
        messages = self.prefix + tool_block("read-a", full_output)
        measured = context_token_count(messages, TOOLS)
        compactor = ContextCompactor(
            self.workspace,
            FakeProvider(),
            TOOLS,
            (measured * 5) // 4 - 10,
            config=CompactionConfig(
                large_result_chars=100,
                result_preview_chars=40,
            ),
            event_logger=logger,
        )

        prepared = self.prepare(compactor, messages)

        self.assertEqual(prepared.persisted_tool_call_ids, ("read-a",))
        compacted = next(
            event["data"]
            for event in logger.events
            if event["event_type"] == "context_compacted"
        )
        self.assertEqual(compacted["persisted_tool_call_ids"], ["read-a"])

    def test_trigger_at_eighty_percent_compacts_old_results_to_fifty_five_percent(self):
        provider = FakeProvider()
        old_a = tool_block("old-a", "A" * 2_600)
        old_b = tool_block("old-b", "B" * 2_600)
        old_c = tool_block("old-c", "C" * 2_600)
        recent = tool_block("recent", "RECENT_" + "R" * 800)
        messages = self.prefix + old_a + old_b + old_c + recent
        compactor = self.compactor(
            provider,
            max_tokens=2_500,
            result_preview_chars=40,
        )

        prepared = self.prepare(compactor, messages)

        self.assertGreater(context_token_count(messages, TOOLS), compactor.soft_limit)
        self.assertLessEqual(prepared.after_tokens, compactor.target_limit)
        self.assertEqual(prepared.persisted_tool_call_ids, ("old-a", "old-b"))
        self.assertEqual(prepared.messages[-2:], recent)
        self.assertFalse(prepared.summarized)
        self.assertEqual(provider.calls, [])

    def test_persisted_result_keeps_bounded_evidence_stub_and_full_artifact(self):
        full_output = "IDENTIFYING_HEAD_" + "X" * 5_000 + "_IDENTIFYING_TAIL"
        messages = self.prefix + tool_block("read-source", full_output)
        messages[-2]["tool_calls"][0]["function"] = {
            "name": "read_file",
            "arguments": json.dumps(
                {"path": "src/important.py", "extra": "Y" * 1_000}
            ),
        }
        compactor = self.compactor(
            max_tokens=1_250,
            result_preview_chars=80,
        )

        prepared = self.prepare(compactor, messages)

        stub = prepared.messages[-1]["content"]
        self.assertIn("Tool: read_file", stub)
        self.assertIn('Arguments: {"path": "src/important.py"', stub)
        self.assertIn("...[arguments truncated]", stub)
        self.assertIn(
            "Content SHA-256: " + hashlib.sha256(full_output.encode()).hexdigest(),
            stub,
        )
        self.assertIn("IDENTIFYING_HEAD_", stub)
        self.assertIn("IDENTIFYING_TAIL", stub)
        self.assertNotIn("X" * 1_000, stub)
        relative = next(
            line.removeprefix("Full output: ")
            for line in stub.splitlines()
            if line.startswith("Full output: ")
        )
        self.assertEqual(
            (self.workspace / relative).read_text(encoding="utf-8"),
            full_output,
        )

    def test_message_count_alone_does_not_trigger_compaction(self):
        messages = self.prefix + [
            {"role": "assistant", "content": f"short-{index}"}
            for index in range(60)
        ]
        provider = FakeProvider()

        prepared = self.prepare(
            self.compactor(provider, max_tokens=25_000, max_messages=50),
            messages,
        )

        self.assertEqual(prepared.messages, messages)
        self.assertEqual(prepared.archived_messages, 0)
        self.assertEqual(prepared.shortened_results, 0)
        self.assertFalse(prepared.summarized)
        self.assertEqual(provider.calls, [])

    def test_recent_execution_units_are_preserved(self):
        old = tool_block("old", "OLD_" + "X" * 4_000)
        recent_a = multi_tool_block(
            [("recent-a1", "RECENT_A1"), ("recent-a2", "RECENT_A2")]
        )
        recent_b = multi_tool_block(
            [("recent-b1", "RECENT_B1"), ("recent-b2", "RECENT_B2")]
        )
        messages = self.prefix + old + recent_a + recent_b

        prepared = self.prepare(
            self.compactor(
                max_tokens=1_250,
                large_result_chars=500,
                result_preview_chars=40,
            ),
            messages,
        )

        self.assertEqual(prepared.messages[-6:], recent_a + recent_b)
        self.assertEqual(prepared.persisted_results, 1)
        self.assertLessEqual(prepared.after_tokens, 1_000)

    def test_progressive_compression_summarizes_only_when_pruning_is_insufficient(self):
        prune_provider = FakeProvider()
        prune_messages = (
            self.prefix
            + tool_block("old-large", "OLD_TOOL_" + "X" * 4_000)
            + tool_block("latest", "LATEST")
        )
        pruned = self.prepare(
            self.compactor(
                prune_provider,
                max_tokens=1_250,
                large_result_chars=500,
                result_preview_chars=40,
            ),
            prune_messages,
        )

        summary_provider = FakeProvider(
            [ModelResponse("OLDER_HISTORY_SUMMARY", None, [], "stop")]
        )
        summary_messages = (
            self.prefix
            + [{"role": "assistant", "content": "OLD_ASSISTANT_" + "Y" * 3_000}]
            + tool_block("latest", "LATEST")
        )
        summarized = self.prepare(
            self.compactor(summary_provider, max_tokens=875),
            summary_messages,
        )

        self.assertEqual(pruned.persisted_results, 1)
        self.assertFalse(pruned.summarized)
        self.assertEqual(prune_provider.calls, [])
        self.assertTrue(summarized.summarized)
        self.assertEqual(len(summary_provider.calls), 1)
        self.assertLessEqual(summarized.after_tokens, 700)

    def test_tool_batch_is_atomic_during_compaction(self):
        latest_batch = multi_tool_block(
            [
                ("a", "A_RESULT"),
                ("b", "B_RESULT"),
                ("c", "C_RESULT"),
            ]
        )
        messages = (
            self.prefix
            + tool_block("old", "OLD_" + "X" * 4_000)
            + latest_batch
        )

        prepared = self.prepare(
            self.compactor(
                max_tokens=1_250,
                large_result_chars=500,
                result_preview_chars=40,
            ),
            messages,
        )

        self.assertEqual(prepared.messages[-4:], latest_batch)
        assistant = prepared.messages[-4]
        results = prepared.messages[-3:]
        self.assertEqual(
            [call["id"] for call in assistant["tool_calls"]],
            [result["tool_call_id"] for result in results],
        )

    def test_summary_uses_original_selected_history(self):
        sentinel = "UNIQUE_ORIGINAL_SELECTED_SENTINEL"
        provider = FakeProvider(
            [ModelResponse("FACTUAL_SUMMARY", None, [], "stop")]
        )
        messages = (
            self.prefix
            + [{"role": "assistant", "content": sentinel + "X" * 3_000}]
            + tool_block("latest", "LATEST_EVIDENCE")
        )

        prepared = self.prepare(
            self.compactor(provider, max_tokens=500, max_messages=3),
            messages,
        )

        self.assertTrue(prepared.summarized)
        self.assertIn(
            sentinel,
            json.dumps(provider.calls[0]["messages"], ensure_ascii=False),
        )

    def test_multi_turn_history_is_protocol_safe_and_unchanged_under_budget(self):
        messages = (
            self.prefix
            + tool_block("first", "FIRST_RESULT")
            + [{"role": "assistant", "content": "first answer"}]
            + [{"role": "user", "content": "SECOND_TASK"}]
            + tool_block("second", "SECOND_RESULT")
            + [{"role": "assistant", "content": "second answer"}]
        )

        prepared = self.prepare(
            self.compactor(max_tokens=25_000),
            messages,
        )

        self.assertEqual(prepared.messages, messages)

    def test_multi_turn_summary_keeps_current_request_and_latest_tool_batch(self):
        provider = FakeProvider([ModelResponse("OLD_TURN_SUMMARY", None, [], "stop")])
        latest = tool_block("latest-a", "A") + tool_block(
            "latest-b",
            "LATEST_EVIDENCE",
        )
        messages = (
            self.prefix
            + tool_block("old", "OLD_RESULT", assistant_text="X" * 4_000)
            + [{"role": "assistant", "content": "old answer"}]
            + [{"role": "user", "content": "CURRENT_TASK"}]
            + latest
        )
        compactor = ContextCompactor(
            self.workspace,
            provider,
            TOOLS,
            450,
            config=CompactionConfig(max_messages=50),
        )

        prepared = self.prepare(compactor, messages)

        compacted = json.dumps(prepared.messages, ensure_ascii=False)
        self.assertTrue(prepared.summarized)
        self.assertIn("OLD_TURN_SUMMARY", compacted)
        self.assertIn("CURRENT_TASK", compacted)
        self.assertIn("latest-b", compacted)
        self.assertIn("LATEST_EVIDENCE", compacted)
        self.assertNotIn("OLD_RESULT", compacted)

    def test_new_turn_preserves_budgeted_recent_execution_tail(self):
        provider = FakeProvider([ModelResponse("PRIOR_SUMMARY", None, [], "stop")])
        messages = (
            self.prefix
            + [{"role": "assistant", "content": "X" * 4_000}]
            + tool_block("prior-tool", "RECENT_EVIDENCE")
            + [{"role": "assistant", "content": "prior answer"}]
            + [{"role": "user", "content": "FOLLOW_UP_TASK"}]
        )
        compactor = ContextCompactor(
            self.workspace,
            provider,
            TOOLS,
            450,
            config=CompactionConfig(max_messages=50),
        )

        prepared = self.prepare(compactor, messages)

        compacted = json.dumps(prepared.messages, ensure_ascii=False)
        self.assertTrue(prepared.summarized)
        self.assertIn("PRIOR_SUMMARY", compacted)
        self.assertIn("RECENT_EVIDENCE", compacted)
        self.assertIn("prior answer", compacted)
        self.assertIn("FOLLOW_UP_TASK", compacted)

    def test_multi_turn_trim_drops_old_turn_without_splitting_latest_batch(self):
        latest_batch = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {}},
                    {"id": "b", "type": "function", "function": {}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "A"},
            {"role": "tool", "tool_call_id": "b", "content": "B"},
        ]
        messages = (
            self.prefix
            + tool_block("old", "X" * 3_000)
            + [{"role": "assistant", "content": "old answer"}]
            + [{"role": "user", "content": "CURRENT_TASK"}]
            + latest_batch
        )

        prepared = trim_context_blocks(messages, TOOLS, 250)

        self.assertIn({"role": "user", "content": "CURRENT_TASK"}, prepared.messages)
        self.assertEqual(prepared.messages[-3:], latest_batch)
        self.assertNotIn("X" * 3_000, json.dumps(prepared.messages))

    def test_current_todo_marker_is_replaced_without_accumulating(self):
        compactor = self.compactor()
        first = self.prepare(compactor, self.prefix, "[>] FIRST_TODO")
        second = self.prepare(
            compactor,
            first.messages,
            "[x] UPDATED_TODO",
        )

        markers = [
            message
            for message in second.messages
            if message.get("name") == "tinyharness_todo_state"
        ]
        self.assertEqual(len(markers), 1)
        self.assertIn("UPDATED_TODO", markers[0]["content"])
        self.assertNotIn("FIRST_TODO", markers[0]["content"])

    def test_summary_is_last_resort_and_preserves_task_todo_and_latest_block(self):
        provider = FakeProvider([ModelResponse("FACTUAL_SUMMARY", None, [], "stop")])
        messages = (
            self.prefix
            + tool_block("old", "old", assistant_text="X" * 3_000)
            + tool_block("latest", "LATEST_EVIDENCE")
        )
        compactor = ContextCompactor(
            self.workspace,
            provider,
            TOOLS,
            375,
            config=CompactionConfig(max_messages=50),
        )

        prepared = self.prepare(compactor, messages, "[>] CURRENT_TODO")

        self.assertTrue(prepared.summarized)
        self.assertLessEqual(prepared.after_tokens, 375)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0]["tools"], [])
        self.assertLessEqual(
            context_token_count(provider.calls[0]["messages"], []),
            375,
        )
        compacted_json = json.dumps(prepared.messages, ensure_ascii=False)
        self.assertIn("ORIGINAL_TASK", compacted_json)
        self.assertIn("CURRENT_TODO", compacted_json)
        self.assertIn("FACTUAL_SUMMARY", compacted_json)
        self.assertIn("latest", compacted_json)
        self.assertIn("LATEST_EVIDENCE", compacted_json)
        self.assertNotIn("\"old\"", compacted_json)

    def test_invalid_summary_response_fails_explicitly(self):
        provider = FakeProvider([ModelResponse(None, None, [], "length")])
        messages = (
            self.prefix
            + tool_block("old", "result", assistant_text="X" * 3_000)
            + tool_block("latest", "evidence")
        )
        compactor = ContextCompactor(
            self.workspace,
            provider,
            TOOLS,
            375,
        )
        original = copy.deepcopy(messages)

        with self.assertRaises(ContextSummaryError):
            self.prepare(compactor, messages)

        self.assertEqual(messages, original)
        self.assertTrue(
            list(
                (self.workspace / ".tinyharness/context/transcripts").glob(
                    "*.jsonl"
                )
            )
        )

    def test_impossible_mandatory_context_fails_before_summary_api_call(self):
        provider = FakeProvider([ModelResponse("must not run", None, [], "stop")])
        messages = [{"role": "user", "content": "T" * 2_000}]
        compactor = ContextCompactor(
            self.workspace,
            provider,
            TOOLS,
            75,
        )

        with self.assertRaises(ContextLimitError):
            self.prepare(compactor, messages)
        self.assertEqual(provider.calls, [])

    def test_reactive_compaction_meets_explicit_shrink_margin(self):
        provider = FakeProvider([ModelResponse("REACTIVE_SUMMARY", None, [], "stop")])
        messages = (
            self.prefix
            + tool_block("old", "old", assistant_text="X" * 10_000)
            + tool_block("latest", "LATEST_EVIDENCE")
        )
        failed_tokens = context_token_count(messages, TOOLS)

        prepared = self.compactor(
            provider,
            max_tokens=25_000,
            reactive_target_ratio=0.75,
        ).reactive_compact(
            messages,
            "[>] CURRENT_TODO",
            failed_request_tokens=failed_tokens,
        )

        self.assertLessEqual(prepared.after_tokens, int(failed_tokens * 0.75))
        self.assertLessEqual(
            context_token_count(provider.calls[0]["messages"], []),
            int(failed_tokens * 0.75),
        )
        compacted = json.dumps(prepared.messages, ensure_ascii=False)
        self.assertIn("ORIGINAL_TASK", compacted)
        self.assertIn("CURRENT_TODO", compacted)
        self.assertIn("REACTIVE_SUMMARY", compacted)
        self.assertIn("LATEST_EVIDENCE", compacted)
        self.assertNotIn("\"old\"", compacted)

    def test_reactive_compaction_without_old_history_fails_before_summary(self):
        provider = FakeProvider([ModelResponse("must not run", None, [], "stop")])
        messages = self.prefix + tool_block("latest", "evidence")

        with self.assertRaisesRegex(ContextLimitError, "no older history"):
            self.compactor(provider).reactive_compact(
                messages,
                "No todos.",
                failed_request_tokens=context_token_count(messages, TOOLS),
            )

        self.assertEqual(provider.calls, [])

    def test_existing_artifact_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside_name:
            link = self.workspace / ".tinyharness"
            try:
                link.symlink_to(Path(outside_name), target_is_directory=True)
            except OSError as error:
                self.skipTest(f"Creating symlinks is unavailable: {error}")

            messages = self.prefix + tool_block("large", "A" * 2_000)
            with self.assertRaises(ContextArtifactError):
                self.prepare(
                    self.compactor(
                        max_tokens=1_250,
                        tool_result_batch_chars=1_000,
                        large_result_chars=100,
                    ),
                    messages,
                )

    def test_tool_result_file_symlink_is_never_followed(self):
        content = "SENSITIVE" * 100
        call_id = "call1"
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        fixed_uuid = "f" * 32
        artifact_dir = self.workspace / ".tinyharness/context/tool-results"
        artifact_dir.mkdir(parents=True)

        with tempfile.TemporaryDirectory() as outside_name:
            outside_target = Path(outside_name) / "escaped.txt"
            candidate = artifact_dir / f"{call_id}-{digest}-{fixed_uuid}.txt"
            try:
                candidate.symlink_to(outside_target)
            except OSError as error:
                self.skipTest(f"Creating symlinks is unavailable: {error}")

            compactor = self.compactor()
            with patch(
                "tiny_harness.runtime.context.uuid4",
                return_value=SimpleNamespace(hex=fixed_uuid),
            ):
                with self.assertRaises(ContextArtifactError):
                    compactor._persist_tool_result(call_id, content)

            self.assertFalse(outside_target.exists())

    def test_tool_result_exclusive_create_never_overwrites_existing_file(self):
        content = "NEW_CONTENT"
        call_id = "call1"
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        fixed_uuid = "e" * 32
        artifact_dir = self.workspace / ".tinyharness/context/tool-results"
        artifact_dir.mkdir(parents=True)
        candidate = artifact_dir / f"{call_id}-{digest}-{fixed_uuid}.txt"
        candidate.write_text("EXISTING", encoding="utf-8")

        with patch(
            "tiny_harness.runtime.context.uuid4",
            return_value=SimpleNamespace(hex=fixed_uuid),
        ):
            with self.assertRaises(ContextArtifactError):
                self.compactor()._persist_tool_result(call_id, content)

        self.assertEqual(candidate.read_text(encoding="utf-8"), "EXISTING")


class ContextAgentLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_budget_enables_compact_tool(self):
        provider = FakeProvider([ModelResponse("done", None, [], "stop")])

        agent_loop(
            provider,
            self.workspace,
            [{"role": "user", "content": "task"}],
            max_context_tokens=25_000,
        )

        names = [schema["function"]["name"] for schema in provider.calls[0]["tools"]]
        self.assertIn("compact", names)

    def test_automatic_summary_does_not_consume_main_turn(self):
        provider = FakeProvider(
            [
                ModelResponse("SUMMARY", None, [], "stop"),
                ModelResponse("done", None, [], "stop"),
            ]
        )
        messages = [
            {"role": "user", "content": "ORIGINAL_TASK"},
            *tool_block("old", "old", assistant_text="X" * 10_000),
            *tool_block("latest", "LATEST"),
        ]
        logger = RecordingEventLogger()

        answer = agent_loop(
            provider,
            self.workspace,
            messages,
            max_turns=1,
            max_context_tokens=1_750,
            event_logger=logger,
        )

        self.assertEqual(answer, "done")
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(provider.calls[0]["tools"], [])
        self.assertEqual(provider.calls[1]["tools"], [])
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "execution turn budget is exhausted"
                in message.get("content", "")
                for message in provider.calls[1]["messages"]
            )
        )
        event_names = [event["event_type"] for event in logger.events]
        self.assertLess(
            event_names.index("context_summary_requested"),
            event_names.index("model_requested"),
        )
        self.assertNotIn("ORIGINAL_TASK", json.dumps(logger.events))
        self.assertNotIn("SUMMARY", json.dumps(logger.events))

    def test_failed_prepare_never_commits_partial_canonical_history(self):
        provider = FakeProvider(
            [ModelResponse(None, None, [], "length")]
        )
        messages = [
            {"role": "user", "content": "ORIGINAL_TASK"},
            *tool_block("old", "old", assistant_text="X" * 10_000),
            *tool_block("latest", "LATEST_EVIDENCE"),
        ]
        original = copy.deepcopy(messages)

        with self.assertRaises(ContextSummaryError):
            agent_loop(
                provider,
                self.workspace,
                messages,
                max_turns=1,
                max_context_tokens=1_500,
                skill_catalog=discover_skills(self.workspace, sources=()),
            )

        self.assertEqual(messages, original)
        self.assertEqual(len(provider.calls), 1)

    def test_summary_model_calls_use_the_same_bounded_recovery_policy(self):
        provider = FakeProvider(
            [
                ModelProviderError(ModelErrorKind.SERVER_UNAVAILABLE),
                ModelResponse("SUMMARY", None, [], "stop"),
                ModelResponse("done", None, [], "stop"),
            ]
        )
        messages = [
            {"role": "user", "content": "task"},
            *tool_block("old", "old", assistant_text="X" * 10_000),
            *tool_block("latest", "latest"),
        ]
        logger = RecordingEventLogger()

        answer = agent_loop(
            provider,
            self.workspace,
            messages,
            max_turns=1,
            max_context_tokens=1_750,
            event_logger=logger,
            recovery_policy=RecoveryPolicy(
                max_retries=1,
                base_delay_seconds=0,
                max_delay_seconds=0,
                jitter_ratio=0,
            ),
        )

        self.assertEqual(answer, "done")
        requested = [
            event["data"]
            for event in logger.events
            if event["event_type"] == "model_requested"
        ]
        self.assertEqual(
            [(event["purpose"], event["attempt"]) for event in requested],
            [("summary", 1), ("summary", 2), ("main", 1)],
        )

    def test_manual_compact_runs_after_all_tools_in_batch(self):
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall(
                            "write-1",
                            "write_file",
                            '{"path":"done.txt","content":"DONE"}',
                        ),
                        ToolCall("compact-1", "compact", "{}"),
                    ],
                    "tool_calls",
                ),
                ModelResponse("BATCH_SUMMARY", None, [], "stop"),
                ModelResponse("finished", None, [], "stop"),
            ]
        )
        messages = (
            [{"role": "user", "content": "earlier task"}]
            + tool_block("previous", "PREVIOUS_EVIDENCE")
            + [{"role": "assistant", "content": "earlier answer"}]
            + [{"role": "user", "content": "write then compact"}]
        )

        answer = agent_loop(
            provider,
            self.workspace,
            messages,
            max_context_tokens=25_000,
        )

        self.assertEqual(answer, "finished")
        self.assertEqual(
            (self.workspace / "done.txt").read_text(encoding="utf-8"),
            "DONE",
        )
        summary_input = json.dumps(provider.calls[1]["messages"])
        self.assertIn("PREVIOUS_EVIDENCE", summary_input)
        self.assertNotIn("write-1", summary_input)
        self.assertNotIn("compact-1", summary_input)
        next_request = provider.calls[2]["messages"]
        self.assertEqual(
            [message["tool_call_id"] for message in next_request[-2:]],
            ["write-1", "compact-1"],
        )

    def test_blocked_compact_does_not_call_summary_model(self):
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("compact-1", "compact", "{}")],
                    "tool_calls",
                ),
                ModelResponse("continued", None, [], "stop"),
            ]
        )
        hooks = ToolHooks()
        hooks.register_pre(
            lambda context: (
                HookBlock("compaction blocked")
                if context.tool_name == "compact"
                else None
            )
        )

        with contextlib.redirect_stdout(io.StringIO()):
            answer = agent_loop(
                provider,
                self.workspace,
                [{"role": "user", "content": "task"}],
                max_context_tokens=25_000,
                tool_hooks=hooks,
            )

        self.assertEqual(answer, "continued")
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("compaction blocked", provider.calls[1]["messages"][-1]["content"])

    def test_unconfigured_budget_preserves_default_runtime_tools(self):
        provider = FakeProvider([ModelResponse("done", None, [], "stop")])
        agent_loop(provider, self.workspace, [])
        names = [schema["function"]["name"] for schema in provider.calls[0]["tools"]]
        self.assertNotIn("compact", names)
        self.assertFalse((self.workspace / ".tinyharness").exists())

    def test_context_rejection_compacts_once_and_retries_same_turn(self):
        provider = FakeProvider(
            [
                ModelProviderError(ModelErrorKind.CONTEXT_LENGTH),
                ModelResponse("REACTIVE_SUMMARY", None, [], "stop"),
                ModelResponse("done", None, [], "stop"),
            ]
        )
        messages = [
            {"role": "user", "content": "ORIGINAL_TASK"},
            *tool_block("old", "old", assistant_text="X" * 10_000),
            *tool_block("latest", "LATEST_EVIDENCE"),
        ]
        logger = RecordingEventLogger()

        answer = agent_loop(
            provider,
            self.workspace,
            messages,
            max_turns=1,
            max_context_tokens=25_000,
            event_logger=logger,
            recovery_policy=RecoveryPolicy(
                max_retries=0,
                base_delay_seconds=0,
                max_delay_seconds=0,
                jitter_ratio=0,
            ),
        )

        self.assertEqual(answer, "done")
        self.assertEqual(len(provider.calls), 3)
        self.assertNotIn("compact", [
            schema["function"]["name"] for schema in provider.calls[1]["tools"]
        ])
        self.assertIn("REACTIVE_SUMMARY", json.dumps(provider.calls[2]["messages"]))
        compacted = next(
            event["data"]
            for event in logger.events
            if event["event_type"] == "context_compacted"
            and event["data"]["reason"] == "reactive"
        )
        self.assertLessEqual(
            compacted["after_tokens"],
            int(compacted["before_tokens"] * 0.75),
        )
        requested = [
            event["data"]
            for event in logger.events
            if event["event_type"] == "model_requested"
        ]
        self.assertEqual(
            [(item["purpose"], item["attempt"]) for item in requested],
            [("main", 1), ("summary", 1), ("main", 2)],
        )

    def test_second_context_rejection_does_not_compact_again(self):
        provider = FakeProvider(
            [
                ModelProviderError(ModelErrorKind.CONTEXT_LENGTH),
                ModelResponse("SUMMARY", None, [], "stop"),
                ModelProviderError(ModelErrorKind.CONTEXT_LENGTH),
            ]
        )
        messages = [
            {"role": "user", "content": "task"},
            *tool_block("old", "old", assistant_text="X" * 10_000),
            *tool_block("latest", "latest"),
        ]
        logger = RecordingEventLogger()

        with self.assertRaises(ModelProviderError):
            agent_loop(
                provider,
                self.workspace,
                messages,
                max_turns=1,
                max_context_tokens=25_000,
                event_logger=logger,
                recovery_policy=RecoveryPolicy(max_retries=0),
            )

        self.assertEqual(len(provider.calls), 3)
        reactive_events = [
            event
            for event in logger.events
            if event["event_type"] == "context_compacted"
            and event["data"]["reason"] == "reactive"
        ]
        self.assertEqual(len(reactive_events), 1)

    def test_transient_retry_budget_survives_reactive_compaction(self):
        provider = FakeProvider(
            [
                ModelProviderError(ModelErrorKind.SERVER_UNAVAILABLE),
                ModelProviderError(ModelErrorKind.CONTEXT_LENGTH),
                ModelResponse("SUMMARY", None, [], "stop"),
                ModelProviderError(ModelErrorKind.CONNECTION),
                ModelResponse("done", None, [], "stop"),
            ]
        )
        messages = [
            {"role": "user", "content": "task"},
            *tool_block("old", "old", assistant_text="X" * 10_000),
            *tool_block("latest", "latest"),
        ]
        logger = RecordingEventLogger()

        answer = agent_loop(
            provider,
            self.workspace,
            messages,
            max_turns=1,
            max_context_tokens=25_000,
            event_logger=logger,
            recovery_policy=RecoveryPolicy(
                max_retries=2,
                base_delay_seconds=0,
                max_delay_seconds=0,
                jitter_ratio=0,
            ),
        )

        self.assertEqual(answer, "done")
        main_attempts = [
            event["data"]["attempt"]
            for event in logger.events
            if event["event_type"] == "model_requested"
            and event["data"]["purpose"] == "main"
        ]
        self.assertEqual(main_attempts, [1, 2, 3, 4])
        transient_retries = [
            event
            for event in logger.events
            if event["event_type"] == "model_retry_scheduled"
            and event["data"].get("recovery") != "reactive_compact"
        ]
        self.assertEqual(
            [event["data"]["retry_number"] for event in transient_retries],
            [1, 2],
        )


if __name__ == "__main__":
    unittest.main()
