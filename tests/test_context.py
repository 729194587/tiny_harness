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

from tiny_harness.agent.loop import agent_loop
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.runtime.context import (
    CompactionConfig,
    ContextArtifactError,
    ContextCompactor,
    ContextLimitError,
    ContextProtocolError,
    ContextSummaryError,
    context_char_count,
    prepare_context,
)
from tiny_harness.runtime.hooks import HookBlock, ToolHooks
from tiny_harness.runtime.recovery import RecoveryPolicy


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
    def test_character_count_includes_messages_and_tool_schemas(self) -> None:
        messages = [{"role": "user", "content": "你好"}]
        expected = len(
            json.dumps(
                {"messages": messages, "tools": TOOLS},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        self.assertEqual(context_char_count(messages, TOOLS), expected)

    def test_phase_four_helper_still_drops_complete_old_blocks(self) -> None:
        prefix = [{"role": "user", "content": "task"}]
        old = tool_block("old")
        latest = tool_block("latest")
        budget = context_char_count(prefix + latest, TOOLS)

        prepared = prepare_context(prefix + old + latest, TOOLS, budget)

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
            prepare_context(orphan, TOOLS, 10_000)
        with self.assertRaises(ContextProtocolError):
            prepare_context(reordered, TOOLS, 10_000)


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

    def compactor(self, provider=None, *, max_chars=10_000, **config):
        return ContextCompactor(
            self.workspace,
            provider or FakeProvider(),
            TOOLS,
            max_chars,
            config=CompactionConfig(**config),
        )

    def test_under_budget_short_history_is_unchanged_and_writes_no_artifacts(self):
        messages = self.prefix + tool_block("one", "short")
        prepared = self.compactor().prepare(messages, "No todos.")

        self.assertEqual(prepared.messages, messages)
        self.assertFalse(prepared.changed)
        self.assertFalse((self.workspace / ".tinyharness").exists())

    def test_tool_result_budget_persists_large_latest_result_with_preview(self):
        full_output = "A" * 2_000
        messages = self.prefix + tool_block("unsafe/id", full_output)
        compactor = self.compactor(
            max_chars=5_000,
            tool_result_batch_chars=1_000,
            large_result_chars=100,
            result_preview_chars=20,
        )

        prepared = compactor.prepare(messages, "No todos.")

        self.assertEqual(prepared.persisted_results, 1)
        result = prepared.messages[-1]["content"]
        self.assertIn("Full output: .tinyharness/context/tool-results/", result)
        self.assertIn("Preview:\n" + "A" * 20, result)
        relative = next(
            line.removeprefix("Full output: ")
            for line in result.splitlines()
            if line.startswith("Full output: ")
        )
        self.assertEqual(
            (self.workspace / relative).read_text(encoding="utf-8"),
            full_output,
        )
        self.assertEqual(messages[-1]["content"], full_output)

    def test_snip_archives_exact_history_and_keeps_complete_latest_block(self):
        messages = (
            self.prefix
            + tool_block("one")
            + tool_block("two")
            + tool_block("three")
        )
        prepared = self.compactor(max_messages=6).prepare(messages, "No todos.")

        self.assertEqual(prepared.archived_messages, 4)
        self.assertEqual(prepared.messages[-2:], tool_block("three"))
        self.assertEqual(
            prepared.messages[2]["name"],
            "tinyharness_context_archive",
        )
        transcripts = list(
            (self.workspace / ".tinyharness/context/transcripts").glob("*.jsonl")
        )
        self.assertEqual(len(transcripts), 1)
        archived = [
            json.loads(line)
            for line in transcripts[0].read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(archived, messages)

    def test_micro_compact_keeps_latest_three_tool_results(self):
        long_results = [tool_block(str(index), str(index) * 200) for index in range(5)]
        messages = self.prefix + [item for block in long_results for item in block]
        prepared = self.compactor(
            max_chars=20_000,
            keep_recent_results=3,
            micro_result_chars=120,
        ).prepare(messages, "No todos.")

        results = [
            message["content"]
            for message in prepared.messages
            if message["role"] == "tool"
        ]
        self.assertEqual(prepared.shortened_results, 2)
        self.assertTrue(results[0].startswith("[Earlier tool result omitted"))
        self.assertTrue(results[1].startswith("[Earlier tool result omitted"))
        self.assertEqual(results[-3:], ["2" * 200, "3" * 200, "4" * 200])

    def test_current_todo_marker_is_replaced_without_accumulating(self):
        compactor = self.compactor()
        first = compactor.prepare(self.prefix, "[>] FIRST_TODO")
        second = compactor.prepare(first.messages, "[x] UPDATED_TODO")

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
            1_500,
            config=CompactionConfig(max_messages=50),
        )

        prepared = compactor.prepare(messages, "[>] CURRENT_TODO")

        self.assertTrue(prepared.summarized)
        self.assertLessEqual(prepared.after_chars, 1_500)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0]["tools"], [])
        self.assertLessEqual(
            context_char_count(provider.calls[0]["messages"], []),
            1_500,
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
            1_500,
        )

        with self.assertRaises(ContextSummaryError):
            compactor.prepare(messages, "No todos.")

    def test_impossible_mandatory_context_fails_before_summary_api_call(self):
        provider = FakeProvider([ModelResponse("must not run", None, [], "stop")])
        messages = [{"role": "user", "content": "T" * 2_000}]
        compactor = ContextCompactor(
            self.workspace,
            provider,
            TOOLS,
            300,
        )

        with self.assertRaises(ContextLimitError):
            compactor.prepare(messages, "No todos.")
        self.assertEqual(provider.calls, [])

    def test_reactive_compaction_meets_explicit_shrink_margin(self):
        provider = FakeProvider([ModelResponse("REACTIVE_SUMMARY", None, [], "stop")])
        messages = (
            self.prefix
            + tool_block("old", "old", assistant_text="X" * 10_000)
            + tool_block("latest", "LATEST_EVIDENCE")
        )
        failed_chars = context_char_count(messages, TOOLS)

        prepared = self.compactor(
            provider,
            max_chars=100_000,
            reactive_target_ratio=0.75,
        ).reactive_compact(
            messages,
            "[>] CURRENT_TODO",
            failed_request_chars=failed_chars,
        )

        self.assertLessEqual(prepared.after_chars, int(failed_chars * 0.75))
        self.assertLessEqual(
            context_char_count(provider.calls[0]["messages"], []),
            int(failed_chars * 0.75),
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
                failed_request_chars=context_char_count(messages, TOOLS),
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
                self.compactor(
                    max_chars=5_000,
                    tool_result_batch_chars=1_000,
                    large_result_chars=100,
                ).prepare(messages, "No todos.")

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
            max_context_chars=100_000,
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
            max_context_chars=6_000,
            event_logger=logger,
        )

        self.assertEqual(answer, "done")
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(provider.calls[0]["tools"], [])
        self.assertIn("compact", [
            schema["function"]["name"] for schema in provider.calls[1]["tools"]
        ])
        event_names = [event["event_type"] for event in logger.events]
        self.assertLess(
            event_names.index("context_summary_requested"),
            event_names.index("model_requested"),
        )
        self.assertNotIn("ORIGINAL_TASK", json.dumps(logger.events))
        self.assertNotIn("SUMMARY", json.dumps(logger.events))

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
            max_context_chars=6_000,
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
        messages = [{"role": "user", "content": "write then compact"}]

        answer = agent_loop(
            provider,
            self.workspace,
            messages,
            max_context_chars=100_000,
        )

        self.assertEqual(answer, "finished")
        self.assertEqual(
            (self.workspace / "done.txt").read_text(encoding="utf-8"),
            "DONE",
        )
        summary_input = json.dumps(provider.calls[1]["messages"])
        self.assertIn("write-1", summary_input)
        self.assertIn("compact-1", summary_input)
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
                max_context_chars=100_000,
                tool_hooks=hooks,
            )

        self.assertEqual(answer, "continued")
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("compaction blocked", provider.calls[1]["messages"][-1]["content"])

    def test_unconfigured_budget_preserves_phase_seven_tools(self):
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
            max_context_chars=100_000,
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
            compacted["after_chars"],
            int(compacted["before_chars"] * 0.75),
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
                max_context_chars=100_000,
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
            max_context_chars=100_000,
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
