import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tiny_harness.agent.loop import run_agent as agent_loop
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.runtime.events import (
    NULL_EVENT_LOGGER,
    EventLogError,
    EventType,
    JsonlEventLogger,
    hash_json,
    hash_text,
)
from tiny_harness.runtime.hooks import HookBlock, HookExecutionError, ToolHooks


FIXED_TIME = datetime(2026, 8, 17, 12, 30, tzinfo=timezone.utc)


class RecordingEventLogger:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event_type, data=None) -> None:
        self.events.append(
            {
                "event_type": event_type.value,
                "data": dict(data or {}),
            }
        )


class FailingEventLogger:
    def emit(self, event_type, data=None) -> None:
        raise EventLogError("log unavailable")


class FakeProvider:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def complete(self, messages, tools):
        self.call_count += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class JsonlEventLoggerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_writes_ordered_json_lines(self) -> None:
        log_path = self.root / "logs" / "events.jsonl"
        logger = JsonlEventLogger(
            log_path,
            run_id="run-1",
            clock=lambda: FIXED_TIME,
        )

        logger.emit(EventType.RUN_STARTED, {"max_turns": 5})
        logger.emit(EventType.RUN_FINISHED, {"turns": 1})

        events = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([event["sequence"] for event in events], [1, 2])
        self.assertEqual({event["run_id"] for event in events}, {"run-1"})
        self.assertEqual(
            [event["event_type"] for event in events],
            ["run_started", "run_finished"],
        )
        self.assertEqual(events[0]["timestamp"], FIXED_TIME.isoformat())
        self.assertEqual(events[0]["data"], {"max_turns": 5})

    def test_appends_runs_and_restarts_sequence_per_run(self) -> None:
        log_path = self.root / "events.jsonl"
        first = JsonlEventLogger(log_path, run_id="run-1", clock=lambda: FIXED_TIME)
        second = JsonlEventLogger(log_path, run_id="run-2", clock=lambda: FIXED_TIME)

        first.emit(EventType.RUN_STARTED)
        second.emit(EventType.RUN_STARTED)

        events = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            [(event["run_id"], event["sequence"]) for event in events],
            [("run-1", 1), ("run-2", 1)],
        )

    def test_non_serializable_data_raises_event_log_error(self) -> None:
        logger = JsonlEventLogger(
            self.root / "events.jsonl",
            clock=lambda: FIXED_TIME,
        )

        with self.assertRaisesRegex(EventLogError, "Failed to write event log"):
            logger.emit(EventType.RUN_STARTED, {"bad": object()})

    def test_unwritable_target_shape_raises_event_log_error(self) -> None:
        directory = self.root / "events.jsonl"
        directory.mkdir()
        logger = JsonlEventLogger(directory, clock=lambda: FIXED_TIME)

        with self.assertRaisesRegex(EventLogError, "Failed to write event log"):
            logger.emit(EventType.RUN_STARTED)

    def test_null_logger_is_a_no_op(self) -> None:
        NULL_EVENT_LOGGER.emit(EventType.RUN_STARTED, {"anything": object()})


class EventLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_final_answer_records_run_and_model_lifecycle(self) -> None:
        logger = RecordingEventLogger()
        provider = FakeProvider([ModelResponse("done", None, [], "stop")])

        answer = agent_loop(
            provider,
            self.workspace,
            [{"role": "user", "content": "task"}],
            event_logger=logger,
        )

        self.assertEqual(answer, "done")
        self.assertEqual(
            [event["event_type"] for event in logger.events],
            [
                "run_started",
                "context_prepared",
                "model_requested",
                "model_responded",
                "run_finished",
            ],
        )
        self.assertEqual(logger.events[3]["data"]["tool_call_count"], 0)
        self.assertEqual(
            logger.events[4]["data"],
            {"turns": 1, "answer_length": 4},
        )

    def test_allowed_tool_records_called_started_then_result_with_identity(self) -> None:
        logger = RecordingEventLogger()
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("write-1", "write_file", '{"path":"a.txt","content":"A"}')],
                    "tool_calls",
                ),
                ModelResponse("done", None, [], "stop"),
            ]
        )

        agent_loop(provider, self.workspace, [], event_logger=logger)

        tool_events = [
            event for event in logger.events if event["event_type"].startswith("tool_")
        ]
        self.assertEqual(
            [event["event_type"] for event in tool_events],
            ["tool_called", "tool_started", "tool_result"],
        )
        self.assertTrue(all(event["data"]["turn"] == 1 for event in tool_events))
        self.assertTrue(all(event["data"]["tool_call_id"] == "write-1" for event in tool_events))
        self.assertTrue(all(event["data"]["tool_name"] == "write_file" for event in tool_events))
        self.assertEqual(tool_events[0]["data"]["path"], "a.txt")
        self.assertEqual(
            tool_events[0]["data"]["arguments_hash"],
            hash_json({"path": "a.txt", "content": "A"}),
        )
        self.assertEqual(tool_events[2]["data"]["outcome"], "returned")
        self.assertEqual(
            tool_events[2]["data"]["content_hash"],
            hash_text("Wrote 1 bytes to a.txt"),
        )

    def test_multiple_tool_calls_record_events_in_execution_order(self) -> None:
        logger = RecordingEventLogger()
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall(
                            "write-1",
                            "write_file",
                            '{"path":"a.txt","content":"A"}',
                        ),
                        ToolCall(
                            "write-2",
                            "write_file",
                            '{"path":"b.txt","content":"B"}',
                        ),
                    ],
                    "tool_calls",
                ),
                ModelResponse("done", None, [], "stop"),
            ]
        )

        agent_loop(provider, self.workspace, [], event_logger=logger)

        tool_events = [
            event for event in logger.events if event["event_type"].startswith("tool_")
        ]
        self.assertEqual(
            [
                (event["event_type"], event["data"]["tool_call_id"])
                for event in tool_events
            ],
            [
                ("tool_called", "write-1"),
                ("tool_called", "write-2"),
                ("tool_started", "write-1"),
                ("tool_result", "write-1"),
                ("tool_started", "write-2"),
                ("tool_result", "write-2"),
            ],
        )

    def test_denied_tool_records_called_and_result_without_started(self) -> None:
        logger = RecordingEventLogger()
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall(
                            "bash-1",
                            "bash",
                            '{"command":"python --version"}',
                        )
                    ],
                    "tool_calls",
                ),
                ModelResponse("denied", None, [], "stop"),
            ]
        )

        agent_loop(
            provider,
            self.workspace,
            [],
            permission_prompt=lambda *_: False,
            event_logger=logger,
        )

        tool_events = [
            event for event in logger.events if event["event_type"].startswith("tool_")
        ]
        self.assertEqual(
            [event["event_type"] for event in tool_events],
            ["tool_called", "tool_denied", "tool_result"],
        )
        self.assertEqual(tool_events[0]["data"]["tool_call_id"], "bash-1")
        self.assertEqual(tool_events[-1]["data"]["outcome"], "permission_denied")

    def test_hook_block_records_result_without_started(self) -> None:
        logger = RecordingEventLogger()
        hooks = ToolHooks()
        hooks.register_pre(lambda _context: HookBlock("PRIVATE_BLOCK_REASON"))
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("write-1", "write_file", '{"path":"blocked.txt","content":"bad"}')],
                    "tool_calls",
                ),
                ModelResponse("handled", None, [], "stop"),
            ]
        )

        agent_loop(
            provider,
            self.workspace,
            [],
            event_logger=logger,
            tool_hooks=hooks,
        )

        tool_events = [
            event for event in logger.events if event["event_type"].startswith("tool_")
        ]
        self.assertEqual(
            [event["event_type"] for event in tool_events],
            ["tool_called", "tool_hook_blocked", "tool_result"],
        )
        self.assertEqual(tool_events[-1]["data"]["outcome"], "hook_blocked")
        self.assertNotIn("PRIVATE_BLOCK_REASON", json.dumps(tool_events))

    def test_all_model_tool_calls_are_logged_before_a_fatal_hook_failure(self) -> None:
        logger = RecordingEventLogger()
        hooks = ToolHooks()

        def fail_first(context):
            if context.tool_call_id == "write-1":
                raise RuntimeError("fatal hook")

        hooks.register_pre(fail_first)
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall("write-1", "write_file", '{"path":"a.txt","content":"A"}'),
                        ToolCall("write-2", "write_file", '{"path":"b.txt","content":"B"}'),
                    ],
                    "tool_calls",
                )
            ]
        )

        with self.assertRaises(HookExecutionError):
            agent_loop(
                provider,
                self.workspace,
                [],
                event_logger=logger,
                tool_hooks=hooks,
            )

        called_ids = [
            event["data"]["tool_call_id"]
            for event in logger.events
            if event["event_type"] == "tool_called"
        ]
        self.assertEqual(called_ids, ["write-1", "write-2"])
        self.assertFalse((self.workspace / "b.txt").exists())

    def test_tool_exception_records_error_outcome(self) -> None:
        logger = RecordingEventLogger()
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("read-1", "read_file", '{"path":"missing.txt"}')],
                    "tool_calls",
                ),
                ModelResponse("handled", None, [], "stop"),
            ]
        )

        agent_loop(provider, self.workspace, [], event_logger=logger)

        finished = [
            event for event in logger.events if event["event_type"] == "tool_result"
        ]
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["data"]["outcome"], "error")

    def test_read_file_called_event_records_path_without_content(self) -> None:
        (self.workspace / "source.txt").write_text(
            "PRIVATE_SOURCE_BODY",
            encoding="utf-8",
        )
        logger = RecordingEventLogger()
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("read-1", "read_file", '{"path":"source.txt"}')],
                    "tool_calls",
                ),
                ModelResponse("done", None, [], "stop"),
            ]
        )

        agent_loop(provider, self.workspace, [], event_logger=logger)

        called = next(
            event["data"]
            for event in logger.events
            if event["event_type"] == "tool_called"
        )
        self.assertEqual(called["path"], "source.txt")
        self.assertNotIn("PRIVATE_SOURCE_BODY", json.dumps(logger.events))

    def test_model_failure_records_run_failed(self) -> None:
        logger = RecordingEventLogger()
        provider = FakeProvider([RuntimeError("provider failed")])

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            agent_loop(provider, self.workspace, [], event_logger=logger)

        self.assertEqual(logger.events[-1]["event_type"], "run_failed")
        self.assertEqual(
            logger.events[-1]["data"],
            {"turn": 1, "error_type": "RuntimeError", "last_finish_reason": None},
        )

    def test_finalization_at_max_turns_records_run_finished(self) -> None:
        logger = RecordingEventLogger()
        provider = FakeProvider(
            [
                ModelResponse("best available answer", None, [], "stop")
            ]
        )

        answer = agent_loop(
            provider,
            self.workspace,
            [],
            max_turns=1,
            event_logger=logger,
        )

        self.assertEqual(answer, "best available answer")
        self.assertEqual(logger.events[-1]["event_type"], "run_finished")
        self.assertNotIn(
            "run_failed", [event["event_type"] for event in logger.events]
        )

    def test_finalization_tool_call_produces_no_tool_events_and_finishes(self) -> None:
        logger = RecordingEventLogger()
        provider = FakeProvider(
            [
                ModelResponse(
                    "best available answer",
                    None,
                    [ToolCall("forbidden-1", "list_files", "{}")],
                    "tool_calls",
                ),
                ModelResponse("best available answer", None, [], "stop"),
            ]
        )

        answer = agent_loop(
            provider,
            self.workspace,
            [],
            max_turns=1,
            event_logger=logger,
        )

        self.assertEqual(answer, "best available answer")
        event_names = [event["event_type"] for event in logger.events]
        self.assertNotIn("tool_called", event_names)
        self.assertNotIn("tool_started", event_names)
        self.assertNotIn("tool_result", event_names)
        self.assertEqual(event_names[-1], "run_finished")

    def test_context_and_turn_budget_are_recorded_for_each_main_request(self) -> None:
        logger = RecordingEventLogger()
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("list-1", "list_files", "{}")],
                    "tool_calls",
                ),
                ModelResponse("done", None, [], "stop"),
            ]
        )

        agent_loop(
            provider,
            self.workspace,
            [{"role": "user", "content": "inspect"}],
            max_turns=2,
            max_context_tokens=25_000,
            event_logger=logger,
        )

        prepared = [
            event["data"]
            for event in logger.events
            if event["event_type"] == "context_prepared"
        ]
        self.assertEqual([item["turn"] for item in prepared], [1, 2])
        self.assertEqual(
            [item.get("finalization", False) for item in prepared],
            [False, True],
        )
        self.assertTrue(all(item["hard_limit"] == 25_000 for item in prepared))
        self.assertTrue(all(item["soft_limit"] == 20_000 for item in prepared))
        self.assertTrue(all(item["pressure"] < 1 for item in prepared))
        self.assertFalse(
            any(event["event_type"] == "context_compacted" for event in logger.events)
        )
        requested = [
            event["data"]
            for event in logger.events
            if event["event_type"] == "model_requested"
            and event["data"]["purpose"] == "main"
        ]
        self.assertEqual([item["remaining_turns"] for item in requested], [1, 0])
        self.assertEqual([item["max_turns"] for item in requested], [2, 2])
        self.assertEqual(
            [item.get("finalization", False) for item in requested],
            [False, True],
        )
        self.assertEqual(
            [item["context_tokens"] for item in requested],
            [item["context_tokens"] for item in prepared],
        )
        responded = [
            event["data"]
            for event in logger.events
            if event["event_type"] == "model_responded"
            and event["data"]["purpose"] == "main"
        ]
        self.assertEqual(
            [item.get("finalization", False) for item in responded],
            [False, True],
        )

    def test_equivalent_arguments_and_results_have_stable_hashes(self) -> None:
        logger = RecordingEventLogger()
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall("list-1", "list_files", '{"path":".","unexpected":1}'),
                        ToolCall("list-2", "list_files", '{"unexpected":1,"path":"."}'),
                    ],
                    "tool_calls",
                ),
                ModelResponse("done", None, [], "stop"),
            ]
        )

        agent_loop(provider, self.workspace, [], event_logger=logger)

        called = [e["data"] for e in logger.events if e["event_type"] == "tool_called"]
        results = [e["data"] for e in logger.events if e["event_type"] == "tool_result"]
        self.assertEqual(called[0]["arguments_hash"], called[1]["arguments_hash"])
        self.assertEqual(results[0]["content_hash"], results[1]["content_hash"])

    def test_task_metadata_does_not_expose_prompt(self) -> None:
        logger = RecordingEventLogger()
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("task-1", "task", '{"prompt":"PRIVATE_TASK_PROMPT"}')],
                    "tool_calls",
                ),
                ModelResponse("child done", None, [], "stop"),
                ModelResponse("parent done", None, [], "stop"),
            ]
        )

        agent_loop(provider, self.workspace, [], event_logger=logger)

        task_called = next(
            event["data"]
            for event in logger.events
            if event["event_type"] == "tool_called"
            and event["data"]["tool_name"] == "task"
        )
        self.assertEqual(task_called["prompt_length"], len("PRIVATE_TASK_PROMPT"))
        self.assertEqual(task_called["prompt_hash"], hash_text("PRIVATE_TASK_PROMPT"))
        self.assertNotIn("PRIVATE_TASK_PROMPT", json.dumps(logger.events))

    def test_event_log_failure_stops_before_model_call(self) -> None:
        provider = FakeProvider([ModelResponse("done", None, [], "stop")])

        with self.assertRaisesRegex(EventLogError, "log unavailable"):
            agent_loop(provider, self.workspace, [], event_logger=FailingEventLogger())

        self.assertEqual(provider.call_count, 0)

    def test_jsonl_does_not_store_payloads(self) -> None:
        log_path = self.workspace / "events.jsonl"
        logger = JsonlEventLogger(
            log_path,
            run_id="run-secret-test",
            clock=lambda: FIXED_TIME,
        )
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    "PRIVATE_REASONING",
                    [
                        ToolCall(
                            "write-1",
                            "write_file",
                            '{"path":"secret.txt","content":"TOP_SECRET"}',
                        ),
                        ToolCall(
                            "edit-1",
                            "edit_file",
                            '{"path":"secret.txt","old_text":"TOP_SECRET",'
                            '"new_text":"REPLACED_SECRET"}',
                        ),
                    ],
                    "tool_calls",
                ),
                ModelResponse("PRIVATE_FINAL_ANSWER", None, [], "stop"),
            ]
        )

        agent_loop(
            provider,
            self.workspace,
            [{"role": "user", "content": "PRIVATE_PROMPT"}],
            event_logger=logger,
        )

        log_text = log_path.read_text(encoding="utf-8")
        for secret in (
            "PRIVATE_PROMPT",
            "PRIVATE_REASONING",
            "TOP_SECRET",
            "REPLACED_SECRET",
            "PRIVATE_FINAL_ANSWER",
        ):
            self.assertNotIn(secret, log_text)


if __name__ == "__main__":
    unittest.main()
