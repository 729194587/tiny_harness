import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tiny_harness.agent.loop import agent_loop
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.runtime.events import (
    NULL_EVENT_LOGGER,
    EventLogError,
    EventType,
    JsonlEventLogger,
)


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
                "model_requested",
                "model_responded",
                "run_finished",
            ],
        )
        self.assertEqual(logger.events[2]["data"]["tool_call_count"], 0)
        self.assertEqual(logger.events[3]["data"], {"turns": 1, "answer_length": 4})

    def test_allowed_tool_records_started_then_finished(self) -> None:
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
            ["tool_started", "tool_finished"],
        )
        self.assertEqual(tool_events[0]["data"]["tool_call_id"], "write-1")
        self.assertEqual(tool_events[1]["data"]["outcome"], "returned")

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
                ("tool_started", "write-1"),
                ("tool_finished", "write-1"),
                ("tool_started", "write-2"),
                ("tool_finished", "write-2"),
            ],
        )

    def test_denied_tool_records_only_tool_denied(self) -> None:
        logger = RecordingEventLogger()
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("bash-1", "bash", '{"command":"cd"}')],
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
            ["tool_denied"],
        )
        self.assertEqual(tool_events[0]["data"]["tool_call_id"], "bash-1")

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
            event for event in logger.events if event["event_type"] == "tool_finished"
        ]
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["data"]["outcome"], "error")

    def test_model_failure_records_run_failed(self) -> None:
        logger = RecordingEventLogger()
        provider = FakeProvider([RuntimeError("provider failed")])

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            agent_loop(provider, self.workspace, [], event_logger=logger)

        self.assertEqual(logger.events[-1]["event_type"], "run_failed")
        self.assertEqual(
            logger.events[-1]["data"],
            {"turn": 1, "error_type": "RuntimeError"},
        )

    def test_max_turns_records_run_failed(self) -> None:
        logger = RecordingEventLogger()
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("write-1", "write_file", '{"path":"a.txt","content":"A"}')],
                    "tool_calls",
                )
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "Maximum model turns reached"):
            agent_loop(
                provider,
                self.workspace,
                [],
                max_turns=1,
                event_logger=logger,
            )

        self.assertEqual(logger.events[-1]["event_type"], "run_failed")

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
                        )
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
            "PRIVATE_FINAL_ANSWER",
        ):
            self.assertNotIn(secret, log_text)


if __name__ == "__main__":
    unittest.main()
