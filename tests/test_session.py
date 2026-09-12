import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tiny_harness.agent.messages import ModelResponse
from tiny_harness.agent.session import AgentSession
from tiny_harness.agent.turn import TOOL_USE_EFFICIENCY_GUIDANCE


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
        if not self.responses:
            raise AssertionError("FakeProvider has no response left")
        return self.responses.pop(0)


class RecordingEventLogger:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event_type, data=None) -> None:
        self.events.append((event_type.value, dict(data or {})))


class AgentSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_submit_reuses_successful_conversation_history(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse("FIRST_ANSWER", None, [], "stop"),
                ModelResponse("SECOND_ANSWER", None, [], "stop"),
            ]
        )
        session = AgentSession(
            provider,
            self.workspace,
            "system",
            max_context_tokens=25_000,
        )

        self.assertEqual(session.submit("FIRST_TASK"), "FIRST_ANSWER")
        self.assertEqual(session.submit("SECOND_TASK"), "SECOND_ANSWER")

        second_request = [
            message
            for message in provider.calls[1]["messages"]
            if message.get("name") != "tinyharness_skill_catalog"
            and not str(message.get("content", "")).startswith(
                "TinyHarness runtime state:"
            )
        ]
        self.assertEqual(
            [(item["role"], item.get("content")) for item in second_request],
            [
                ("system", "system"),
                ("system", TOOL_USE_EFFICIENCY_GUIDANCE),
                ("user", "FIRST_TASK"),
                ("assistant", "FIRST_ANSWER"),
                ("user", "SECOND_TASK"),
            ],
        )
        self.assertFalse(
            any(
                str(message.get("content", "")).startswith(
                    "TinyHarness runtime state:"
                )
                for message in session.messages
            )
        )
        self.assertEqual(session.messages[-1]["content"], "SECOND_ANSWER")
        self.assertNotIn(
            {"role": "system", "content": TOOL_USE_EFFICIENCY_GUIDANCE},
            session.messages,
        )

    def test_session_freezes_skill_catalog_and_next_session_refreshes_it(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse("FIRST_ANSWER", None, [], "stop"),
                ModelResponse("SECOND_ANSWER", None, [], "stop"),
            ]
        )
        session = AgentSession(provider, self.workspace, "system")

        skill_path = (
            self.workspace / ".tinyharness" / "skills" / "debug" / "SKILL.md"
        )
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text(
            "---\n"
            "name: debug\n"
            "description: Debug the current change\n"
            "---\n\n"
            "SESSION_SNAPSHOT_SKILL_BODY\n",
            encoding="utf-8",
        )

        self.assertEqual(session.submit("first"), "FIRST_ANSWER")
        self.assertEqual(session.submit("second"), "SECOND_ANSWER")

        for request in provider.calls:
            catalog = next(
                message
                for message in request["messages"]
                if message.get("name") == "tinyharness_skill_catalog"
            )
            self.assertNotIn("Debug the current change", catalog["content"])
            self.assertNotIn("SESSION_SNAPSHOT_SKILL_BODY", catalog["content"])

        next_provider = FakeProvider([ModelResponse("THIRD_ANSWER", None, [], "stop")])
        next_session = AgentSession(next_provider, self.workspace, "system")
        self.assertEqual(next_session.submit("third"), "THIRD_ANSWER")
        next_catalog = next(
            message
            for message in next_provider.calls[0]["messages"]
            if message.get("name") == "tinyharness_skill_catalog"
        )
        self.assertIn("Debug the current change", next_catalog["content"])
        self.assertNotIn("SESSION_SNAPSHOT_SKILL_BODY", next_catalog["content"])
        self.assertFalse(
            any(
                message.get("name") == "tinyharness_skill_catalog"
                for message in session.messages
            )
        )

    def test_submit_removes_run_markers_but_keeps_context_markers(self) -> None:
        provider = FakeProvider([ModelResponse("done", None, [], "stop")])
        session = AgentSession(provider, self.workspace, "system")
        session.messages.extend(
            [
                {"role": "user", "name": "tinyharness_todo_state", "content": "old"},
                {
                    "role": "user",
                    "name": "tinyharness_context_summary",
                    "content": "summary",
                },
            ]
        )

        session.submit("next")

        names = [message.get("name") for message in provider.calls[0]["messages"]]
        self.assertNotIn("tinyharness_todo_state", names)
        self.assertIn("tinyharness_context_summary", names)

    def test_clear_keeps_only_a_fresh_system_message(self) -> None:
        provider = FakeProvider([ModelResponse("done", None, [], "stop")])
        session = AgentSession(provider, self.workspace, "system")
        session.submit("task")

        session.clear()

        self.assertEqual(
            session.messages,
            [{"role": "system", "content": "system"}],
        )

    def test_each_submit_requests_a_new_event_logger(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse("one", None, [], "stop"),
                ModelResponse("two", None, [], "stop"),
            ]
        )
        loggers = []

        def make_logger():
            logger = RecordingEventLogger()
            loggers.append(logger)
            return logger

        session = AgentSession(
            provider,
            self.workspace,
            "system",
            event_logger_factory=make_logger,
        )

        session.submit("one")
        session.submit("two")

        self.assertEqual(len(loggers), 2)
        self.assertEqual(loggers[0].events[0][0], "run_started")
        self.assertEqual(loggers[1].events[0][0], "run_started")

    def test_empty_task_is_rejected_before_model_call(self) -> None:
        provider = FakeProvider([])
        session = AgentSession(provider, self.workspace, "system")

        with self.assertRaisesRegex(ValueError, "task cannot be empty"):
            session.submit("   ")

        self.assertEqual(provider.calls, [])


if __name__ == "__main__":
    unittest.main()
