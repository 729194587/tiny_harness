import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.agent.session import AgentSession


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
            max_context_chars=100_000,
        )

        self.assertEqual(session.submit("FIRST_TASK"), "FIRST_ANSWER")
        self.assertEqual(session.submit("SECOND_TASK"), "SECOND_ANSWER")

        second_request = provider.calls[1]["messages"]
        self.assertEqual(
            [(item["role"], item.get("content")) for item in second_request],
            [
                ("system", "system"),
                ("user", "FIRST_TASK"),
                ("assistant", "FIRST_ANSWER"),
                ("user", "SECOND_TASK"),
            ],
        )
        self.assertEqual(session.messages[-1]["content"], "SECOND_ANSWER")

    def test_each_submit_refreshes_the_workspace_skill_catalog(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse("FIRST_ANSWER", None, [], "stop"),
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall(
                            "skill-1",
                            "load_skill",
                            '{"name":"review"}',
                        )
                    ],
                    "tool_calls",
                ),
                ModelResponse("SECOND_ANSWER", None, [], "stop"),
            ]
        )
        session = AgentSession(provider, self.workspace, "system")

        self.assertEqual(session.submit("first"), "FIRST_ANSWER")
        skill_path = self.workspace / "skills" / "review" / "SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text(
            "---\n"
            "name: review\n"
            "description: Review the current change\n"
            "---\n\n"
            "SESSION_REFRESH_SKILL_BODY\n",
            encoding="utf-8",
        )

        self.assertEqual(session.submit("second"), "SECOND_ANSWER")

        first_request = provider.calls[0]
        self.assertNotIn(
            "load_skill",
            [tool["function"]["name"] for tool in first_request["tools"]],
        )
        second_request = provider.calls[1]
        self.assertIn(
            "load_skill",
            [tool["function"]["name"] for tool in second_request["tools"]],
        )
        catalog = next(
            message
            for message in second_request["messages"]
            if message.get("name") == "tinyharness_skill_catalog"
        )
        self.assertIn("Review the current change", catalog["content"])
        self.assertNotIn("SESSION_REFRESH_SKILL_BODY", catalog["content"])
        loaded_result = next(
            message
            for message in provider.calls[2]["messages"]
            if message.get("tool_call_id") == "skill-1"
        )
        self.assertIn("SESSION_REFRESH_SKILL_BODY", loaded_result["content"])
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
                {"role": "user", "name": "tinyharness_goal_state", "content": "old"},
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
        self.assertNotIn("tinyharness_goal_state", names)
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

    def test_goal_is_allowed_on_first_submit(self) -> None:
        provider = FakeProvider([])
        session = AgentSession(provider, self.workspace, "system")

        with patch(
            "tiny_harness.agent.session.create_run_context",
            return_value=object(),
        ) as create_context:
            with patch(
                "tiny_harness.agent.session.agent_loop",
                return_value="done",
            ):
                answer = session.submit(
                    "task", goal_condition="tests pass"
                )

        self.assertEqual(answer, "done")
        self.assertEqual(
            create_context.call_args.kwargs["goal_condition"],
            "tests pass",
        )

    def test_goal_is_rejected_after_a_previous_user_turn(self) -> None:
        provider = FakeProvider([ModelResponse("first", None, [], "stop")])
        session = AgentSession(provider, self.workspace, "system")
        session.submit("first task")
        before = copy.deepcopy(session.messages)

        with self.assertRaisesRegex(
            ValueError,
            "only supported on the first submit",
        ):
            session.submit("second task", goal_condition="new tests pass")

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(session.messages, before)

    def test_clear_makes_the_session_fresh_for_a_goal(self) -> None:
        provider = FakeProvider([ModelResponse("first", None, [], "stop")])
        session = AgentSession(provider, self.workspace, "system")
        session.submit("first task")
        session.clear()

        with patch(
            "tiny_harness.agent.session.create_run_context",
            return_value=object(),
        ) as create_context:
            with patch(
                "tiny_harness.agent.session.agent_loop",
                return_value="done",
            ):
                answer = session.submit(
                    "fresh task", goal_condition="tests pass"
                )

        self.assertEqual(answer, "done")
        self.assertEqual(
            create_context.call_args.kwargs["goal_condition"],
            "tests pass",
        )


if __name__ == "__main__":
    unittest.main()
