import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from tiny_harness.agent.subagent import SubagentExecutor
from tiny_harness.runtime.events import EventType
from tiny_harness.runtime.permissions import DEFAULT_PERMISSION_POLICY
from tiny_harness.runtime.recovery import RecoveryPolicy


class FakeTestRunner:
    def run(self, workspace: Path) -> str:
        return "Exit code: 0"


class RecordingEventLogger:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event_type, data=None) -> None:
        self.events.append(
            {"event_type": event_type.value, "data": dict(data or {})}
        )


class FakeProvider:
    def complete(self, messages, tools):
        raise AssertionError("provider should be called through run_agent")


class SubagentExecutorTest(unittest.TestCase):
    def test_builds_fresh_child_context_and_scopes_events(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        workspace = Path(temporary_directory.name)
        logger = RecordingEventLogger()
        calls = []

        def run_agent(provider, child_workspace, messages, **options):
            calls.append(
                {
                    "provider": provider,
                    "workspace": child_workspace,
                    "messages": messages,
                    "options": options,
                }
            )
            options["event_logger"].emit(EventType.RUN_STARTED, {})
            return "child final"

        provider = FakeProvider()
        test_runner = FakeTestRunner()
        executor = SubagentExecutor(
            run_agent,
            provider,
            workspace,
            max_turns=4,
            permission_policy=DEFAULT_PERMISSION_POLICY,
            permission_prompt=None,
            event_logger=logger,
            max_context_chars=10_000,
            tool_hooks=None,
            recovery_policy=RecoveryPolicy(max_retries=0),
            test_runner=test_runner,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            answer = executor("delegated work", "task-1")

        self.assertEqual(answer, "child final")
        self.assertIs(calls[0]["provider"], provider)
        self.assertEqual(calls[0]["workspace"], workspace)
        self.assertEqual(
            [message["role"] for message in calls[0]["messages"]],
            ["system", "user"],
        )
        self.assertEqual(
            calls[0]["messages"][-1]["content"],
            "delegated work",
        )
        self.assertFalse(calls[0]["options"]["allow_subagent"])
        self.assertIs(calls[0]["options"]["test_runner"], test_runner)
        self.assertNotIn("goal_condition", calls[0]["options"])
        self.assertEqual(
            logger.events[0]["data"],
            {
                "agent_scope": "subagent",
                "parent_tool_call_id": "task-1",
            },
        )


if __name__ == "__main__":
    unittest.main()
