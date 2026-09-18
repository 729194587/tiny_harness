import tempfile
import unittest
from pathlib import Path
from dataclasses import fields
from unittest.mock import Mock, patch

from tiny_harness.agent.context import RunConfig, build_run_context
from tiny_harness.context.token_meter import CalibratedTokenMeter
from tiny_harness.agent.subagent import SubagentExecutor
from tiny_harness.runtime.events import EventType, EventLogError
from tiny_harness.runtime.tool_trace import ToolTraceConfig
from tiny_harness.runtime.permissions import DEFAULT_PERMISSION_POLICY
from tiny_harness.runtime.recovery import RecoveryPolicy
from tiny_harness.runtime.skills import discover_skills


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

        def run_agent(messages, context, active_request):
            calls.append(
                {
                    "provider": context.provider,
                    "workspace": context.workspace,
                    "messages": messages,
                    "options": vars(context),
                }
            )
            context.event_logger.emit(EventType.RUN_STARTED, {})
            return "child final"

        provider = FakeProvider()
        test_runner = FakeTestRunner()
        skill_catalog = discover_skills(workspace, sources=())
        executor = SubagentExecutor(RunConfig(
            provider,
            workspace,
            max_turns=4,
            subagent_max_turns=4,
            permission_policy=DEFAULT_PERMISSION_POLICY,
            permission_prompt=None,
            event_logger=logger,
            max_context_tokens=25_000,
            tool_hooks=None,
            recovery_policy=RecoveryPolicy(max_retries=0),
            test_runner=test_runner,
            skill_catalog=skill_catalog,
        ))

        with patch("tiny_harness.agent.loop.agent_loop", side_effect=run_agent):
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
        self.assertIs(calls[0]["options"]["skill_catalog"], skill_catalog)
        self.assertEqual(
            logger.events[0]["data"],
            {
                "agent_scope": "subagent",
                "parent_tool_call_id": "task-1",
            },
        )

    def test_configuration_inheritance_and_runtime_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(
                FakeProvider(), Path(directory),
                max_turns=8, subagent_max_turns=4,
                permission_policy=Mock(), permission_prompt=Mock(), tool_hooks=Mock(),
                shell_runner=Mock(), test_runner=FakeTestRunner(),
                memory_enabled=True, token_meter=CalibratedTokenMeter(),
                max_context_tokens=25_000,
                working_context_trigger_tokens=18_000,
                working_context_target_tokens=12_000,
                event_logger=RecordingEventLogger(),
                tool_trace=ToolTraceConfig(enabled=True, result_preview_chars=37),
            )
            parent = build_run_context(config)
            children = []
            configs = []

            def build(child_config):
                configs.append(child_config)
                child = build_run_context(child_config)
                children.append(child)
                return child

            with patch("tiny_harness.agent.context.build_run_context", side_effect=build), patch(
                "tiny_harness.agent.loop.agent_loop", return_value=""
            ) as loop:
                self.assertEqual(parent.subagent_runner("one", "task-1"), "(no summary)")
                parent.subagent_runner("two", "task-2")
            self.assertIsNot(loop.call_args_list[0].args[0], loop.call_args_list[1].args[0])
            overrides = {"max_turns", "allow_subagent", "memory_extraction_enabled", "event_logger", "token_meter"}
            for child_config in configs:
                for field in fields(RunConfig):
                    if field.name not in overrides:
                        self.assertIs(getattr(child_config, field.name),
                                      getattr(parent.subagent_runner.config, field.name))
                self.assertFalse(child_config.memory_extraction_enabled)
            for child in children:
                self.assertEqual(child.max_turns, 4)
                self.assertFalse(child.allow_subagent)
                self.assertIsNone(child.subagent_runner)
                self.assertIsNone(child.final_answer_hook)
                self.assertEqual(child.current_turn, 0)
                self.assertEqual(child.compaction_config, parent.compaction_config)
                for name in ("provider", "permission_policy", "permission_prompt", "tool_hooks",
                             "recovery_policy", "skill_catalog", "shell_runner", "test_runner", "tool_trace"):
                    self.assertIs(getattr(child, name), getattr(parent, name))
                for other in [parent] + [c for c in children if c is not child]:
                    for name in ("tool_registry", "todo_manager", "recovery_executor", "memory",
                                 "permission_rejections", "compactor", "token_meter"):
                        self.assertIsNot(getattr(child, name), getattr(other, name))
                self.assertIs(child.token_meter.heuristic, parent.token_meter.heuristic)
            self.assertIs(parent.token_meter, config.token_meter)
            self.assertIsNotNone(parent.final_answer_hook)

    def test_failure_events_preserve_event_log_error_propagation(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = RecordingEventLogger()
            executor = SubagentExecutor(RunConfig(FakeProvider(), Path(directory), event_logger=logger))
            with patch("tiny_harness.agent.loop.agent_loop", side_effect=ValueError("private")):
                with self.assertRaises(ValueError):
                    executor("task", "failed")
            self.assertEqual(logger.events[-1], {
                "event_type": "subagent_failed",
                "data": {"agent_scope": "subagent", "parent_tool_call_id": "failed", "error_type": "ValueError"},
            })
            with patch("tiny_harness.agent.loop.agent_loop", side_effect=EventLogError("unavailable")):
                with self.assertRaises(EventLogError):
                    executor("task", "log-failed")
            self.assertEqual(logger.events[-1]["event_type"], "subagent_started")


if __name__ == "__main__":
    unittest.main()
