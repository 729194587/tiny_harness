import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from tiny_harness.agent.loop import run_agent as agent_loop
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.runtime.events import EventLogError
from tiny_harness.runtime.hooks import HookBlock, ToolHooks
from tiny_harness.runtime.recovery import RecoveryPolicy


class ScriptedProvider:
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
            raise AssertionError("ScriptedProvider has no response left")
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


class RecordingTestRunner:
    def __init__(self) -> None:
        self.workspaces = []

    def run(self, workspace: Path) -> str:
        self.workspaces.append(workspace)
        return "Exit code: 0\nOK"


class ChildFailingEventLogger(RecordingEventLogger):
    def emit(self, event_type, data=None) -> None:
        event_data = dict(data or {})
        if event_data.get("agent_scope") == "subagent":
            raise EventLogError("child log unavailable")
        super().emit(event_type, event_data)


def tool_names(call) -> list[str]:
    return [schema["function"]["name"] for schema in call["tools"]]


class SubagentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_child_uses_independent_retry_state_with_scoped_events(self) -> None:
        provider = ScriptedProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("task-retry", "task", '{"prompt":"child work"}')],
                    "tool_calls",
                ),
                ModelProviderError(ModelErrorKind.SERVER_UNAVAILABLE),
                ModelResponse("child done", None, [], "stop"),
                ModelResponse("parent done", None, [], "stop"),
            ]
        )
        logger = RecordingEventLogger()

        with contextlib.redirect_stdout(io.StringIO()):
            answer = agent_loop(
                provider,
                self.workspace,
                [],
                event_logger=logger,
                recovery_policy=RecoveryPolicy(
                    max_retries=1,
                    base_delay_seconds=0,
                    max_delay_seconds=0,
                    jitter_ratio=0,
                ),
            )

        self.assertEqual(answer, "parent done")
        child_requests = [
            event["data"]
            for event in logger.events
            if event["event_type"] == "model_requested"
            and event["data"].get("agent_scope") == "subagent"
        ]
        self.assertEqual([event["attempt"] for event in child_requests], [1, 2])
        self.assertTrue(
            all(
                event["parent_tool_call_id"] == "task-retry"
                for event in child_requests
            )
        )
        parent_requests = [
            event["data"]
            for event in logger.events
            if event["event_type"] == "model_requested"
            and "agent_scope" not in event["data"]
        ]
        self.assertEqual([event["attempt"] for event in parent_requests], [1, 1])

    def test_child_context_is_fresh_and_parent_gets_only_final_text(self) -> None:
        provider = ScriptedProvider(
            [
                ModelResponse(
                    None,
                    "PARENT_PRIVATE_REASONING",
                    [
                        ToolCall(
                            "task-1",
                            "task",
                            '{"prompt":"DELEGATED_PROMPT"}',
                        )
                    ],
                    "tool_calls",
                ),
                ModelResponse("CHILD_FINAL_ONLY", None, [], "stop"),
                ModelResponse("parent final", None, [], "stop"),
            ]
        )
        messages = [
            {"role": "system", "content": "PARENT_SECRET_SYSTEM"},
            {"role": "user", "content": "PARENT_SECRET_TASK"},
        ]
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            answer = agent_loop(provider, self.workspace, messages)

        self.assertEqual(answer, "parent final")
        self.assertEqual(len(provider.calls), 3)
        child_request = provider.calls[1]
        self.assertEqual(
            [message["role"] for message in child_request["messages"]],
            ["system", "user"],
        )
        self.assertEqual(
            child_request["messages"][1],
            {"role": "user", "content": "DELEGATED_PROMPT"},
        )
        child_json = json.dumps(child_request["messages"])
        self.assertNotIn("PARENT_SECRET_SYSTEM", child_json)
        self.assertNotIn("PARENT_SECRET_TASK", child_json)
        self.assertNotIn("PARENT_PRIVATE_REASONING", child_json)
        self.assertNotIn("task", tool_names(child_request))

        parent_second_request = provider.calls[2]["messages"]
        self.assertEqual(parent_second_request[-1]["role"], "tool")
        self.assertEqual(parent_second_request[-1]["tool_call_id"], "task-1")
        self.assertEqual(parent_second_request[-1]["content"], "CHILD_FINAL_ONLY")
        self.assertNotIn(
            "You are a coding subagent",
            json.dumps(parent_second_request),
        )
        self.assertIn("[子 Agent 已启动]", stdout.getvalue())
        self.assertIn("[子 Agent 已完成]", stdout.getvalue())

    def test_child_inherits_and_can_call_run_tests_capability(self) -> None:
        runner = RecordingTestRunner()
        provider = ScriptedProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("task-1", "task", '{"prompt":"verify"}')],
                    "tool_calls",
                ),
                ModelResponse(
                    None,
                    None,
                    [ToolCall("child-tests", "run_tests", "{}")],
                    "tool_calls",
                ),
                ModelResponse("child verified", None, [], "stop"),
                ModelResponse("parent done", None, [], "stop"),
            ]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            answer = agent_loop(
                provider,
                self.workspace,
                [],
                test_runner=runner,
            )

        self.assertEqual(answer, "parent done")
        self.assertIn("run_tests", tool_names(provider.calls[1]))
        self.assertNotIn("task", tool_names(provider.calls[1]))
        self.assertEqual(runner.workspaces, [self.workspace])
        self.assertEqual(
            provider.calls[2]["messages"][-1]["content"],
            "Exit code: 0\nOK",
        )

    def test_child_discovers_and_loads_workspace_skill_in_isolated_history(self) -> None:
        skill_path = self.workspace / "skills" / "review" / "SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text(
            "---\n"
            "name: review\n"
            "description: Review delegated code\n"
            "---\n\n"
            "SUBAGENT_PRIVATE_SKILL_BODY\n",
            encoding="utf-8",
        )
        provider = ScriptedProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("task-1", "task", '{"prompt":"review child work"}')],
                    "tool_calls",
                ),
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall(
                            "child-skill",
                            "load_skill",
                            '{"name":"review"}',
                        )
                    ],
                    "tool_calls",
                ),
                ModelResponse("child used review guidance", None, [], "stop"),
                ModelResponse("parent final", None, [], "stop"),
            ]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            answer = agent_loop(provider, self.workspace, [])

        self.assertEqual(answer, "parent final")
        child_first_request = provider.calls[1]
        self.assertIn("load_skill", tool_names(child_first_request))
        self.assertNotIn("task", tool_names(child_first_request))
        child_catalog = next(
            message
            for message in child_first_request["messages"]
            if message.get("name") == "tinyharness_skill_catalog"
        )
        self.assertIn("Review delegated code", child_catalog["content"])
        self.assertNotIn(
            "SUBAGENT_PRIVATE_SKILL_BODY",
            json.dumps(child_first_request, ensure_ascii=False),
        )

        child_loaded_result = next(
            message
            for message in provider.calls[2]["messages"]
            if message.get("tool_call_id") == "child-skill"
        )
        self.assertIn(
            "SUBAGENT_PRIVATE_SKILL_BODY",
            child_loaded_result["content"],
        )
        parent_follow_up = json.dumps(
            provider.calls[3]["messages"],
            ensure_ascii=False,
        )
        self.assertIn("child used review guidance", parent_follow_up)
        self.assertNotIn("SUBAGENT_PRIVATE_SKILL_BODY", parent_follow_up)

    def test_child_loads_memory_but_does_not_extract_child_history(self) -> None:
        memory_path = (
            self.workspace / ".tinyharness" / "memory" / "tabs.md"
        )
        memory_path.parent.mkdir(parents=True)
        memory_path.write_text(
            "---\n"
            "name: tabs\n"
            "description: User prefers tabs for indentation\n"
            "type: user\n"
            "---\n\n"
            "SUBAGENT_MEMORY_BODY\n",
            encoding="utf-8",
        )
        provider = ScriptedProvider(
            [
                ModelResponse(
                    '{"selected_memories":["tabs.md"]}',
                    None,
                    [],
                    "stop",
                ),
                ModelResponse(
                    None,
                    None,
                    [ToolCall("task-1", "task", '{"prompt":"use tabs"}')],
                    "tool_calls",
                ),
                ModelResponse(
                    '{"selected_memories":["tabs.md"]}',
                    None,
                    [],
                    "stop",
                ),
                ModelResponse("child done", None, [], "stop"),
                ModelResponse("parent done", None, [], "stop"),
                ModelResponse('{"memories":[]}', None, [], "stop"),
            ]
        )
        logger = RecordingEventLogger()

        with contextlib.redirect_stdout(io.StringIO()):
            answer = agent_loop(
                provider,
                self.workspace,
                [{"role": "user", "content": "delegate using tabs"}],
                memory_enabled=True,
                event_logger=logger,
            )

        self.assertEqual(answer, "parent done")
        self.assertIn(
            "SUBAGENT_MEMORY_BODY",
            json.dumps(provider.calls[1]["messages"], ensure_ascii=False),
        )
        self.assertIn(
            "SUBAGENT_MEMORY_BODY",
            json.dumps(provider.calls[3]["messages"], ensure_ascii=False),
        )
        extraction_events = [
            event
            for event in logger.events
            if event["event_type"] == "memory_extraction_requested"
        ]
        self.assertEqual(len(extraction_events), 1)
        self.assertNotIn("agent_scope", extraction_events[0]["data"])
        selected_events = [
            event
            for event in logger.events
            if event["event_type"] == "memory_selected"
        ]
        self.assertEqual(len(selected_events), 2)
        self.assertTrue(
            any(event["data"].get("agent_scope") == "subagent" for event in selected_events)
        )

    def test_child_has_independent_compaction_with_scoped_events(self) -> None:
        provider = ScriptedProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("task-1", "task", '{"prompt":"child work"}')],
                    "tool_calls",
                ),
                ModelResponse(
                    None,
                    None,
                    [ToolCall("child-compact", "compact", "{}")],
                    "tool_calls",
                ),
                ModelResponse("CHILD_SUMMARY", None, [], "stop"),
                ModelResponse("child done", None, [], "stop"),
                ModelResponse("parent done", None, [], "stop"),
            ]
        )
        logger = RecordingEventLogger()

        with contextlib.redirect_stdout(io.StringIO()):
            answer = agent_loop(
                provider,
                self.workspace,
                [{"role": "user", "content": "delegate"}],
                max_context_chars=100_000,
                event_logger=logger,
            )

        self.assertEqual(answer, "parent done")
        child_tool_names = tool_names(provider.calls[1])
        self.assertIn("compact", child_tool_names)
        self.assertNotIn("task", child_tool_names)
        self.assertEqual(provider.calls[2]["tools"], [])
        summary_events = [
            event
            for event in logger.events
            if event["event_type"].startswith("context_summary_")
        ]
        self.assertEqual(len(summary_events), 2)
        self.assertTrue(
            all(
                event["data"].get("agent_scope") == "subagent"
                and event["data"].get("parent_tool_call_id") == "task-1"
                for event in summary_events
            )
        )

    def test_child_tool_side_effect_is_shared_but_history_is_isolated(self) -> None:
        provider = ScriptedProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("task-1", "task", '{"prompt":"create a file"}')],
                    "tool_calls",
                ),
                ModelResponse(
                    None,
                    "CHILD_INTERMEDIATE_REASONING",
                    [
                        ToolCall(
                            "child-write",
                            "write_file",
                            '{"path":"child.txt","content":"shared"}',
                        )
                    ],
                    "tool_calls",
                ),
                ModelResponse("created by child", None, [], "stop"),
                ModelResponse("parent observed result", None, [], "stop"),
            ]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            agent_loop(provider, self.workspace, [])

        self.assertEqual(
            (self.workspace / "child.txt").read_text(encoding="utf-8"),
            "shared",
        )
        child_follow_up = json.dumps(provider.calls[2]["messages"])
        self.assertIn("child-write", child_follow_up)
        parent_follow_up = json.dumps(provider.calls[3]["messages"])
        self.assertIn("created by child", parent_follow_up)
        self.assertNotIn("child-write", parent_follow_up)
        self.assertNotIn("CHILD_INTERMEDIATE_REASONING", parent_follow_up)
        self.assertNotIn("Wrote 6 bytes", parent_follow_up)

    def test_fabricated_child_task_call_cannot_recurse(self) -> None:
        provider = ScriptedProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("task-1", "task", '{"prompt":"delegate again"}')],
                    "tool_calls",
                ),
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall(
                            "nested-task",
                            "task",
                            '{"prompt":"must not run"}',
                        )
                    ],
                    "tool_calls",
                ),
                ModelResponse("recursion rejected", None, [], "stop"),
                ModelResponse("parent done", None, [], "stop"),
            ]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            agent_loop(provider, self.workspace, [])

        self.assertEqual(len(provider.calls), 4)
        child_error = provider.calls[2]["messages"][-1]
        self.assertEqual(child_error["tool_call_id"], "nested-task")
        self.assertEqual(
            child_error["content"],
            "Error: ValueError: Unknown tool: task",
        )

    def test_permission_and_hooks_are_shared_with_child_tools(self) -> None:
        command = (
            f'"{sys.executable}" -c '
            '"from pathlib import Path; Path(\'blocked.txt\').write_text(\'bad\')"'
        )
        provider = ScriptedProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("task-1", "task", '{"prompt":"run command"}')],
                    "tool_calls",
                ),
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall(
                            "child-bash",
                            "bash",
                            json.dumps({"command": command}),
                        )
                    ],
                    "tool_calls",
                ),
                ModelResponse("child handled denial", None, [], "stop"),
                ModelResponse("parent done", None, [], "stop"),
            ]
        )
        pre_observed = []
        post_observed = []
        prompts = []
        hooks = ToolHooks()
        hooks.register_pre(
            lambda context: pre_observed.append(context.tool_name)
        )
        hooks.register_post(
            lambda context, result: post_observed.append(context.tool_name)
        )

        with contextlib.redirect_stdout(io.StringIO()):
            agent_loop(
                provider,
                self.workspace,
                [],
                permission_prompt=lambda name, arguments: (
                    prompts.append((name, dict(arguments))) or False
                ),
                tool_hooks=hooks,
            )

        self.assertEqual(pre_observed, ["task", "bash"])
        self.assertEqual(post_observed, ["task"])
        self.assertEqual([name for name, _ in prompts], ["bash"])
        self.assertFalse((self.workspace / "blocked.txt").exists())
        child_result = provider.calls[2]["messages"][-1]["content"]
        self.assertIn("Permission denied", child_result)

    def test_pre_hook_can_block_task_before_child_starts(self) -> None:
        provider = ScriptedProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("task-1", "task", '{"prompt":"blocked"}')],
                    "tool_calls",
                ),
                ModelResponse("handled block", None, [], "stop"),
            ]
        )
        hooks = ToolHooks()
        hooks.register_pre(
            lambda context: (
                HookBlock("delegation disabled")
                if context.tool_name == "task"
                else None
            )
        )
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            agent_loop(provider, self.workspace, [], tool_hooks=hooks)

        self.assertEqual(len(provider.calls), 2)
        self.assertNotIn("[子 Agent 已启动]", stdout.getvalue())
        parent_result = provider.calls[1]["messages"][-1]["content"]
        self.assertIn("delegation disabled", parent_result)

    def test_child_max_turn_failure_becomes_parent_tool_result(self) -> None:
        provider = ScriptedProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("task-1", "task", '{"prompt":"keep working"}')],
                    "tool_calls",
                ),
                ModelResponse(
                    None,
                    None,
                    [ToolCall("child-list", "list_files", "{}")],
                    "tool_calls",
                ),
                ModelResponse("parent handled child failure", None, [], "stop"),
            ]
        )
        logger = RecordingEventLogger()
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            answer = agent_loop(
                provider,
                self.workspace,
                [],
                subagent_max_turns=1,
                event_logger=logger,
            )

        self.assertEqual(answer, "parent handled child failure")
        self.assertIn("[子 Agent 执行失败]", stdout.getvalue())
        parent_result = provider.calls[2]["messages"][-1]["content"]
        self.assertIn("Maximum model turns reached: 1", parent_result)

        child_failed_index = next(
            index
            for index, event in enumerate(logger.events)
            if event["event_type"] == "run_failed"
            and event["data"].get("agent_scope") == "subagent"
        )
        task_finished_index = next(
            index
            for index, event in enumerate(logger.events)
            if event["event_type"] == "tool_finished"
            and event["data"].get("tool_name") == "task"
        )
        self.assertLess(child_failed_index, task_finished_index)
        self.assertEqual(
            logger.events[task_finished_index]["data"]["outcome"],
            "error",
        )

    def test_child_events_are_correlated_with_parent_task_call(self) -> None:
        (self.workspace / "source.txt").write_text("evidence", encoding="utf-8")
        provider = ScriptedProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall(
                            "task-private-id",
                            "task",
                            '{"prompt":"PRIVATE_DELEGATED_PROMPT"}',
                        )
                    ],
                    "tool_calls",
                ),
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall(
                            "child-read",
                            "read_file",
                            '{"path":"source.txt"}',
                        )
                    ],
                    "tool_calls",
                ),
                ModelResponse("child evidence summary", None, [], "stop"),
                ModelResponse("parent final", None, [], "stop"),
            ]
        )
        logger = RecordingEventLogger()

        with contextlib.redirect_stdout(io.StringIO()):
            agent_loop(provider, self.workspace, [], event_logger=logger)

        task_started = next(
            index
            for index, event in enumerate(logger.events)
            if event["event_type"] == "tool_started"
            and event["data"].get("tool_name") == "task"
        )
        child_started = next(
            index
            for index, event in enumerate(logger.events)
            if event["event_type"] == "run_started"
            and event["data"].get("agent_scope") == "subagent"
        )
        child_finished = next(
            index
            for index, event in enumerate(logger.events)
            if event["event_type"] == "run_finished"
            and event["data"].get("agent_scope") == "subagent"
        )
        task_finished = next(
            index
            for index, event in enumerate(logger.events)
            if event["event_type"] == "tool_finished"
            and event["data"].get("tool_name") == "task"
        )
        self.assertLess(task_started, child_started)
        self.assertLess(child_started, child_finished)
        self.assertLess(child_finished, task_finished)

        child_events = [
            event
            for event in logger.events
            if event["data"].get("agent_scope") == "subagent"
        ]
        self.assertTrue(child_events)
        self.assertTrue(
            all(
                event["data"]["parent_tool_call_id"] == "task-private-id"
                for event in child_events
            )
        )
        self.assertNotIn("agent_scope", logger.events[task_started]["data"])
        self.assertNotIn("PRIVATE_DELEGATED_PROMPT", json.dumps(logger.events))

    def test_multiple_tasks_run_sequentially_with_fresh_contexts(self) -> None:
        provider = ScriptedProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall("task-1", "task", '{"prompt":"first child"}'),
                        ToolCall("task-2", "task", '{"prompt":"second child"}'),
                    ],
                    "tool_calls",
                ),
                ModelResponse("first summary", None, [], "stop"),
                ModelResponse("second summary", None, [], "stop"),
                ModelResponse("parent done", None, [], "stop"),
            ]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            agent_loop(provider, self.workspace, [])

        self.assertEqual(
            provider.calls[1]["messages"][-1]["content"],
            "first child",
        )
        self.assertEqual(
            provider.calls[2]["messages"][-1]["content"],
            "second child",
        )
        self.assertNotIn(
            "first child",
            json.dumps(provider.calls[2]["messages"]),
        )
        parent_results = provider.calls[3]["messages"][-2:]
        self.assertEqual(
            [message["tool_call_id"] for message in parent_results],
            ["task-1", "task-2"],
        )
        self.assertEqual(
            [message["content"] for message in parent_results],
            ["first summary", "second summary"],
        )

    def test_child_todo_state_does_not_inherit_parent_todos(self) -> None:
        provider = ScriptedProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall(
                            "parent-todo",
                            "todo_write",
                            '{"todos":[{"content":"PARENT_TODO_SECRET","status":"pending"}]}',
                        ),
                        ToolCall(
                            "task-1",
                            "task",
                            '{"prompt":"inspect independently"}',
                        ),
                    ],
                    "tool_calls",
                ),
                ModelResponse(
                    None,
                    None,
                    [ToolCall("child-list-1", "list_files", "{}")],
                    "tool_calls",
                ),
                ModelResponse(
                    None,
                    None,
                    [ToolCall("child-list-2", "list_files", "{}")],
                    "tool_calls",
                ),
                ModelResponse(
                    None,
                    None,
                    [ToolCall("child-list-3", "list_files", "{}")],
                    "tool_calls",
                ),
                ModelResponse("child done", None, [], "stop"),
                ModelResponse("parent done", None, [], "stop"),
            ]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            agent_loop(provider, self.workspace, [])

        child_reminder = provider.calls[4]["messages"][-1]["content"]
        self.assertIn("Current todos:\nNo todos.", child_reminder)
        self.assertNotIn("PARENT_TODO_SECRET", child_reminder)

    def test_child_event_log_failure_aborts_parent_run(self) -> None:
        provider = ScriptedProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("task-1", "task", '{"prompt":"child task"}')],
                    "tool_calls",
                ),
                ModelResponse("must not be requested", None, [], "stop"),
            ]
        )
        logger = ChildFailingEventLogger()

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(EventLogError, "child log unavailable"):
                agent_loop(
                    provider,
                    self.workspace,
                    [],
                    event_logger=logger,
                )

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(logger.events[-1]["event_type"], "tool_started")
        self.assertEqual(logger.events[-1]["data"]["tool_name"], "task")


if __name__ == "__main__":
    unittest.main()
