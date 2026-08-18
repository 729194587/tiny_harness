import copy
import json
import tempfile
import unittest
from pathlib import Path

from tiny_harness.agent.loop import agent_loop
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.runtime.hooks import (
    HookBlock,
    HookExecutionError,
    ToolHooks,
)
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.tools.registry import dispatch


class RecordingEventLogger:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event_type, data=None) -> None:
        self.events.append(
            {"event_type": event_type.value, "data": dict(data or {})}
        )


class AlwaysAskPolicy:
    def decide(self, tool_name, arguments):
        del tool_name, arguments
        return PermissionDecision.ASK


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


class ToolHooksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_pre_and_post_hooks_run_in_registration_order(self) -> None:
        observed = []
        hooks = ToolHooks()
        hooks.register_pre(lambda context: observed.append(("pre-1", context.tool_name)))
        hooks.register_pre(lambda context: observed.append(("pre-2", context.tool_name)))
        hooks.register_post(
            lambda context, result: observed.append(
                ("post-1", context.tool_name, result.tool_call_id)
            )
        )
        hooks.register_post(
            lambda context, result: observed.append(
                ("post-2", context.tool_name, result.tool_call_id)
            )
        )

        result = dispatch(
            self.workspace,
            ToolCall(
                "write-1",
                "write_file",
                '{"path":"a.txt","content":"A"}',
            ),
            tool_hooks=hooks,
        )

        self.assertEqual(result.tool_call_id, "write-1")
        self.assertEqual(
            observed,
            [
                ("pre-1", "write_file"),
                ("pre-2", "write_file"),
                ("post-1", "write_file", "write-1"),
                ("post-2", "write_file", "write-1"),
            ],
        )

    def test_pre_hook_block_skips_remaining_hooks_permission_and_handler(self) -> None:
        observed = []
        prompted = []
        logger = RecordingEventLogger()
        hooks = ToolHooks()

        def block(context):
            observed.append(("block", context.tool_call_id))
            return HookBlock("PRIVATE_BLOCK_REASON")

        hooks.register_pre(block)
        hooks.register_pre(lambda context: observed.append(("late", context.tool_name)))
        hooks.register_post(lambda context, result: observed.append(("post", result)))

        result = dispatch(
            self.workspace,
            ToolCall(
                "write-1",
                "write_file",
                '{"path":"blocked.txt","content":"bad"}',
            ),
            permission_policy=AlwaysAskPolicy(),
            permission_prompt=lambda *_: prompted.append(True) or True,
            event_logger=logger,
            tool_hooks=hooks,
        )

        self.assertEqual(observed, [("block", "write-1")])
        self.assertEqual(prompted, [])
        self.assertFalse((self.workspace / "blocked.txt").exists())
        self.assertEqual(result.tool_call_id, "write-1")
        self.assertIn("PRIVATE_BLOCK_REASON", result.content)
        self.assertEqual(
            [event["event_type"] for event in logger.events],
            ["tool_hook_blocked"],
        )
        self.assertNotIn("PRIVATE_BLOCK_REASON", json.dumps(logger.events))

    def test_permission_denial_does_not_run_post_hook(self) -> None:
        observed = []
        hooks = ToolHooks()
        hooks.register_post(lambda context, result: observed.append(result.content))

        result = dispatch(
            self.workspace,
            ToolCall("bash-1", "bash", '{"command":"python --version"}'),
            tool_hooks=hooks,
        )

        self.assertIn("Permission denied", result.content)
        self.assertEqual(observed, [])

    def test_post_hook_observes_success_and_handler_error(self) -> None:
        observed = []
        hooks = ToolHooks()
        hooks.register_post(
            lambda context, result: observed.append(
                (context.tool_call_id, result.content)
            )
        )

        success = dispatch(
            self.workspace,
            ToolCall(
                "write-1",
                "write_file",
                '{"path":"a.txt","content":"A"}',
            ),
            tool_hooks=hooks,
        )
        failure = dispatch(
            self.workspace,
            ToolCall("read-1", "read_file", '{"path":"missing.txt"}'),
            tool_hooks=hooks,
        )

        self.assertEqual(observed[0], ("write-1", success.content))
        self.assertEqual(observed[1], ("read-1", failure.content))
        self.assertTrue(observed[1][1].startswith("Error: FileNotFoundError:"))

    def test_post_hook_cannot_modify_returned_tool_result(self) -> None:
        hooks = ToolHooks()

        def mutate_result(context, result):
            del context
            result.content = "tampered"

        hooks.register_post(mutate_result)
        result = dispatch(
            self.workspace,
            ToolCall(
                "write-1",
                "write_file",
                '{"path":"a.txt","content":"A"}',
            ),
            tool_hooks=hooks,
        )

        self.assertNotEqual(result.content, "tampered")
        self.assertTrue(result.content.startswith("Wrote "))

    def test_pre_hook_exception_is_fail_closed_and_recorded(self) -> None:
        logger = RecordingEventLogger()
        hooks = ToolHooks()

        def fail(context):
            del context
            raise ValueError("PRIVATE_HOOK_FAILURE")

        hooks.register_pre(fail)

        with self.assertRaises(HookExecutionError) as raised:
            dispatch(
                self.workspace,
                ToolCall(
                    "write-1",
                    "write_file",
                    '{"path":"a.txt","content":"A"}',
                ),
                event_logger=logger,
                tool_hooks=hooks,
            )

        self.assertEqual(raised.exception.stage, "pre")
        self.assertEqual(raised.exception.hook_index, 1)
        self.assertFalse((self.workspace / "a.txt").exists())
        self.assertEqual(logger.events[-1]["event_type"], "tool_hook_failed")
        self.assertEqual(logger.events[-1]["data"]["error_type"], "ValueError")
        self.assertNotIn("PRIVATE_HOOK_FAILURE", json.dumps(logger.events))

    def test_pre_hook_cannot_modify_handler_arguments(self) -> None:
        hooks = ToolHooks()

        def mutate_arguments(context):
            context.arguments["content"] = "tampered"

        hooks.register_pre(mutate_arguments)

        with self.assertRaises(HookExecutionError) as raised:
            dispatch(
                self.workspace,
                ToolCall(
                    "write-1",
                    "write_file",
                    '{"path":"a.txt","content":"A"}',
                ),
                tool_hooks=hooks,
            )

        self.assertEqual(raised.exception.error_type, "TypeError")
        self.assertFalse((self.workspace / "a.txt").exists())

    def test_post_hook_exception_happens_after_tool_finished(self) -> None:
        logger = RecordingEventLogger()
        hooks = ToolHooks()

        def fail(context, result):
            del context, result
            raise RuntimeError("post failed")

        hooks.register_post(fail)

        with self.assertRaises(HookExecutionError) as raised:
            dispatch(
                self.workspace,
                ToolCall(
                    "write-1",
                    "write_file",
                    '{"path":"a.txt","content":"A"}',
                ),
                event_logger=logger,
                tool_hooks=hooks,
            )

        self.assertEqual(raised.exception.stage, "post")
        self.assertEqual((self.workspace / "a.txt").read_text(encoding="utf-8"), "A")
        self.assertEqual(
            [event["event_type"] for event in logger.events],
            ["tool_started", "tool_finished", "tool_hook_failed"],
        )

    def test_invalid_hook_return_is_an_execution_error(self) -> None:
        hooks = ToolHooks()
        hooks.register_pre(lambda context: "not a HookBlock")

        with self.assertRaises(HookExecutionError) as raised:
            dispatch(
                self.workspace,
                ToolCall("read-1", "read_file", '{"path":"a.txt"}'),
                tool_hooks=hooks,
            )

        self.assertEqual(raised.exception.error_type, "TypeError")

    def test_one_blocked_call_does_not_skip_other_calls_in_same_response(self) -> None:
        hooks = ToolHooks()
        hooks.register_pre(
            lambda context: (
                HookBlock("first file blocked")
                if context.arguments.get("path") == "a.txt"
                else None
            )
        )
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
        messages = [{"role": "user", "content": "write two files"}]

        answer = agent_loop(
            provider,
            self.workspace,
            messages,
            tool_hooks=hooks,
            max_context_chars=10_000,
        )

        self.assertEqual(answer, "done")
        self.assertFalse((self.workspace / "a.txt").exists())
        self.assertEqual((self.workspace / "b.txt").read_text(encoding="utf-8"), "B")
        tool_messages = provider.calls[1]["messages"][-2:]
        self.assertEqual(
            [message["tool_call_id"] for message in tool_messages],
            ["write-1", "write-2"],
        )
        self.assertIn("blocked", tool_messages[0]["content"])

    def test_hook_failure_becomes_run_failed_in_agent_loop(self) -> None:
        hooks = ToolHooks()

        def fail(context):
            del context
            raise RuntimeError("hook failed")

        hooks.register_pre(fail)
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
                        )
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

        self.assertEqual(logger.events[-1]["event_type"], "run_failed")
        self.assertEqual(
            logger.events[-1]["data"]["error_type"],
            "HookExecutionError",
        )

    def test_blocked_todo_does_not_reset_reminder_counter(self) -> None:
        hooks = ToolHooks()
        hooks.register_pre(
            lambda context: (
                HookBlock("planning blocked")
                if context.tool_name == "todo_write"
                else None
            )
        )
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("list-1", "list_files", "{}")],
                    "tool_calls",
                ),
                ModelResponse(
                    None,
                    None,
                    [ToolCall("list-2", "list_files", "{}")],
                    "tool_calls",
                ),
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall(
                            "todo-1",
                            "todo_write",
                            '{"todos":[{"content":"Plan","status":"pending"}]}',
                        )
                    ],
                    "tool_calls",
                ),
                ModelResponse("done", None, [], "stop"),
            ]
        )

        agent_loop(provider, self.workspace, [], tool_hooks=hooks)

        blocked_result = provider.calls[-1]["messages"][-1]["content"]
        self.assertIn("planning blocked", blocked_result)
        self.assertIn("<todo-reminder>", blocked_result)
        self.assertIn("Current todos:\nNo todos.", blocked_result)


if __name__ == "__main__":
    unittest.main()
