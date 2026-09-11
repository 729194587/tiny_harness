import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.messages import ToolCall
from tiny_harness.agent.tool_batch import execute_tool_batch
from tiny_harness.runtime.events import EventType, hash_text
from tiny_harness.runtime.hooks import HookBlock, ToolHooks
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.runtime.tool_trace import ToolTraceConfig
from tiny_harness.tools.registry import ToolRegistry, dispatch
from tiny_harness.tools.shell import build_tools


class Recorder:
    def __init__(self):
        self.events = []

    def emit(self, event_type, data=None):
        self.events.append((event_type, dict(data or {})))


class ToolTraceTest(unittest.TestCase):
    def run_tool(self, config=ToolTraceConfig(), *, arguments=None, policy=None,
                 hooks=None, fail=False, output="你好abcdef"):
        def run(workspace, command):
            if fail:
                raise ValueError("failure details")
            return output

        registry = ToolRegistry()
        registry.register(build_tools(SimpleNamespace(
            workspace=Path.cwd(), shell_runner=SimpleNamespace(run=run),
            tool_trace=config,
        ))[0])
        logger = Recorder()
        result = dispatch(
            registry, ToolCall("1", "bash", arguments if arguments is not None
                               else json.dumps({"command": "echo secret"})),
            permission_policy=SimpleNamespace(
                decide=policy or (lambda *args: PermissionDecision.ALLOW)),
            event_logger=logger, tool_trace=config, tool_hooks=hooks,
        )
        return result, logger.events

    def test_default_payload_is_unchanged(self):
        result, events = self.run_tool()
        self.assertEqual(set(events[0][1]), {
            "turn", "tool_call_id", "tool_name", "arguments_hash"})
        self.assertEqual(set(events[-1][1]), {
            "turn", "tool_call_id", "tool_name", "outcome", "duration_ms",
            "content_length", "content_hash"})
        self.assertEqual(events[-1][1]["content_hash"], hash_text(result.content))

    def test_command_and_bounded_unicode_preview(self):
        result, events = self.run_tool(ToolTraceConfig(True, 3))
        self.assertEqual(events[0][1]["command"], "echo secret")
        self.assertEqual(events[-1][1]["content_preview"], "你好a")
        self.assertEqual(result.content, "你好abcdef")
        self.assertEqual(events[-1][1]["content_length"], len(result.content))
        self.assertEqual(events[-1][1]["content_hash"], hash_text(result.content))
        self.assertEqual([kind for kind, _ in events], [
            EventType.TOOL_CALLED, EventType.TOOL_STARTED, EventType.TOOL_RESULT])

    def test_preview_boundaries(self):
        for output in ("", "ab", "abc", "abcd"):
            with self.subTest(output=output):
                _, events = self.run_tool(ToolTraceConfig(True, 3), output=output)
                self.assertEqual(events[-1][1]["content_preview"], output[:3])
        for config in (ToolTraceConfig(True, 0), ToolTraceConfig(False, 3)):
            _, events = self.run_tool(config)
            self.assertNotIn("content_preview", events[-1][1])

    def test_all_ordinary_failure_paths_have_bounded_preview(self):
        hooks = ToolHooks()
        hooks.register_pre(lambda context: HookBlock("blocked details"))
        for options, outcome in (
            ({"arguments": "{"}, "error"),
            ({"fail": True}, "error"),
            ({"policy": lambda *args: PermissionDecision.DENY}, "permission_denied"),
            ({"hooks": hooks}, "hook_blocked"),
        ):
            with self.subTest(outcome=outcome, options=options):
                result, events = self.run_tool(ToolTraceConfig(True, 5), **options)
                self.assertEqual(events[-1][1]["outcome"], outcome)
                self.assertEqual(events[-1][1]["content_preview"], result.content[:5])

    def test_runtime_batch_and_child_inherit_config(self):
        config = ToolTraceConfig(True, 2)
        logger = Recorder()
        with tempfile.TemporaryDirectory() as directory:
            context = create_run_context(
                object(), Path(directory), tool_trace=config, event_logger=logger,
                permission_policy=SimpleNamespace(decide=lambda *args: PermissionDecision.ALLOW),
                shell_runner=SimpleNamespace(run=lambda *args: "abcdef"),
            )
            self.assertIs(context.subagent_runner.tool_trace, config)
            messages = []
            execute_tool_batch(messages, [ToolCall("1", "bash", '{"command":"echo x"}')], context)
            results = [data for kind, data in logger.events if kind == EventType.TOOL_RESULT]
            self.assertEqual(results[0]["content_preview"], "ab")
            self.assertEqual(messages[0]["content"], "abcdef")

    def test_invalid_configuration(self):
        for value in (-1, True, 1.5, "3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ToolTraceConfig(result_preview_chars=value)
