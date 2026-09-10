import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tiny_harness.__main__ import main, _ask_permission
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.agent.loop import run_agent as agent_loop
from tiny_harness.runtime.console import ConsoleEventLogger
from tiny_harness.runtime.events import EventLogError, CompositeEventLogger, JsonlEventLogger, EventType, ScopedEventLogger
from tiny_harness.runtime.hooks import ToolHooks, HookBlock, HookExecutionError
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.runtime.todos import TodoManager
from tiny_harness.tools.todo import build_tools
from tiny_harness.tools.definition import ToolDefinition
from tiny_harness.tools.registry import ToolRegistry, dispatch, emit_tool_called


class ConsoleV1Examples:
    def setUp(self):
        self.output = io.StringIO()
        self.console = ConsoleEventLogger(stream=self.output)
        self.registry = ToolRegistry()
        self.registry.register(ToolDefinition(
            "read_file", "read", {}, lambda call, args: "SECRET_RESULT",
            trace_metadata=lambda args: {"path": args.get("path", "")},
        ))

    def call(self, name="read_file", arguments=None, **options):
        return dispatch(self.registry, ToolCall("1", name, json.dumps(arguments or {"path": "a.py"})),
                        event_logger=options.pop("event_logger", self.console), **options)

    def test_success_and_jsonl_duration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.jsonl"
            logger = CompositeEventLogger(self.console, JsonlEventLogger(path))
            with patch("tiny_harness.tools.registry.perf_counter", side_effect=[1, 1.125]):
                self.call(event_logger=logger)
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([e["sequence"] for e in events], [1, 2, 3])
        self.assertEqual(events[-1]["data"]["duration_ms"], 125)
        self.assertEqual(self.output.getvalue(), "[正在调用工具：read_file] a.py\n[工具调用完成：read_file]\n")
        self.assertNotIn("SECRET_RESULT", self.output.getvalue())

    def test_failures_denial_and_hook_block(self):
        def fail(call, args):
            raise FileNotFoundError("SECRET_EXCEPTION")
        self.registry.register(ToolDefinition("broken", "", {}, fail))
        self.call("broken", permission_policy=SimpleNamespace(decide=lambda *args: PermissionDecision.ALLOW))
        self.call(permission_policy=SimpleNamespace(decide=lambda *args: PermissionDecision.DENY))
        hooks = ToolHooks()
        hooks.register_pre(lambda context: HookBlock("SECRET_REASON"))
        self.call(tool_hooks=hooks)
        text = self.output.getvalue()
        self.assertIn("[工具调用失败：broken] FileNotFoundError", text)
        self.assertIn("[工具调用被权限策略拒绝：read_file]", text)
        self.assertIn("[工具调用被 Hook 阻止：read_file]", text)
        self.assertNotIn("SECRET", text)
        self.assertNotIn("工具调用完成", text)

    def test_safe_bounded_metadata_and_execution_order(self):
        call = ToolCall("1", "read_file", json.dumps({"path": "x" * 10000, "content": "PRIVATE" * 10000}))
        emit_tool_called(self.registry, call, self.console, turn=0)
        self.assertEqual(self.output.getvalue(), "")
        dispatch(self.registry, call, event_logger=self.console, tool_called_logged=True)
        self.assertLess(len(self.output.getvalue()), 220)
        self.assertNotIn("PRIVATE", self.output.getvalue())
        self.output.seek(0)
        self.output.truncate()
        self.call(arguments={"path": "a\n\x1b[31m.py"})
        self.assertEqual(len(self.output.getvalue().splitlines()), 2)
        self.assertNotIn("\x1b", self.output.getvalue())

    def test_skill_compaction_retry_and_scope(self):
        self.registry.register(ToolDefinition("load_skill", "", {}, lambda c, a: "SECRET_SKILL",
                                              trace_metadata=lambda a: {"name": a["name"]}))
        self.call("load_skill", {"name": "code_review"})
        child = ScopedEventLogger(self.console, {"agent_scope": "subagent", "parent_tool_call_id": "task1"})
        child.emit(EventType.SUBAGENT_STARTED)
        self.call(event_logger=child)
        child.emit(EventType.SUBAGENT_FINISHED)
        self.console.emit(EventType.CONTEXT_COMPACTED, {"before_tokens": 23079, "after_tokens": 12056})
        self.console.emit(EventType.MODEL_RETRY_SCHEDULED, {"attempt": 1, "delay_ms": 50})
        text = self.output.getvalue()
        self.assertIn("[正在装载 Skill：code_review]\n[Skill 装载完成：code_review]", text)
        self.assertIn("[子 Agent 已启动]\n  [正在调用工具：read_file] a.py\n  [工具调用完成：read_file]\n[子 Agent 已完成]", text)
        self.assertIn("[上下文已压缩：23079 → 12056 tokens]", text)
        self.assertIn("[模型调用将重试]", text)

    def test_todo_bound_logger_and_no_direct_print(self):
        registry = ToolRegistry()
        context = SimpleNamespace(todo_manager=TodoManager(), event_logger=self.console)
        registry.register(build_tools(context)[0])
        with patch("builtins.print", wraps=print) as printer:
            result = dispatch(registry, ToolCall("todo1", "todo_write", json.dumps({"todos": [
                {"content": "PRIVATE_PLAN", "status": "in_progress"}]})))
        self.assertIn("PRIVATE_PLAN", result.content)
        self.assertIn("[任务进度：已完成 0/1，进行中 1]", self.output.getvalue())
        self.assertEqual(printer.call_count, 1)  # Console is the sole printer.
        self.assertNotIn("PRIVATE_PLAN", self.output.getvalue())

    def test_real_subagent_progress_and_no_direct_print(self):
        responses = iter([
            ModelResponse(None, None, [ToolCall("task1", "task", '{"prompt":"PRIVATE"}')], "tool_calls"),
            ModelResponse(None, None, [ToolCall("child1", "read_file", '{"path":"a.py"}')], "tool_calls"),
            ModelResponse("child done", None, [], "stop"),
            ModelResponse("final", None, [], "stop"),
        ])
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            (workspace / "a.py").write_text("PRIVATE_RESULT", encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                answer = agent_loop(SimpleNamespace(complete=lambda *args: next(responses)), workspace, [],
                                    event_logger=self.console)
        self.assertEqual(answer, "final")
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("  [正在调用工具：read_file] a.py", self.output.getvalue())
        self.assertIn("[子 Agent 已完成]", self.output.getvalue())
        self.assertNotIn("PRIVATE", self.output.getvalue())

    def test_cli_modes_with_jsonl_and_final_answer(self):
        for mode in ([], ["--quiet"], ["--verbose"]):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                log = Path(folder) / "run.jsonl"
                out, err = io.StringIO(), io.StringIO()
                with patch.dict(os.environ, {"TINYHARNESS_API_KEY": "test"}), \
                     patch("tiny_harness.__main__.ChatCompletionsProvider"), \
                     patch("tiny_harness.__main__.AgentSession") as session:
                    def submit(task):
                        logger = session.call_args.kwargs["event_logger_factory"]()
                        logger.emit(EventType.MODEL_REQUESTED, {"attempt": 1})
                        self.call(event_logger=logger)
                        self.call("unknown", event_logger=logger)
                        return "最终回答"
                    session.return_value.submit.side_effect = submit
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        self.assertEqual(main(["task", "--workspace", folder, "--event-log", str(log)] + mode), 0)
                self.assertEqual(out.getvalue(), "最终回答\n")
                self.assertIn("[工具调用失败：unknown] ValueError", err.getvalue())
                self.assertEqual("正在调用工具" in err.getvalue(), mode != ["--quiet"])
                self.assertEqual("正在请求模型" in err.getvalue(), mode == ["--verbose"])
                self.assertEqual("耗时" in err.getvalue(), mode == ["--verbose"])
                self.assertIn('"tool_started"', log.read_text(encoding="utf-8"))

    def test_console_io_failure_retains_event_log_error_contract(self):
        stream = io.StringIO()
        stream.close()
        logger = ConsoleEventLogger(stream=stream)
        with self.assertRaises(EventLogError):
            logger.emit(EventType.SUBAGENT_STARTED)

    def test_hook_failure_and_subagent_failure_stay_visible_in_quiet(self):
        logger = ConsoleEventLogger(quiet=True, stream=self.output)
        hooks = ToolHooks()
        def fail(context, result):
            raise RuntimeError("PRIVATE_HOOK")
        hooks.register_post(fail)
        with self.assertRaises(HookExecutionError):
            self.call(event_logger=logger, tool_hooks=hooks)
        logger.emit(EventType.SUBAGENT_FAILED, {"agent_scope": "subagent"})
        self.assertIn("[工具 Hook 执行失败：read_file] RuntimeError", self.output.getvalue())
        self.assertIn("[子 Agent 执行失败]", self.output.getvalue())
        self.assertNotIn("PRIVATE", self.output.getvalue())
        self.assertNotIn("正在", self.output.getvalue())

    def test_quiet_cli_fatal_error_is_terse_and_nonzero(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {"TINYHARNESS_API_KEY": "test"}), \
             patch("tiny_harness.__main__.ChatCompletionsProvider"), \
             patch("tiny_harness.__main__.AgentSession") as session:
            session.return_value.submit.side_effect = RuntimeError("PRIVATE_ERROR")
            with contextlib.redirect_stderr(self.output):
                self.assertEqual(main(["task", "--workspace", folder, "--quiet"]), 1)
        self.assertEqual(self.output.getvalue(), "[任务已终止：RuntimeError]\n")

    def test_real_cli_failure_is_reported_once_in_all_modes(self):
        for mode in ([], ["--quiet"], ["--verbose"]):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder, \
                 patch.dict(os.environ, {"TINYHARNESS_API_KEY": "test"}), \
                 patch("tiny_harness.__main__.ChatCompletionsProvider") as provider:
                provider.return_value.complete.side_effect = RuntimeError("PRIVATE_ERROR")
                err = io.StringIO()
                path = Path(folder) / "events.jsonl"
                with contextlib.redirect_stderr(err):
                    self.assertEqual(main(["task", "--workspace", folder, "--event-log", str(path)] + mode), 1)
                self.assertEqual(err.getvalue().count("[运行失败] RuntimeError"), 1)
                self.assertNotIn("任务已终止", err.getvalue())
                self.assertNotIn("PRIVATE_ERROR", err.getvalue())
                events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(sum(e["event_type"] == "run_failed" for e in events), 1)

    def test_real_subagent_failure_is_reported_once_and_parent_continues(self):
        for quiet, verbose in ((False, False), (True, False), (False, True)):
            with self.subTest(quiet=quiet, verbose=verbose), tempfile.TemporaryDirectory() as folder:
                responses = iter([
                    ModelResponse(None, None, [ToolCall("task1", "task", '{"prompt":"PRIVATE"}')], "tool_calls"),
                    RuntimeError("PRIVATE_ERROR"),
                    ModelResponse("parent recovered", None, [], "stop"),
                ])
                def complete(*args):
                    response = next(responses)
                    if isinstance(response, Exception):
                        raise response
                    return response
                output = io.StringIO()
                console = ConsoleEventLogger(stream=output, quiet=quiet, verbose=verbose)
                path = Path(folder) / "events.jsonl"
                logger = CompositeEventLogger(console, JsonlEventLogger(path))
                answer = agent_loop(SimpleNamespace(complete=complete), Path(folder), [], event_logger=logger)
                self.assertEqual(answer, "parent recovered")
                self.assertEqual(output.getvalue().count("[子 Agent 执行失败] RuntimeError"), 1)
                self.assertNotIn("[运行失败]", output.getvalue())
                self.assertNotIn("[工具调用失败：task]", output.getvalue())
                self.assertNotIn("PRIVATE_ERROR", output.getvalue())
                events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                self.assertTrue(any(e["event_type"] == "run_failed" for e in events))
                self.assertTrue(any(e["event_type"] == "subagent_failed" for e in events))
                self.assertTrue(any(e["event_type"] == "tool_result" and e["data"].get("outcome") == "error" for e in events))
                # A later task failure without a child execution must still appear.
                console.emit(EventType.TOOL_RESULT, {"tool_call_id": "task2", "tool_name": "task",
                                                     "outcome": "error", "error_type": "ValueError"})
                self.assertIn("[工具调用失败：task] ValueError", output.getvalue())

    def test_jsonl_failure_after_run_failure_still_gets_cli_fallback(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {"TINYHARNESS_API_KEY": "test"}), \
             patch("tiny_harness.__main__.ChatCompletionsProvider") as provider:
            provider.return_value.complete.side_effect = RuntimeError("PRIVATE_ERROR")
            original_emit = JsonlEventLogger.emit
            def emit(logger, event_type, data=None):
                if event_type == EventType.RUN_FAILED:
                    raise EventLogError("PRIVATE_LOG_ERROR")
                return original_emit(logger, event_type, data)
            with patch.object(JsonlEventLogger, "emit", emit), contextlib.redirect_stderr(self.output):
                self.assertEqual(main(["task", "--workspace", folder, "--event-log", str(Path(folder) / "log.jsonl")]), 1)
        self.assertIn("[运行失败] RuntimeError", self.output.getvalue())
        self.assertIn("[任务已终止：EventLogError]", self.output.getvalue())
        self.assertNotIn("PRIVATE", self.output.getvalue())

    def test_tty_colors_labels_identifiers_and_keeps_jsonl_plain(self):
        output = io.StringIO()
        orange, blue, reset = "\x1b[38;5;208m", "\x1b[38;5;75m", "\x1b[0m"
        with patch.object(output, "isatty", return_value=True), \
             patch.dict(os.environ, {}, clear=True), tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.jsonl"
            console = ConsoleEventLogger(stream=output)
            logger = CompositeEventLogger(console, JsonlEventLogger(path))
            self.call(event_logger=logger)
            self.assertEqual(output.getvalue().splitlines()[0],
                             f"{orange}[正在调用工具：{blue}read_file{orange}]{reset} a.py")
            logger.emit(EventType.TOOL_CALLED, {"tool_call_id": "skill1", "name": "code_review"})
            logger.emit(EventType.TOOL_STARTED, {"tool_call_id": "skill1", "tool_name": "load_skill"})
            logger.emit(EventType.RUN_FAILED, {"error_type": "RuntimeError"})
            self.assertIn(f"{orange}[正在装载 Skill：{blue}code_review{orange}]{reset}", output.getvalue())
            self.assertIn(f"{orange}[运行失败]{reset} {blue}RuntimeError{reset}", output.getvalue())
            raw = path.read_text(encoding="utf-8")
            self.assertNotIn("\x1b", raw)
            self.assertNotIn("\\u001b", raw)
            self.assertEqual(json.loads(raw.splitlines()[-1])["data"], {"error_type": "RuntimeError"})

    def test_color_disabled_for_redirects_and_any_no_color_value(self):
        for tty, no_color in ((False, None), (True, ""), (True, "1"), (True, "0")):
            with self.subTest(tty=tty, no_color=no_color):
                output = io.StringIO()
                environment = {} if no_color is None else {"NO_COLOR": no_color}
                with patch.object(output, "isatty", return_value=tty), \
                     patch.dict(os.environ, environment, clear=True):
                    self.call(event_logger=ConsoleEventLogger(stream=output))
                self.assertEqual(output.getvalue(),
                                 "[正在调用工具：read_file] a.py\n[工具调用完成：read_file]\n")

    def test_color_uses_current_stderr_and_preserves_indent_and_metadata(self):
        output = io.StringIO()
        console = ConsoleEventLogger(verbose=True)
        with patch.object(output, "isatty", return_value=True), \
             patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(output):
            child = ScopedEventLogger(console, {"agent_scope": "subagent", "parent_tool_call_id": "task1"})
            self.call(event_logger=child)
        lines = output.getvalue().splitlines()
        self.assertTrue(lines[0].startswith("  \x1b["))
        self.assertTrue(lines[0].endswith("\x1b[0m a.py"))
        self.assertIn("\x1b[0m 耗时 ", lines[1])
        # The same logger follows a later redirection instead of caching TTY state.
        redirected = io.StringIO()
        with contextlib.redirect_stderr(redirected):
            self.call(event_logger=console)
        self.assertNotIn("\x1b", redirected.getvalue())

    def test_permission_prompt_never_dumps_content(self):
        with patch("builtins.input", return_value="no"), contextlib.redirect_stdout(self.output):
            _ask_permission("write_file", {"path": "a.py", "content": "SECRET" * 10000})
        self.assertNotIn("SECRET", self.output.getvalue())


class ConsoleV2Test(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.console = ConsoleEventLogger(stream=self.output)

    def tool(self, call_id, name, outcome="returned", error_type=None):
        self.console.emit(EventType.TOOL_CALLED, {
            "tool_call_id": call_id, "tool_name": name,
        })
        self.console.emit(EventType.TOOL_STARTED, {
            "tool_call_id": call_id, "tool_name": name,
        })
        result = {"tool_call_id": call_id, "tool_name": name, "outcome": outcome}
        if error_type:
            result["error_type"] = error_type
        self.console.emit(EventType.TOOL_RESULT, result)

    def test_consecutive_successes_are_aggregated_at_activity_boundary(self):
        self.tool("1", "read_file")
        self.tool("2", "read_file")
        self.tool("3", "grep")
        self.assertEqual(self.output.getvalue(), "")
        self.console.emit(EventType.MODEL_REQUESTED, {"turn": 2})
        self.assertEqual(
            self.output.getvalue(),
            "🔎 Exploring\n\nread_file ×2\ngrep ×1\n",
        )
        self.assertNotIn("finished", self.output.getvalue().lower())

    def test_failure_and_denial_are_immediate_and_not_hidden(self):
        self.tool("1", "read_file")
        self.tool("2", "broken", "error", "FileNotFoundError")
        self.assertEqual(
            self.output.getvalue(),
            "🔎 Exploring\n\nread_file ×1\n✗ broken failed: FileNotFoundError\n",
        )
        self.console.emit(EventType.TOOL_CALLED, {"tool_call_id": "3", "tool_name": "write_file"})
        self.console.emit(EventType.TOOL_DENIED, {"tool_call_id": "3", "tool_name": "write_file"})
        self.assertIn("✗ write_file permission denied\n", self.output.getvalue())

    def test_context_is_only_shown_near_limit_and_when_compacted(self):
        self.console.emit(EventType.CONTEXT_PREPARED, {
            "context_tokens": 50_000, "hard_limit": 100_000, "pressure": .5,
        })
        self.assertEqual(self.output.getvalue(), "")
        self.console.emit(EventType.CONTEXT_PREPARED, {
            "context_tokens": 82_000, "hard_limit": 100_000, "pressure": .82,
        })
        self.console.emit(EventType.CONTEXT_COMPACTED, {
            "before_tokens": 82_000, "after_tokens": 51_000,
        })
        self.assertEqual(
            self.output.getvalue(),
            "⚠ Context nearing limit\n\n82k / 100k tokens\n"
            "⚡ Context compacted\n\n82k → 51k tokens\n",
        )

    def test_summary_counts_run_statistics(self):
        clock = iter((10.0, 52.0))
        console = ConsoleEventLogger(stream=self.output, clock=lambda: next(clock))
        console.emit(EventType.RUN_STARTED)
        console.emit(EventType.MODEL_REQUESTED, {"turn": 1, "input_tokens": 62_000})
        console.emit(EventType.MODEL_RESPONDED, {"turn": 1, "output_tokens": 3_000})
        console.emit(EventType.TOOL_CALLED, {"tool_call_id": "1", "tool_name": "read_file"})
        console.emit(EventType.CONTEXT_COMPACTED, {"before_tokens": 82_000, "after_tokens": 51_000})
        console.emit(EventType.RUN_FINISHED, {"turns": 8})
        self.assertEqual(
            console.summary(model="deepseek-v4-flash", stream=self.output),
            "────────────────\n\nRun summary\nmodel: deepseek-v4-flash\nturns: 8\n"
            "tool calls: 1\ntokens: 62k input / 3k output\ncompactions: 1\nduration: 42s",
        )

    def test_verbose_preserves_every_event_name_without_raw_payloads(self):
        console = ConsoleEventLogger(verbose=True, stream=self.output)
        events = (
            (EventType.TOOL_CALLED, {"tool_call_id": "1", "tool_name": "read_file", "secret": "PRIVATE"}),
            (EventType.TOOL_STARTED, {"tool_call_id": "1", "tool_name": "read_file"}),
            (EventType.TOOL_RESULT, {"tool_call_id": "1", "tool_name": "read_file", "outcome": "returned"}),
            (EventType.CONTEXT_COMPACTED, {"before_tokens": 10, "after_tokens": 5}),
        )
        for event_type, data in events:
            console.emit(event_type, data)
        text = self.output.getvalue()
        for event_type, _ in events:
            self.assertIn(event_type.name, text)
        self.assertNotIn("PRIVATE", text)
        self.assertNotIn("Exploring", text)

    def test_cli_appends_summary_after_answer(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {"TINYHARNESS_API_KEY": "test", "TINYHARNESS_MODEL": "test-model"}), \
             patch("tiny_harness.__main__.ChatCompletionsProvider"), \
             patch("tiny_harness.__main__.AgentSession") as session:
            def submit(task):
                logger = session.call_args.kwargs["event_logger_factory"]()
                logger.emit(EventType.RUN_STARTED)
                logger.emit(EventType.MODEL_REQUESTED, {"context_tokens": 1_500})
                logger.emit(EventType.RUN_FINISHED, {"turns": 1})
                return "final answer"
            session.return_value.submit.side_effect = submit
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["task", "--workspace", folder]), 0)
        text = stdout.getvalue()
        self.assertTrue(text.startswith("final answer\n\n────────────────\n"))
        self.assertIn("model: test-model\nturns: 1\ntool calls: 0", text)
        self.assertIn("tokens: 1.5k (预估) input / ? output", text)

    def test_tty_styles_are_faint_and_redirects_are_plain(self):
        tty = io.StringIO()
        with patch.object(tty, "isatty", return_value=True), patch.dict(os.environ, {}, clear=True):
            console = ConsoleEventLogger(stream=tty)
            console.emit(EventType.TOOL_CALLED, {"tool_call_id": "1", "tool_name": "read_file"})
            console.emit(EventType.TOOL_RESULT, {"tool_call_id": "1", "tool_name": "read_file", "outcome": "returned"})
            console.emit(EventType.MODEL_REQUESTED)
        self.assertTrue(tty.getvalue().startswith("\x1b[2;36m"))
        self.assertNotIn("\x1b", self.output.getvalue())

    def test_console_io_failure_retains_event_log_error_contract(self):
        stream = io.StringIO()
        stream.close()
        logger = ConsoleEventLogger(stream=stream)
        with self.assertRaises(EventLogError):
            logger.emit(EventType.SUBAGENT_STARTED)

    def test_jsonl_remains_complete_and_has_no_terminal_color(self):
        tty = io.StringIO()
        with patch.object(tty, "isatty", return_value=True), \
             patch.dict(os.environ, {}, clear=True), \
             tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.jsonl"
            logger = CompositeEventLogger(ConsoleEventLogger(stream=tty), JsonlEventLogger(path))
            logger.emit(EventType.TOOL_CALLED, {"tool_call_id": "1", "tool_name": "read_file"})
            logger.emit(EventType.TOOL_STARTED, {"tool_call_id": "1", "tool_name": "read_file"})
            logger.emit(EventType.TOOL_RESULT, {"tool_call_id": "1", "tool_name": "read_file", "outcome": "returned"})
            logger.emit(EventType.RUN_FINISHED, {"turns": 1})
            raw = path.read_text(encoding="utf-8")
            events = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual([event["event_type"] for event in events], [
            "tool_called", "tool_started", "tool_result", "run_finished",
        ])
        self.assertNotIn("\x1b", raw)

    def test_quiet_suppresses_status_but_keeps_failures(self):
        console = ConsoleEventLogger(quiet=True, stream=self.output)
        console.emit(EventType.SUBAGENT_STARTED)
        console.emit(EventType.TOOL_CALLED, {"tool_call_id": "1", "tool_name": "broken"})
        console.emit(EventType.TOOL_RESULT, {"tool_call_id": "1", "tool_name": "broken",
                                                    "outcome": "error", "error_type": "ValueError"})
        self.assertEqual(self.output.getvalue(), "✗ broken failed: ValueError\n")
        self.assertEqual(console.summary(model="model", stream=self.output), "")

    def test_permission_prompt_never_dumps_content(self):
        with patch("builtins.input", return_value="no"), contextlib.redirect_stdout(self.output):
            _ask_permission("write_file", {"path": "a.py", "content": "SECRET" * 10000})
        self.assertNotIn("SECRET", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
