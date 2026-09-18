import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tiny_harness.__main__ import main, _ask_permission
from tiny_harness.runtime.console import ConsoleEventLogger
from tiny_harness.runtime.events import EventLogError, CompositeEventLogger, JsonlEventLogger, EventType


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
            "tool calls: 1\ntokens: 62k input / 3k output\ncache: n/a\ncompactions: 1\nduration: 42s",
        )

    def test_summary_cache_rate_uses_aggregate_tokens(self):
        for verbose in (False, True):
            with self.subTest(verbose=verbose):
                console = ConsoleEventLogger(stream=self.output, verbose=verbose)
                console.emit(EventType.RUN_STARTED)
                for turn, hit, miss in ((1, 90, 10), (2, 100, 800)):
                    console.emit(EventType.MODEL_REQUESTED, {"turn": turn})
                    console.emit(EventType.MODEL_RESPONDED, {
                        "turn": turn, "prompt_cache_hit_tokens": hit,
                        "prompt_cache_miss_tokens": miss,
                        "cache_hit_rate": hit / (hit + miss),
                    })
                console.emit(EventType.RUN_FINISHED, {"turns": 2})
                summary = console.summary(model="test", stream=self.output)
                self.assertIn("\ncache: 19.0% hit\n", summary)
                self.assertEqual(summary.count("cache:"), 1)
                self.assertNotIn("\x1b", summary)
                console.emit(EventType.RUN_STARTED)
                console.emit(EventType.RUN_FINISHED, {"turns": 0})
                self.assertIn("\ncache: n/a\n", console.summary(model="test", stream=self.output))

    def test_summary_cache_unavailable_or_zero(self):
        for usage in ({}, {"cache_hit_rate": .91},
                      {"prompt_cache_hit_tokens": 10},
                      {"prompt_cache_miss_tokens": 10},
                      {"prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0}):
            with self.subTest(usage=usage):
                self.console.emit(EventType.RUN_STARTED)
                self.console.emit(EventType.MODEL_RESPONDED, usage)
                self.console.emit(EventType.RUN_FINISHED, {"turns": 1})
                self.assertIn("\ncache: n/a\n", self.console.summary(model="test", stream=self.output))

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
        self.assertIn("\x1b[2;36m", tty.getvalue())
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


class LiveConsoleTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.tty = patch.object(self.output, "isatty", return_value=True)
        self.tty.start()
        self.addCleanup(self.tty.stop)
        self.env = patch.dict(os.environ, {"NO_COLOR": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.console = ConsoleEventLogger(stream=self.output)

    def test_updates_replace_one_row_and_completion_erases_it(self):
        self.console.emit(EventType.RUN_STARTED)
        for turn in range(1, 21):
            self.console.emit(EventType.MODEL_REQUESTED, {"turn": turn, "context_tokens": 12000})
            self.console.emit(EventType.MODEL_RESPONDED, {"prompt_tokens": 12000, "cache_hit_rate": .91})
        raw = self.output.getvalue()
        self.assertNotIn("\n", raw)
        self.assertEqual(raw.count("\r\x1b[2K"), 40)
        self.assertIn("命中率 91.0%", raw)
        self.assertNotIn("\x1b[2;90m", raw)
        self.console.emit(EventType.RUN_FINISHED, {"turns": 20})
        self.assertTrue(self.output.getvalue().endswith("\r\x1b[2K"))
        self.assertFalse(self.console._live_visible)
        print("助手> final", file=self.output)
        print(self.console.summary(model="test", stream=self.output), file=self.output)
        # Each erase replaces the current row; only the final answer/summary
        # survives in the terminal after the last erase.
        visible = self.output.getvalue().rsplit("\r\x1b[2K", 1)[1]
        self.assertTrue(visible.startswith("助手> final\n"))
        self.assertIn("turns: 20", visible)
        self.assertIn("cache: n/a", visible)
        self.assertNotIn("命中率", visible)

    def test_permission_suspends_and_resumes_even_on_eof(self):
        self.console.emit(EventType.MODEL_REQUESTED, {"turn": 1})
        def read_prompt(prompt):
            self.assertFalse(self.console._live_visible)
            self.assertTrue(self.console._suspended)
            self.assertIn("需要工具授权", self.output.getvalue())
            raise EOFError
        with contextlib.redirect_stdout(self.output), patch("builtins.input", side_effect=read_prompt):
            with self.console.suspend_live():
                self.assertFalse(_ask_permission("read_file", {"path": "a.py"}))
        self.assertTrue(self.console._live_visible)
        self.assertFalse(self.console._suspended)
        self.assertIn("\r\x1b[2K\n需要工具授权", self.output.getvalue())

    def test_fatal_error_is_persistent_in_normal_and_verbose_modes(self):
        for verbose in (False, True):
            self.output.seek(0)
            self.output.truncate()
            console = ConsoleEventLogger(stream=self.output, verbose=verbose)
            console.emit(EventType.MODEL_REQUESTED, {"turn": 1})
            console.emit(EventType.RUN_FAILED, {"error_type": "ValueError"})
            visible = self.output.getvalue().rsplit("\r\x1b[2K", 1)[1]
            self.assertIn("ValueError", visible)
            self.assertTrue(visible.endswith("\n"))
            self.assertTrue(console.run_failure_reported)
            self.assertFalse(console._live_visible)

    def test_narrow_terminal_does_not_wrap_cjk_status(self):
        with patch.object(self.output, "fileno", return_value=2), \
             patch("os.get_terminal_size", return_value=os.terminal_size((12, 24))):
            self.console.emit(EventType.TOOL_STARTED, {"tool_name": "很长的工具名称"})
        visible = self.output.getvalue().split("\x1b[2K")[-1]
        self.assertLessEqual(len(visible) * 2, 11)
        self.assertNotIn("\n", visible)


if __name__ == "__main__":
    unittest.main()
