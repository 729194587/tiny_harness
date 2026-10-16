"""Offline coverage for opt-in visible Assistant response logging."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.swe_bench_lite.__main__ import _parser
from evals.swe_bench_lite.report import analyze_run
from evals.swe_bench_lite.pipeline import rollout_task
from test_swe_bench_pipeline import FakeEnvironment, task
from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.runtime.events import JsonlEventLogger
from tiny_harness.runtime.recovery import RecoveryState
from tiny_harness.runtime.tool_trace import ToolTraceConfig


class AssistantContentTest(unittest.TestCase):
    def test_content_capture_preserves_requests_events_and_legacy_report(self):
        for content, calls in (
            (None, []), ("", []), ("修复判断：你好 🌍\nnext", []),
            ("Inspect before editing", [ToolCall("call-1", "read_file", '{"path":"x"}')]),
        ):
            with self.subTest(content=content, calls=calls), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                response = ModelResponse(content, "HIDDEN_REASONING_SENTINEL", calls,
                                         "tool_calls" if calls else "stop")
                requests = []

                class Provider:
                    def complete(self, messages, tools):
                        requests.append((messages, tools))
                        return response

                events = []
                reports = []
                for enabled in (False, True):
                    log_path = root / str(enabled) / "events.jsonl"
                    context = create_run_context(
                        Provider(), root,
                        event_logger=JsonlEventLogger(log_path),
                        tool_trace=ToolTraceConfig(capture_assistant_content=enabled),
                    )
                    messages = [{"role": "user", "content": "PROMPT_SENTINEL"}]
                    result = context.recovery_executor.complete(
                        context.provider, messages, [], purpose="main", turn=32,
                        state=RecoveryState(),
                    )
                    self.assertIs(result, response)
                    raw = log_path.read_text(encoding="utf-8")
                    self.assertNotIn("HIDDEN_REASONING_SENTINEL", raw)
                    self.assertNotIn("PROMPT_SENTINEL", raw)
                    recorded = [json.loads(line) for line in raw.splitlines()]
                    self.assertNotIn("reasoning_content", recorded[-1]["data"])
                    events.append([(event["event_type"], event["data"]) for event in recorded])
                    report = analyze_run(log_path.parent)
                    report.pop("run_dir")
                    reports.append(report)
                self.assertEqual(requests[0], requests[1])
                self.assertEqual(reports[0], reports[1])
                captured = events[1][-1][1]
                self.assertEqual(captured.pop("content"), content)
                self.assertEqual(captured["turn"], 32)
                self.assertEqual(captured["attempt"], 1)
                self.assertEqual(captured["tool_call_count"], len(calls))
                self.assertEqual(events[0], events[1])

    def test_auxiliary_response_content_is_not_captured(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            class Provider:
                def complete(self, messages, tools):
                    return ModelResponse("PRIVATE_SUMMARY", "PRIVATE_REASONING", [], "stop")

            context = create_run_context(
                Provider(), root, event_logger=JsonlEventLogger(root / "events.jsonl"),
                tool_trace=ToolTraceConfig(capture_assistant_content=True),
            )
            context.recovery_executor.complete(
                context.provider, [], [], purpose="summary", turn=1, state=RecoveryState(),
            )
            raw = (root / "events.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("PRIVATE_", raw)
            self.assertNotIn('"content":', raw)

    def test_subagent_does_not_capture_content_but_parent_still_does(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            class Provider:
                def complete(self, messages, tools):
                    return ModelResponse("visible answer", "HIDDEN_REASONING", [], "stop")

            config = ToolTraceConfig(
                enabled=True, result_preview_chars=37, capture_assistant_content=True,
            )
            context = create_run_context(
                Provider(), root, event_logger=JsonlEventLogger(root / "events.jsonl"),
                tool_trace=config,
            )
            self.assertEqual(context.subagent_runner("delegated work", "task-1"), "visible answer")
            context.recovery_executor.complete(
                context.provider, [], [], purpose="main", turn=1, state=RecoveryState(),
            )
            events = [json.loads(line) for line in
                      (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            responses = [event["data"] for event in events if event["event_type"] == "model_responded"]
            self.assertEqual(len(responses), 2)
            child, parent = responses
            self.assertEqual(child["purpose"], "main")
            self.assertEqual(child["parent_tool_call_id"], "task-1")
            self.assertNotIn("content", child)
            self.assertEqual(parent["content"], "visible answer")
            self.assertIs(context.tool_trace, config)
            self.assertTrue(config.capture_assistant_content)

    def test_rollout_cli_switch_and_config_validation(self):
        self.assertFalse(_parser().parse_args(["run"]).capture_assistant_content)
        self.assertTrue(_parser().parse_args([
            "run", "--capture-assistant-content",
        ]).capture_assistant_content)
        with self.assertRaises(ValueError):
            ToolTraceConfig(capture_assistant_content="true")

    def test_rollout_passes_capture_flag_without_changing_tool_previews(self):
        observed = []

        def agent(provider, workspace, messages, **options):
            observed.append(options["tool_trace"])
            return "done"

        with tempfile.TemporaryDirectory() as directory, patch(
            "evals.swe_bench_lite.pipeline.source_metadata", return_value={},
        ):
            rollout_task(
                task(), object(), Path(directory), model_name_or_path="offline",
                environment_factory=FakeEnvironment, agent_entrypoint=agent,
                capture_assistant_content=True,
            )
        self.assertEqual(observed, [ToolTraceConfig(
            enabled=True, result_preview_chars=200, capture_assistant_content=True,
        )])
