"""Scripted full-loop regressions for working pressure, without external models."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evals.swe_bench_lite.report import analyze_run
from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.loop import agent_loop
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.context.attribution import request_attribution
from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.runtime.context import CompactionConfig, ContextArtifactError, context_token_count
from tiny_harness.runtime.events import EventLogError, EventType, JsonlEventLogger
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.runtime.recovery import RecoveryPolicy
from tiny_harness.runtime.skills import discover_skills
from tiny_harness.tools.definition import ToolDefinition
from tiny_harness.tools.registry import ToolRegistry
from test_working_context import history, pruned_ids


class WorkingContextInvariantTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.requests = []
        self.retry_done = False
        self.provider = Mock()
        self.provider.complete.side_effect = self.complete
        self.logger = JsonlEventLogger(self.root / "events.jsonl")
        self.context = create_run_context(
            self.provider, self.root, max_turns=27, max_context_tokens=125_000,
            allow_subagent=False, event_logger=self.logger,
            skill_catalog=discover_skills(self.root, sources=()),
            recovery_policy=RecoveryPolicy(base_delay_seconds=0, jitter_ratio=0),
        )
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="bash", description="scripted tool", parameters={"type": "object", "properties": {}},
            execute=lambda call, arguments: "x" * 8_000,
        ))
        self.context.tool_registry = registry
        self.context.compactor.tools = self.context.tools
        self.context.permission_policy = Mock()
        self.context.permission_policy.decide.return_value = PermissionDecision.ALLOW

    def complete(self, messages, tools, **kwargs):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        count = sum(m.get("role") == "tool" for m in messages)
        if count == 10 and not self.retry_done:
            self.retry_done = True
            raise ModelProviderError(ModelErrorKind.CONNECTION, message="offline retry")
        if count == 24:
            return ModelResponse("done", None, [], "stop")
        return ModelResponse(None, None, [ToolCall(str(count), "bash", "{}")], "tool_calls")

    def events(self):
        return [json.loads(line) for line in (self.root / "events.jsonl").read_text().splitlines()]

    def artifacts(self):
        return list(self.root.glob(".tinyharness/context/tool-results/*.txt"))

    def test_repeated_pressure_cycles_stable_history_protection_retry_and_reporting(self):
        messages = [{"role": "user", "content": "task"}]
        with patch.object(self.context.compactor, "prune_working_context",
                          wraps=self.context.compactor.prune_working_context) as prune:
            self.assertEqual(agent_loop(messages, self.context, "task"), "done")
        events = self.events()
        working = [e["data"] for e in events if e["event_type"] == "context_compacted"
                   and e["data"]["reason"] == "working"]
        self.assertGreaterEqual(len(working), 3)
        self.assertGreaterEqual(sum(event["target_reached"] for event in working), 2)
        for event in working:
            self.assertGreaterEqual(event["before_tokens"], 20_000)
            if event["target_reached"]:
                self.assertLessEqual(event["after_tokens"], 14_000)
                self.assertFalse(event["blocked_by_recent_protection"])
            else:
                self.assertTrue(event["blocked_by_recent_protection"])
        stable = {}
        for request, tools in self.requests:
            results = [m for m in request if m.get("role") == "tool"]
            self.assertFalse(pruned_ids(results[-3:]))
            for result in results:
                call_id = result["tool_call_id"]
                if call_id in stable:
                    self.assertEqual(result["content"], stable[call_id])
                if call_id in pruned_ids([result]):
                    stable[call_id] = result["content"]
        self.assertEqual(len(self.artifacts()), len(stable))
        self.assertEqual(prune.call_count, 25)  # 24 tool turns and one final answer.
        self.assertEqual(len(self.requests), 26)  # One provider retry, no extra pruning.
        self.assertEqual(self.requests[10], self.requests[11])
        requested = [e["data"] for e in events if e["event_type"] == "model_requested"]
        for event, (request, tools) in zip(requested, self.requests, strict=True):
            attribution = dict(event["context_attribution"])
            self.assertEqual(attribution.pop("calibration_adjustment_tokens"), 0)
            self.assertEqual(attribution, request_attribution(request, tools))
            self.assertEqual(event["context_tokens"], context_token_count(request, tools))
        report = analyze_run(self.root)
        self.assertEqual(report["turns"], 25)
        self.assertEqual(report["working_context_prune_events"], len(working))
        self.assertEqual(report["context_attribution_summary"]["requests"], len(self.requests))
        self.assertEqual(report["peak_request_context_tokens"], max(
            event["context_tokens"] for event in requested))
        self.assertEqual(report["peak_pre_prune_context_tokens"], max(
            event["before_tokens"] for event in working))
        self.assertEqual(report["context_attribution_summary"]["categories"]["historical_tool_results"]["sum_estimated_tokens"],
                         sum(e["context_attribution"]["categories"].get("historical_tool_results", {}).get("estimated_tokens", 0)
                             for e in requested))

    def test_hard_pressure_can_replace_working_protected_results(self):
        messages = history(3, size=200_000)
        protected = {m["tool_call_id"] for m in messages if m.get("role") == "tool"}
        self.provider.complete.side_effect = None
        self.provider.complete.return_value = ModelResponse("done", None, [], "stop")
        self.assertEqual(agent_loop(messages, self.context, "task"), "done")
        self.assertTrue(protected.intersection(pruned_ids(messages)))
        self.assertTrue(any(e["event_type"] == "context_compacted" and e["data"]["reason"] == "automatic"
                            for e in self.events()))
        self.assertLessEqual(context_token_count(messages, self.context.tools), self.context.compactor.soft_limit)
        self.provider.complete.assert_called_once()

    def test_report_attribution_uses_read_file_projection_not_canonical_body(self):
        messages = history(4, size=40_000, name="read_file")
        self.provider.complete.side_effect = None
        self.provider.complete.return_value = ModelResponse("done", None, [], "stop")
        self.assertEqual(agent_loop(messages, self.context, "task"), "done")
        request, tools = self.provider.complete.call_args.args
        requested = next(e["data"] for e in self.events() if e["event_type"] == "model_requested")
        self.assertEqual(requested["context_tokens"], context_token_count(request, tools))
        self.assertGreater(context_token_count(messages, tools), requested["context_tokens"])
        self.assertTrue(any("<read-file-preview>" in m.get("content", "") for m in request))
        self.assertEqual(self.artifacts(), [])
        report = analyze_run(self.root)
        self.assertEqual(report["peak_request_context_tokens"], requested["context_tokens"])
        self.assertIsNone(report["peak_pre_prune_context_tokens"])
        self.assertEqual(report["context_attribution_summary"]["categories"]["historical_tool_results"]["sum_estimated_tokens"],
                         requested["context_attribution"]["categories"]["historical_tool_results"]["estimated_tokens"])

    def test_loop_failures_do_not_commit_pruning_or_leave_artifacts(self):
        for kind in ("event", "persistence"):
            with self.subTest(kind=kind):
                messages = history()
                original = {}
                prune = self.context.compactor.prune_working_context
                def capture(canonical, request, *args, **kwargs):
                    original["canonical"] = copy.deepcopy(canonical)
                    original["request"] = copy.deepcopy(request)
                    try:
                        return prune(canonical, request, *args, **kwargs)
                    finally:
                        self.assertEqual(canonical, original["canonical"])
                        self.assertEqual(request, original["request"])
                persist = self.context.compactor._persist_tool_result
                written = 0
                def fail_persist(*args, **kwargs):
                    nonlocal written
                    written += 1
                    if written == 2:
                        raise ContextArtifactError("failed")
                    return persist(*args, **kwargs)
                emit = self.logger.emit
                def fail_emit(event_type, data=None):
                    if event_type == EventType.CONTEXT_COMPACTED and data.get("reason") == "working":
                        raise EventLogError("failed")
                    return emit(event_type, data)
                with patch.object(self.context.compactor, "prune_working_context", side_effect=capture):
                    target, name, failure = ((self.logger, "emit", fail_emit) if kind == "event" else
                                             (self.context.compactor, "_persist_tool_result", fail_persist))
                    with patch.object(target, name, side_effect=failure):
                        with self.assertRaises((ContextArtifactError, EventLogError)):
                            agent_loop(messages, self.context, "task")
                self.assertEqual(self.artifacts(), [])
                self.provider.complete.assert_not_called()

    def test_configuration_relationships(self):
        for hard, trigger, target, recent in (
            (20_000, 20_000, 14_000, 3), (19_999, 20_000, 14_000, 3),
            (125_000, 20_000, 20_000, 3), (125_000, 20_000, 14_000, -1),
        ):
            with self.subTest(hard=hard, trigger=trigger, target=target, recent=recent):
                with self.assertRaises(ValueError):
                    create_run_context(Mock(), self.root, max_context_tokens=hard,
                                       working_context_trigger_tokens=trigger,
                                       working_context_target_tokens=target, keep_recent_tool_batches=recent)
        self.assertEqual(CompactionConfig().working_context_trigger_tokens, 20_000)
        self.assertIsNone(create_run_context(Mock(), self.root, max_context_tokens=None).compactor)
