"""Scripted full-loop regressions for context continuity and pressure."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from evals.swe_bench_lite.report import analyze_run
from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.loop import agent_loop
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.runtime.context import context_token_count
from tiny_harness.runtime.events import JsonlEventLogger
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.runtime.recovery import RecoveryPolicy
from tiny_harness.runtime.skills import discover_skills
from tiny_harness.tools.definition import ToolDefinition
from tiny_harness.tools.registry import ToolRegistry
from test_context_policy import history


class ContextInvariantTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.requests = []
        self.retry_done = False
        self.provider = Mock()
        self.provider.supports_tool_choice = True
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
        count = self.context.current_turn
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

    def test_normal_turns_preserve_history_above_old_working_threshold(self):
        messages = [{"role": "user", "content": "task"}]
        self.assertEqual(agent_loop(messages, self.context, "task"), "done")
        self.assertGreater(context_token_count(messages, self.context.tools), 20_000)
        self.assertLess(context_token_count(messages, self.context.tools), self.context.compactor.soft_limit)
        self.assertFalse(any(e["event_type"] in (
            "context_compacted", "context_summary_requested", "context_compaction_skipped",
        ) for e in self.events()))
        for previous, current in zip(self.requests, self.requests[1:]):
            prior, tools = previous
            request, schemas = current
            self.assertEqual(tools, schemas)
            self.assertEqual(request[:len(prior)], prior)
        self.assertTrue(self.retry_done)
        self.assertEqual(self.artifacts(), [])
        self.assertFalse(list(self.root.glob(".tinyharness/context/transcripts/*")))
        report = analyze_run(self.root)
        self.assertEqual(report["turns"], 24)
        self.assertEqual(report["working_context_prune_events"], 0)

    def test_hard_pressure_persists_large_results(self):
        messages = history(3, size=200_000)
        protected = {m["tool_call_id"] for m in messages if m.get("role") == "tool"}
        self.provider.complete.side_effect = None
        self.provider.complete.return_value = ModelResponse("done", None, [], "stop")
        self.assertEqual(agent_loop(messages, self.context, "task"), "done")
        self.assertTrue(protected.intersection(
            m["tool_call_id"] for m in messages if m.get("role") == "tool"
            and m["content"].startswith("<persisted-tool-result>\n")))
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


    def test_hard_budget_no_longer_requires_working_threshold_relationship(self):
        for hard in (2_000, 14_000, 20_000, 125_000):
            with self.subTest(hard=hard):
                context = create_run_context(Mock(), self.root, max_context_tokens=hard)
                self.assertEqual(context.compactor.max_tokens, hard)
        self.assertIsNone(create_run_context(Mock(), self.root, max_context_tokens=None).compactor)
