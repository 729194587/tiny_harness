import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.messages import ModelResponse
from tiny_harness.agent.turn import call_model, model_request_inputs
from tiny_harness.context.attribution import request_attribution
from tiny_harness.context.token_meter import CalibratedTokenMeter, DEFAULT_TOKEN_METER
from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.runtime.events import EventType
from tiny_harness.runtime.recovery import RecoveryExecutor, RecoveryPolicy, RecoveryState


def batch(name, content):
    return [{"role": "assistant", "tool_calls": [
        {"id": "reused", "type": "function", "function": {"name": name, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "reused", "content": content}]


class AttributionTest(unittest.TestCase):
    def test_assistant_breakdown_is_additive_safe_and_read_only(self):
        calls = batch("read_file", "unused")[0]["tool_calls"]
        calls = calls + [{"id": "second", "type": "function",
                          "function": {"name": "bash", "arguments": '{"command":"SECRET"}'}}]
        cases = [
            {"reasoning_content": "SECRET中文" * 100, "content": "visible" * 30,
             "tool_calls": calls},
            {"reasoning_content": "SECRET" * 100},
            {"content": "visible"},
            {"reasoning_content": "SECRET", "content": None, "tool_calls": calls},
            {"content": "", "tool_calls": calls},
            {"content": None},
        ]
        meter = DEFAULT_TOKEN_METER
        for fields in cases:
            with self.subTest(fields=list(fields)):
                messages = [{"role": "assistant", **fields}]
                original = copy.deepcopy(messages)
                result = request_attribution(messages, [], meter)
                breakdown = result["assistant_history_breakdown"]
                aggregate = result["categories"]["assistant_history"]["estimated_tokens"]
                self.assertEqual(aggregate, meter.estimate(messages, []) - meter.estimate([], []))
                self.assertEqual(sum(b["estimated_tokens"] for b in breakdown.values()), aggregate)
                self.assertEqual(result["estimated_tokens"], aggregate + result["envelope_and_rounding_tokens"])
                for field, name in (("reasoning_content", "reasoning_content"),
                                    ("content", "visible_content"), ("tool_calls", "tool_calls")):
                    value = breakdown[name]["estimated_tokens"]
                    if fields.get(field):
                        self.assertGreater(value, 0)
                    else:
                        self.assertEqual(value, 0)
                if fields.get("tool_calls"):
                    payload_only = {"role": "assistant", **fields}
                    payload_only.pop("reasoning_content", None)
                    if payload_only.get("content"):
                        payload_only.pop("content")
                    without_calls = {k: v for k, v in payload_only.items() if k != "tool_calls"}
                    self.assertEqual(breakdown["tool_calls"]["estimated_tokens"],
                                     meter.estimate([payload_only], []) - meter.estimate([without_calls], []))
                self.assertEqual(messages, original)
                self.assertNotIn("SECRET", json.dumps(result))
        combined = request_attribution([{"role": "assistant", **f} for f in cases], [])
        self.assertEqual(sum(b["estimated_tokens"] for b in combined["assistant_history_breakdown"].values()),
                         combined["categories"]["assistant_history"]["estimated_tokens"])
        self.assertEqual(request_attribution([], [])["assistant_history_breakdown"]["reasoning_content"]["estimated_tokens"], 0)

    def test_categories_totals_and_no_mutation(self):
        messages = [{"role": "system", "content": "SECRET"},
                    {"role": "user", "content": "SECRET"}]
        messages += batch("read_file", "SECRET" * 100)
        messages += batch("bash", "SECRET中文")
        for marker in ("working_memory", "environment_context", "skill_catalog"):
            messages.append({"role": "user", "name": "tinyharness_" + marker,
                             "content": "SECRET"})
        original = copy.deepcopy(messages)
        meter = CalibratedTokenMeter()
        meter.observe(messages, messages, [], 9999)
        result = request_attribution(messages, [], meter)
        self.assertEqual(messages, original)
        self.assertEqual(meter.estimate_request(messages, messages, []), 9999)
        self.assertNotIn("SECRET", json.dumps(result))
        categories = result["categories"]
        for key in ("system_runtime_guidance", "user_task_messages", "assistant_history",
                    "fresh_tool_results", "historical_tool_results", "working_memory_projection",
                    "environment_projection", "skill_projection"):
            self.assertIn(key, categories)
        self.assertEqual(categories["fresh_tool_results"]["count"], 1)
        self.assertEqual(result["tool_results_by_name"]["read_file"], categories["historical_tool_results"])
        self.assertEqual(result["tool_results_by_name"]["bash"], categories["fresh_tool_results"])
        self.assertEqual(result["estimated_tokens"], sum(b["estimated_tokens"] for b in categories.values())
                         + result["envelope_and_rounding_tokens"])

    def test_provider_request_unchanged_including_existing_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            for finalization in (False, True):
                logger = Mock()
                provider = Mock()
                provider.complete.return_value = ModelResponse("done", None, [], "stop")
                context = create_run_context(provider, Path(directory), event_logger=logger,
                                             working_memory_enabled=True)
                messages = [{"role": "user", "content": "task"}] + batch("read_file", "x" * 50000)
                messages[1]["reasoning_content"] = "SECRET reasoning replay"
                original = copy.deepcopy(messages)
                expected = copy.deepcopy(model_request_inputs(messages, context, finalization=finalization))
                call_model(messages, context, finalization=finalization)
                self.assertEqual(provider.complete.call_args.args, expected)
                self.assertEqual(messages, original)
                data = next(c.args[1] for c in logger.emit.call_args_list
                            if c.args[0] == EventType.MODEL_REQUESTED)
                expected_attribution = request_attribution(*expected)
                expected_attribution["calibration_adjustment_tokens"] = 0
                self.assertEqual(data["context_attribution"], expected_attribution)
                self.assertGreater(data["context_attribution"]["assistant_history_breakdown"]
                                   ["reasoning_content"]["estimated_tokens"], 0)
                self.assertNotIn("SECRET", json.dumps(data))

    def test_each_physical_retry_is_measured(self):
        provider = Mock()
        provider.complete.side_effect = [ModelProviderError(ModelErrorKind.CONNECTION, message="offline"),
                                        ModelResponse("done", None, [], "stop")]
        logger = Mock()
        executor = RecoveryExecutor(RecoveryPolicy(base_delay_seconds=0), event_logger=logger)
        executor.complete(provider, [], [], purpose="summary", turn=1, state=RecoveryState())
        requests = [c.args[1] for c in logger.emit.call_args_list if c.args[0] == EventType.MODEL_REQUESTED]
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["context_attribution"], requests[1]["context_attribution"])

    def test_unmatched_tool_is_unknown(self):
        result = request_attribution([{"role": "tool", "tool_call_id": "missing", "content": "x"}], [])
        self.assertIn("unknown", result["tool_results_by_name"])
