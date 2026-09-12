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
from tiny_harness.context.token_meter import CalibratedTokenMeter
from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.runtime.events import EventType
from tiny_harness.runtime.recovery import RecoveryExecutor, RecoveryPolicy, RecoveryState


def batch(name, content):
    return [{"role": "assistant", "tool_calls": [
        {"id": "reused", "type": "function", "function": {"name": name, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "reused", "content": content}]


class AttributionTest(unittest.TestCase):
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
