"""Offline cache accounting from compatible transport through events and reports."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from evals.swe_bench_lite.report import analyze_run
from tiny_harness.agent.messages import ModelResponse
from tiny_harness.models.chat_completions import ChatCompletionsProvider
from tiny_harness.runtime.console import ConsoleEventLogger
from tiny_harness.runtime.events import EventType, JsonlEventLogger
from tiny_harness.runtime.recovery import RecoveryExecutor, RecoveryState


class CacheUsageTest(unittest.TestCase):
    def test_provider_optional_cache_fields_and_invalid_values(self):
        for hit, miss in ((80, 20), (0, 100), (80, None), (None, None), (-1, True)):
            with self.subTest(hit=hit, miss=miss):
                usage = SimpleNamespace(prompt_tokens=123)
                for name, value in (("prompt_cache_hit_tokens", hit), ("prompt_cache_miss_tokens", miss)):
                    if value is not None:
                        setattr(usage, name, value)
                client = Mock()
                client.chat.completions.create.return_value = SimpleNamespace(
                    usage=usage, choices=[SimpleNamespace(finish_reason="stop",
                        message=SimpleNamespace(content="done", tool_calls=None))],
                )
                response = ChatCompletionsProvider("fake", "https://example.test", "fake", client=client).complete([], [])
                self.assertEqual(response.prompt_tokens, 123)
                self.assertEqual(response.prompt_cache_hit_tokens, hit if type(hit) is int and hit >= 0 else None)
                self.assertEqual(response.prompt_cache_miss_tokens, miss if type(miss) is int and miss >= 0 else None)

    def test_events_console_and_weighted_report_include_auxiliary_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logger = JsonlEventLogger(root / "events.jsonl")
            executor = RecoveryExecutor(event_logger=logger)
            for purpose, hit, miss in (("main", 90, 10), ("summary", 0, 900)):
                provider = Mock()
                provider.complete.return_value = ModelResponse(
                    "done", None, [], "stop", prompt_tokens=hit + miss,
                    prompt_cache_hit_tokens=hit, prompt_cache_miss_tokens=miss,
                )
                executor.complete(provider, [], [], purpose=purpose, turn=1, state=RecoveryState())
            report = analyze_run(root)
            self.assertEqual(report["total_prompt_tokens"], 1000)
            self.assertEqual(report["total_cache_hit_tokens"], 90)
            self.assertEqual(report["total_cache_miss_tokens"], 910)
            self.assertEqual(report["overall_cache_hit_rate"], .09)
            self.assertEqual(report["prompt_tokens"], 1000)
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            responses = [e["data"] for e in events if e["event_type"] == EventType.MODEL_RESPONDED]
            self.assertEqual([d["cache_hit_rate"] for d in responses], [.9, 0])
            for verbose in (False, True):
                output = io.StringIO()
                console = ConsoleEventLogger(stream=output, verbose=verbose)
                console.emit(EventType.MODEL_RESPONDED, responses[0])
                self.assertIn("90", output.getvalue())
                self.assertIn("10", output.getvalue())
                self.assertIn("cache", output.getvalue())
                self.assertIn("cache_hit_rate=0.9" if verbose else "90.0%", output.getvalue())

    def test_missing_partial_and_zero_cache_usage_remain_unknown(self):
        for hit, miss in ((None, None), (10, None), (0, 0)):
            with self.subTest(hit=hit, miss=miss), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                logger = JsonlEventLogger(root / "events.jsonl")
                response = ModelResponse("done", None, [], "stop", prompt_tokens=20,
                                         prompt_cache_hit_tokens=hit, prompt_cache_miss_tokens=miss)
                provider = Mock()
                provider.complete.return_value = response
                spy = Mock(wraps=logger)
                RecoveryExecutor(event_logger=spy).complete(provider, [], [], purpose="main", turn=1, state=RecoveryState())
                data = spy.emit.call_args.args[1]
                self.assertNotIn("cache_hit_rate", data)
                if miss is None:
                    self.assertNotIn("prompt_cache_miss_tokens", data)
                report = analyze_run(root)
                self.assertEqual(report["total_cache_hit_tokens"], hit)
                self.assertEqual(report["total_cache_miss_tokens"], miss)
                self.assertIsNone(report["overall_cache_hit_rate"])
