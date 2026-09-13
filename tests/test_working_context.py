import copy
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from tiny_harness.agent.context import create_run_context, run_started_data
from tiny_harness.agent.loop import agent_loop
from tiny_harness.agent.messages import ModelResponse
from tiny_harness.agent.session import AgentSession
from tiny_harness.agent.turn import (
    call_model, model_request_inputs, prepare_model_request_inputs,
)
from tiny_harness.context.attribution import request_attribution
from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.runtime.context import (
    CompactionConfig, ContextArtifactError, context_token_count, validate_active_request,
    PreparedContext,
)
from tiny_harness.runtime.events import EventLogError, EventType
from tiny_harness.runtime.recovery import RecoveryPolicy


def batch(index, size=12_000, count=1, name="bash"):
    calls = [{"id": f"{index}-{n}", "type": "function", "function": {
        "name": name, "arguments": "{}",
    }} for n in range(count)]
    return [{"role": "assistant", "content": f"investigation {index}", "tool_calls": calls},
            *[{"role": "tool", "tool_call_id": call["id"], "content": "x" * size}
              for call in calls]]


def history(batch_count=8, **kwargs):
    return [{"role": "user", "content": "task"}] + [
        message for index in range(batch_count) for message in batch(index, **kwargs)
    ]


def pruned_ids(messages):
    return [m["tool_call_id"] for m in messages if m.get("role") == "tool"
            and m["content"].startswith("<persisted-tool-result>\n")]


class WorkingContextTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.provider = Mock()
        self.provider.complete.return_value = ModelResponse("done", None, [], "stop")
        self.logger = Mock()
        self.context = create_run_context(
            self.provider, self.workspace, max_context_tokens=125_000,
            event_logger=self.logger, allow_subagent=False,
            recovery_policy=RecoveryPolicy(base_delay_seconds=0, jitter_ratio=0),
        )

    def request(self, messages):
        return prepare_model_request_inputs(messages, self.context, finalization=False)

    def artifacts(self):
        return list(self.workspace.glob(".tinyharness/context/tool-results/*.txt"))

    def test_below_trigger_does_not_prune(self):
        messages = history(4, size=4000)
        original = copy.deepcopy(messages)
        self.request(messages)
        self.assertEqual(messages, original)
        self.assertEqual(self.artifacts(), [])

    def test_projection_below_trigger_does_not_prune_canonical_large_results(self):
        messages = history(4, size=100_000, name="read_file")
        original = copy.deepcopy(messages)
        self.assertGreater(context_token_count(messages, self.context.tools), 20_000)
        request, tools = self.request(messages)
        self.assertLess(context_token_count(request, tools), 20_000)
        self.assertEqual(messages, original)
        self.assertEqual(self.artifacts(), [])

    def test_oldest_first_stops_at_target_and_protects_recent_three_batches(self):
        messages = history()
        original = copy.deepcopy(messages)
        request, tools = self.request(messages)
        selected = pruned_ids(messages)
        self.assertTrue(selected)
        self.assertEqual(selected, [f"{n}-0" for n in range(len(selected))])
        self.assertLessEqual(len(selected), 5)
        self.assertLessEqual(context_token_count(request, tools), 14_000)
        # Restoring just the last chosen result crosses target: pruning stopped early.
        last = selected[-1]
        previous = copy.deepcopy(request)
        next(m for m in previous if m.get("tool_call_id") == last)["content"] = "x" * 12_000
        self.assertGreater(context_token_count(previous, tools), 14_000)
        self.assertEqual(messages[-6:], original[-6:])
        self.assertEqual([m for m in messages if m["role"] != "tool"],
                         [m for m in original if m["role"] != "tool"])
        validate_active_request(request, "task")
        self.provider.complete.assert_not_called()
        self.assertEqual(len(self.artifacts()), len(selected))
        for artifact in self.artifacts():
            self.assertEqual(artifact.read_text(encoding="utf-8"), "x" * 12_000)
        preview = next(m["content"] for m in messages if m.get("tool_call_id") == last)
        self.assertIn("Content SHA-256: " + hashlib.sha256(("x" * 12_000).encode()).hexdigest(), preview)
        self.assertIn("Full output: .tinyharness/context/tool-results/", preview)
        self.assertIn("Head:", preview)
        self.assertIn("Tail:", preview)

    def test_multi_tool_batch_counts_once_and_exhaustion_can_exceed_target(self):
        messages = history(4, size=16_000, count=3)
        original = copy.deepcopy(messages)
        request, tools = self.request(messages)
        self.assertEqual(pruned_ids(messages), ["0-0", "0-1", "0-2"])
        self.assertEqual(messages[5:], original[5:])
        self.assertGreater(context_token_count(request, tools), 14_000)
        self.provider.complete.assert_not_called()

    def test_no_eligible_batches_leaves_large_request_unchanged_without_summary(self):
        messages = history(3, size=40_000)
        original = copy.deepcopy(messages)
        request, tools = self.request(messages)
        self.assertGreater(context_token_count(request, tools), 20_000)
        self.assertEqual(messages, original)
        self.assertEqual(self.artifacts(), [])
        self.provider.complete.assert_not_called()

    def test_trigger_includes_request_guidance_tools_and_working_memory(self):
        self.context.working_memory = Mock()
        self.context.working_memory.projection.return_value = {
            "role": "user", "name": "tinyharness_working_memory", "content": "state" * 1000,
        }
        messages = history(4, size=4000)
        request, tools = model_request_inputs(messages, self.context, finalization=False)
        threshold = self.context.token_meter.estimate_request(messages, request, tools)
        self.assertGreater(threshold, context_token_count(messages, tools))
        self.context.compactor.config = replace(
            self.context.compactor.config,
            working_context_trigger_tokens=threshold,
            working_context_target_tokens=threshold - 1,
        )
        self.request(messages)
        self.assertEqual(pruned_ids(messages), ["0-0"])

    def test_pruned_results_remain_stable_and_reused_call_ids_are_batch_local(self):
        messages = history(4, size=24_000)
        for message in messages:
            if message.get("tool_calls"):
                message["tool_calls"][0]["id"] = "reused"
            elif message["role"] == "tool":
                message["tool_call_id"] = "reused"
        self.context.compactor.config = replace(
            self.context.compactor.config, working_context_target_tokens=1000,
        )
        first, _ = self.request(messages)
        artifacts = self.artifacts()
        second, _ = self.request(messages)
        self.assertEqual(first, second)
        self.assertEqual(self.artifacts(), artifacts)
        self.assertEqual(pruned_ids(messages), ["reused"])
        self.assertEqual(messages[-1]["content"], "x" * 24_000)

    def test_persistence_failure_does_not_commit_partial_history(self):
        messages = history()
        original = copy.deepcopy(messages)
        persist = self.context.compactor._persist_tool_result
        calls = 0

        def failing_persist(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ContextArtifactError("failed")
            return persist(*args, **kwargs)

        with patch.object(self.context.compactor, "_persist_tool_result", side_effect=failing_persist):
            with self.assertRaises(ContextArtifactError):
                self.request(messages)
        self.assertEqual(messages, original)
        self.assertEqual(self.artifacts(), [])

    def test_partial_artifact_write_is_cleaned(self):
        original_open = Path.open

        class BrokenWriter:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def write(self, content):
                self.stream.write(content[:10])
                raise OSError("disk write failed")

            def __exit__(self, *args):
                self.stream.close()

        def open_file(path, mode="r", *args, **kwargs):
            stream = original_open(path, mode, *args, **kwargs)
            return BrokenWriter(stream) if mode == "x" else stream

        messages = history()
        original = copy.deepcopy(messages)
        with patch.object(Path, "open", open_file):
            with self.assertRaises(ContextArtifactError):
                self.request(messages)
        self.assertEqual(messages, original)
        self.assertEqual(self.artifacts(), [])

    def test_emit_failure_does_not_commit_pruned_canonical_history(self):
        for error_type in (EventLogError, RuntimeError):
            with self.subTest(error_type=error_type):
                messages = history()
                original = copy.deepcopy(messages)
                error = error_type("emit failed")

                def fail_emit(event_type, data):
                    self.assertEqual(event_type, EventType.CONTEXT_COMPACTED)
                    self.assertEqual(data["reason"], "working")
                    self.assertGreater(data["persisted_results"], 0)
                    self.assertEqual(messages, original)
                    raise error

                with patch.object(self.logger, "emit", side_effect=fail_emit) as emit:
                    with self.assertRaises(error_type) as raised:
                        call_model(messages, self.context)
                self.assertIs(raised.exception, error)
                emit.assert_called_once()
                self.assertEqual(messages, original)
                self.assertEqual(self.artifacts(), [])
                self.provider.complete.assert_not_called()

    def test_pruning_observability_includes_zero_change_and_batch_counts(self):
        for count, expected_results, expected_batches in ((3, 0, 0), (4, 3, 1)):
            with self.subTest(count=count):
                self.logger.reset_mock()
                self.context.current_turn = 7
                self.request(history(count, size=16_000, count=3))
                event = self.logger.emit.call_args.args[1]
                self.assertEqual(event["turn"], 7)
                self.assertEqual(event["pruned_results"], expected_results)
                self.assertEqual(event["pruned_batches"], expected_batches)
                self.assertFalse(event["target_reached"])
                self.assertTrue(event["blocked_by_recent_protection"])
                self.assertGreaterEqual(event["before_tokens"], event["after_tokens"])
                self.assertNotIn("x" * 100, str(event))

    def test_rejected_candidates_and_measurement_failure_clean_only_new_files(self):
        self.request(history())
        existing = set(self.artifacts())
        for fail in (False, True):
            messages = history()
            request, _ = model_request_inputs(messages, self.context, finalization=False)
            original, original_request = copy.deepcopy(messages), copy.deepcopy(request)
            measure = Mock(side_effect=[100_000, RuntimeError("measure failed")] if fail else None,
                           return_value=100_000)
            if fail:
                with self.assertRaises(RuntimeError):
                    self.context.compactor.prune_working_context(messages, request, measure)
            else:
                self.context.compactor.prune_working_context(messages, request, measure)
            self.assertEqual(messages, original)
            self.assertEqual(request, original_request)
            self.assertEqual(set(self.artifacts()), existing)

    def test_transient_retry_uses_same_pruned_request_and_attribution(self):
        messages = history()
        self.provider.complete.side_effect = [
            ModelProviderError(ModelErrorKind.CONNECTION, message="offline"),
            ModelResponse("done", None, [], "stop"),
        ]
        with patch.object(self.context.compactor, "prune_working_context",
                          wraps=self.context.compactor.prune_working_context) as prune:
            call_model(messages, self.context)
        self.assertEqual(prune.call_count, 1)
        calls = self.provider.complete.call_args_list
        self.assertEqual(calls[0].args, calls[1].args)
        request, tools = calls[0].args
        self.assertTrue(pruned_ids(request))
        expected = request_attribution(request, tools)
        events = [c.args[1] for c in self.logger.emit.call_args_list
                  if c.args[0] == EventType.MODEL_REQUESTED]
        self.assertEqual(len(events), 2)
        for event in events:
            attribution = dict(event["context_attribution"])
            self.assertEqual(attribution.pop("calibration_adjustment_tokens"), 0)
            self.assertEqual(attribution, expected)

    def test_loop_prepares_pruning_only_once_before_sending(self):
        messages = history()
        with patch.object(self.context.compactor, "prune_working_context",
                          wraps=self.context.compactor.prune_working_context) as prune:
            self.assertEqual(agent_loop(messages, self.context, "task"), "done")
        self.assertEqual(prune.call_count, 1)
        self.assertEqual(self.provider.complete.call_count, 1)
        self.assertTrue(pruned_ids(self.provider.complete.call_args.args[0]))

    def test_context_length_recovery_does_not_make_another_working_decision(self):
        messages = history()
        self.provider.complete.side_effect = [
            ModelProviderError(ModelErrorKind.CONTEXT_LENGTH, message="context full"),
            ModelResponse("done", None, [], "stop"),
        ]
        # Existing hard recovery owns the new history; working pruning must not
        # reselect results in the same logical request, even above its trigger.
        recovered = PreparedContext(messages=history(5, size=24_000),
                                    before_tokens=40_000, after_tokens=30_000)
        with (
            patch.object(self.context.compactor, "reactive_compact", return_value=recovered) as reactive,
            patch.object(self.context.compactor, "prune_working_context",
                         wraps=self.context.compactor.prune_working_context) as prune,
        ):
            call_model(messages, self.context)
        reactive.assert_called_once()
        prune.assert_called_once()
        self.assertFalse(pruned_ids(self.provider.complete.call_args.args[0]))

    def test_zero_recent_batch_configuration_allows_pruning_latest_batch(self):
        self.context.compactor.config = replace(
            self.context.compactor.config, keep_recent_tool_batches=0,
        )
        messages = history(1, size=100_000)
        request, tools = self.request(messages)
        self.assertEqual(pruned_ids(messages), ["0-0"])
        self.assertLessEqual(context_token_count(request, tools), 14_000)

    def test_configuration_validation_and_run_metadata(self):
        for kwargs in (
            {"working_context_target_tokens": 20_000},
            {"working_context_target_tokens": 0},
            {"keep_recent_tool_batches": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CompactionConfig(**kwargs)
        data = run_started_data(self.context)
        self.assertEqual(data["working_context_trigger_tokens"], 20_000)
        self.assertEqual(data["working_context_target_tokens"], 14_000)
        self.assertEqual(data["keep_recent_tool_batches"], 3)

    def test_session_and_subagent_forward_configuration(self):
        options = dict(working_context_trigger_tokens=18_000,
                       working_context_target_tokens=12_000, keep_recent_tool_batches=2)
        session = AgentSession(self.provider, self.workspace, "system", max_context_tokens=125_000,
                               **options)
        with patch("tiny_harness.agent.session.create_run_context", wraps=create_run_context) as create:
            session.submit("task")
        for name, value in options.items():
            self.assertEqual(create.call_args.kwargs[name], value)
        context = create_run_context(self.provider, self.workspace, max_context_tokens=125_000, **options)
        child = Mock(return_value="done")
        context.subagent_runner.run_agent = child
        context.subagent_runner("child task", "parent")
        for name, value in options.items():
            self.assertEqual(getattr(context.compactor.config, name), value)
            self.assertEqual(child.call_args.kwargs[name], value)
