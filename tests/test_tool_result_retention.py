import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.loop import run_agent
from tiny_harness.agent.messages import ModelResponse, ToolCall, assistant_message_from_response
from tiny_harness.agent.tool_batch import execute_tool_batch
from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.runtime.context import (
    CompactionConfig, ContextArtifacts, retain_tool_result, validate_active_request,
)
from tiny_harness.runtime.events import EventLogError, EventType
from tiny_harness.runtime.hooks import ToolHooks
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.runtime.recovery import RecoveryPolicy
from tiny_harness.tools.definition import ToolDefinition
from tiny_harness.tools.registry import dispatch


class ToolResultRetentionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.content = "HEAD\r\n" + "private body\n" * 8000 + "\rTAIL\n"
        self.call = ToolCall("large", "bash", '{"command":"inspect"}')
        self.logger = Mock()
        self.runner = Mock()
        self.runner.run.return_value = self.content
        self.policy = Mock()
        self.policy.decide.return_value = PermissionDecision.ALLOW

    def artifacts(self):
        return list(self.workspace.glob(".tinyharness/context/tool-results/*.txt"))

    def retain(self, content=None, call=None):
        return retain_tool_result(
            self.workspace, call or self.call,
            self.content if content is None else content, turn=2, event_logger=self.logger,
        )

    def context(self, **kwargs):
        return create_run_context(
            Mock(), self.workspace, shell_runner=self.runner,
            permission_policy=self.policy, event_logger=self.logger, **kwargs,
        )

    def test_oversized_spill_recovers_exact_text_and_has_safe_metadata_and_guidance(self):
        retained = self.retain()
        self.assertTrue(retained.startswith("<persisted-tool-result>\n"))
        self.assertLess(len(retained), 3000)
        self.assertIn(self.content[:1000], retained)
        self.assertIn(self.content[-1000:], retained)
        self.assertIn("bounded line range", retained)
        artifacts = self.artifacts()
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].read_bytes().decode("utf-8"), self.content)
        relative = artifacts[0].relative_to(self.workspace).as_posix()
        self.assertIn("Full output: " + relative, retained)
        event, data = self.logger.emit.call_args.args
        self.assertEqual(event, EventType.TOOL_RESULT_RETAINED)
        self.assertEqual(data, {
            "turn": 2, "tool_call_id": "large", "tool_name": "bash",
            "original_chars": len(self.content), "retained_chars": len(retained),
            "outcome": "spilled", "artifact_path": relative,
            "content_hash": hashlib.sha256(self.content.encode("utf-8")).hexdigest(),
        })
        self.assertNotIn("private body", json.dumps(data))

    def test_small_results_and_threshold_boundary_remain_unchanged(self):
        for content in ("", "small\r\nresult", "x" * CompactionConfig.large_result_chars):
            with self.subTest(length=len(content)):
                self.assertIs(self.retain(content), content)
        self.assertEqual(self.artifacts(), [])
        self.logger.emit.assert_not_called()
        self.assertLess(len(self.retain("x" * (CompactionConfig.large_result_chars + 1))), 3000)

    def test_already_spilled_and_identical_raw_outputs_reuse_locator(self):
        retained = self.retain()
        original_artifacts = self.artifacts()
        self.logger.reset_mock()
        self.assertEqual(self.retain(retained), retained)
        self.logger.emit.assert_not_called()
        again = self.retain(call=ToolCall("again", "general", "{}"))
        locator = "Full output: " + original_artifacts[0].relative_to(self.workspace).as_posix()
        self.assertIn(locator, again)
        self.assertEqual(self.artifacts(), original_artifacts)
        self.assertEqual(self.logger.emit.call_args.args[1]["outcome"], "reused")

    def test_direct_artifact_read_reuses_original_locator_without_nesting(self):
        self.retain()
        artifact = self.artifacts()[0]
        call = ToolCall("read", "read_file", json.dumps({"path": str(artifact)}))
        messages = []
        execute_tool_batch(messages, [call], self.context())
        self.assertEqual(self.artifacts(), [artifact])
        self.assertIn("Full output: " + artifact.relative_to(self.workspace).as_posix(),
                      messages[0]["content"])
        self.assertLess(len(messages[0]["content"]), 3000)
        self.assertEqual(self.logger.emit.call_args.args[1]["outcome"], "reused")

    def test_persistence_failure_keeps_raw_success_and_batch_continues(self):
        context = self.context()
        messages = []
        with patch.object(ContextArtifacts, "_persist_tool_result", side_effect=OSError("disk full")):
            execute_tool_batch(messages, [self.call, ToolCall("next", "list_files", "{}")], context)
        self.assertEqual(messages[0]["content"], self.content)
        self.assertEqual([m["tool_call_id"] for m in messages], ["large", "next"])
        result_event = next(c.args[1] for c in self.logger.emit.call_args_list
                            if c.args[0] == EventType.TOOL_RESULT and c.args[1]["tool_call_id"] == "large")
        self.assertEqual(result_event["outcome"], "returned")
        retained_event = next(c.args[1] for c in self.logger.emit.call_args_list
                              if c.args[0] == EventType.TOOL_RESULT_RETAINED)
        self.assertEqual(retained_event["outcome"], "persistence_failed")
        self.assertEqual(retained_event["retained_chars"], len(self.content))
        self.assertNotIn("disk full", json.dumps(retained_event))

    def test_partial_write_cleanup_retains_existing_artifacts(self):
        self.retain()
        original_artifacts = self.artifacts()
        persist = ContextArtifacts._persist_tool_result

        def fail_after_write(storage, *args, **kwargs):
            persist(storage, *args, **kwargs)
            raise OSError("write failed")

        content = self.content + "different result"
        with patch.object(ContextArtifacts, "_persist_tool_result", new=fail_after_write):
            self.assertEqual(self.retain(content), content)
        self.assertEqual(self.artifacts(), original_artifacts)

    def test_hooks_dispatch_and_history_order_preserved_for_general_results(self):
        seen = []
        hooks = ToolHooks()
        hooks.register_post(lambda call, result: seen.append(result.content))
        context = self.context(tool_hooks=hooks)
        result = dispatch(context.tool_registry, self.call, permission_policy=self.policy,
                          tool_hooks=hooks)
        self.assertEqual(result.content, self.content)
        self.assertEqual(seen, [self.content])
        self.assertEqual(self.artifacts(), [])
        context.tool_registry.register(ToolDefinition(
            name="general", description="general output", parameters={"type": "object"},
            execute=lambda call, arguments: self.content + "extra",
        ))
        calls = [self.call, ToolCall("general", "general", "{}"),
                 ToolCall("small", "list_files", "{}")]
        messages = [{"role": "user", "content": "task"},
                    assistant_message_from_response(ModelResponse(None, None, calls, "tool_calls"))]
        assistant = copy.deepcopy(messages[1])
        execute_tool_batch(messages, calls, context)
        self.assertEqual(messages[1], assistant)
        self.assertEqual([m["tool_call_id"] for m in messages[2:]], ["large", "general", "small"])
        self.assertEqual(seen[1:3], [self.content, self.content + "extra"])
        self.assertLess(len(messages[2]["content"]), 3000)
        self.assertLess(len(messages[3]["content"]), 3000)
        validate_active_request(messages, "task")
        events = [c.args[0] for c in self.logger.emit.call_args_list]
        self.assertLess(events.index(EventType.TOOL_RESULT), events.index(EventType.TOOL_RESULT_RETAINED))
        self.assertEqual(events[:3], [EventType.TOOL_CALLED] * 3)

    def test_event_failure_cleans_new_artifacts_without_deleting_reused_artifacts(self):
        self.retain()
        existing = self.artifacts()
        for error_type in (EventLogError, RuntimeError):
            for content in (self.content, self.content + "new"):
                with self.subTest(error=error_type, reuse=content == self.content):
                    error = error_type("log failed")
                    self.runner.run.return_value = content

                    def emit(event, data):
                        if event == EventType.TOOL_RESULT_RETAINED:
                            raise error

                    self.logger.emit.side_effect = emit
                    messages = []
                    with self.assertRaises(error_type) as raised:
                        execute_tool_batch(messages, [self.call], self.context())
                    self.assertIs(raised.exception, error)
                    self.assertEqual(messages, [])
                    self.assertEqual(self.artifacts(), existing)

    def test_child_spills_through_shared_path_and_retry_reuses_request(self):
        responses = iter([
            ModelResponse(None, None, [ToolCall("child", "task", '{"prompt":"inspect"}')], "tool_calls"),
            ModelResponse(None, None, [self.call], "tool_calls"),
            ModelProviderError(ModelErrorKind.CONNECTION, message="offline"),
            ModelResponse("child done", None, [], "stop"),
            ModelResponse("done", None, [], "stop"),
        ])
        requests = []

        def complete(messages, tools):
            requests.append(copy.deepcopy(messages))
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

        provider = Mock()
        provider.supports_tool_choice = False
        provider.complete.side_effect = complete
        self.assertEqual(run_agent(
            provider, self.workspace, [{"role": "user", "content": "task"}],
            shell_runner=self.runner, permission_policy=self.policy, event_logger=self.logger,
            recovery_policy=RecoveryPolicy(base_delay_seconds=0, jitter_ratio=0),
        ), "done")
        self.runner.run.assert_called_once()
        self.assertEqual(requests[2], requests[3])
        result = next(m for m in requests[2] if m["role"] == "tool")
        self.assertLess(len(result["content"]), 3000)
        self.assertEqual(len(self.artifacts()), 1)
        events = [c.args[1] for c in self.logger.emit.call_args_list
                  if c.args[0] == EventType.TOOL_RESULT_RETAINED]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["agent_scope"], "subagent")
        self.assertEqual(events[0]["parent_tool_call_id"], "child")
