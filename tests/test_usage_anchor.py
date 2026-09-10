"""Usage calibration regressions without an external model connection."""

import copy
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.messages import ModelResponse
from tiny_harness.agent.session import AgentSession
from tiny_harness.agent.turn import call_model, model_request_inputs
from tiny_harness.context.token_meter import CalibratedTokenMeter, HeuristicTokenMeter
from tiny_harness.models.chat_completions import ChatCompletionsProvider
from tiny_harness.runtime.console import ConsoleEventLogger
from tiny_harness.runtime.events import EventType


def test_anchor_append_recalibration_and_missing_usage():
    meter = CalibratedTokenMeter()
    messages = [{"role": "user", "content": "hello"}]
    original = copy.deepcopy(messages)
    h = meter.heuristic
    meter.observe(messages, messages, [], 1000)
    assert meter.estimate(messages, []) == 1000
    messages.append({"role": "assistant", "content": "answer"})
    assert meter.estimate(messages, []) == 1000 + h.estimate(messages, []) - h.estimate(original, [])
    meter.observe(messages, messages, [], 700)
    assert meter.estimate(messages, []) == 700
    meter.observe(messages, messages, [], None)
    assert meter.estimate(messages, []) == h.estimate(messages, [])


@pytest.mark.parametrize("rewrite", [
    lambda m: m.pop(),
    lambda m: m[0].update(content="summary"),
    lambda m: m[1].update(content="artifact reference"),
    lambda m: m.clear(),
])
def test_committed_rewrite_invalidates_even_if_original_is_restored(rewrite):
    meter = CalibratedTokenMeter()
    messages = [{"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"}]
    original = copy.deepcopy(messages)
    meter.observe(messages, messages, [], 1000)
    rewrite(messages)
    meter.reconcile(messages, [])
    assert meter.estimate(original, []) == meter.heuristic.estimate(original, [])


def test_candidates_and_schema_changes():
    meter = CalibratedTokenMeter()
    messages = [{"role": "user", "content": "question"}]
    meter.observe(messages, messages, [], 1000)
    assert meter.estimate([], []) == meter.heuristic.estimate([], [])
    assert meter.estimate(messages, []) == 1000
    meter.reconcile(messages, [{"name": "new"}])
    assert meter.estimate(messages, []) == meter.heuristic.estimate(messages, [])


def test_turn_runtime_overhead_usage_events_and_rewrite():
    provider = Mock()
    provider.supports_tool_choice = False
    provider.complete.return_value = ModelResponse("ok", None, [], "stop", 1000, 20, 1020)
    logger = Mock()
    with TemporaryDirectory() as directory:
        context = create_run_context(provider, Path(directory), event_logger=logger,
                                     max_context_tokens=100000, allow_subagent=False)
        messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}]
        context.current_turn = 1
        first, tools = model_request_inputs(messages, context, finalization=False)
        call_model(messages, context)
        messages.append({"role": "assistant", "content": "ok"})
        context.current_turn = 2
        second, _ = model_request_inputs(messages, context, finalization=False)
        h = HeuristicTokenMeter()
        expected = 1000 + h.estimate(second, tools) - h.estimate(first, tools)
        assert context.token_meter.estimate_request(messages, second, tools) == expected
        assert context.compactor.token_meter is context.token_meter
        call_model(messages, context)
        requested = [c.args[1] for c in logger.emit.call_args_list if c.args[0] == EventType.MODEL_REQUESTED]
        responded = [c.args[1] for c in logger.emit.call_args_list if c.args[0] == EventType.MODEL_RESPONDED]
        assert requested[-1]["context_tokens"] == expected
        assert "prompt_tokens" not in requested[-1]
        assert responded[-1]["prompt_tokens"] == 1000
        assert responded[-1]["completion_tokens"] == 20
        assert responded[-1]["total_tokens"] == 1020
        messages[0]["content"] = "rewritten"
        request, _ = model_request_inputs(messages, context, finalization=False)
        assert context.token_meter.estimate_request(messages, request, tools) == h.estimate(request, tools)


@pytest.mark.parametrize("usage", [None, SimpleNamespace(prompt_tokens=123, completion_tokens=4, total_tokens=127)])
def test_provider_normalizes_optional_usage(usage):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        usage=usage, choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="ok", tool_calls=[]))])
    provider = ChatCompletionsProvider("key", "https://example.test", "model", client=client)
    response = provider.complete([], [])
    assert response.prompt_tokens == (123 if usage else None)
    assert response.completion_tokens == (4 if usage else None)
    assert response.total_tokens == (127 if usage else None)


def test_session_clear_resets_anchor():
    with TemporaryDirectory() as directory:
        session = AgentSession(Mock(), Path(directory), "rules")
        session.token_meter.observe(session.messages, session.messages, [], 1000)
        session.clear()
        assert session.token_meter.estimate(session.messages, []) == HeuristicTokenMeter().estimate(session.messages, [])


def test_console_actual_usage():
    stream = io.StringIO()
    console = ConsoleEventLogger(stream=stream)
    console.emit(EventType.MODEL_RESPONDED, {"prompt_tokens": 123, "completion_tokens": 4})
    assert "实际输入 123 tokens" in stream.getvalue()
    assert "实际输出 4 tokens" in stream.getvalue()
