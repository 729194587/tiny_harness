"""Read-only estimates of the final request; never retain message bodies."""

import json
import re

from tiny_harness.context.token_meter import DEFAULT_TOKEN_METER


PROJECTIONS = {
    "tinyharness_working_memory": "working_memory_projection",
    "tinyharness_environment_context": "environment_projection",
    "tinyharness_skill_catalog": "skill_projection",
    "tinyharness_memory_catalog": "memory_projection",
    "tinyharness_relevant_memory": "memory_projection",
    "tinyharness_context_archive": "context_projection",
    "tinyharness_context_summary": "context_projection",
    "tinyharness_todo_state": "todo_projection",
}


def request_attribution(messages, tools, token_meter=DEFAULT_TOKEN_METER):
    """Use standalone marginal estimates, with envelope/rounding kept separate.

    Fresh means after the last assistant message in this request. In normal
    harness history these results have no subsequent accepted model response.
    This does not assert that a remote failed attempt never read those results.

    Assistant breakdown is diagnostic, outside category totals. Field costs are
    sequential marginal estimates on shallow copies, removing nonempty reasoning,
    content, then the entire tool-call payload. They include field serialization
    overhead; empty fields and role/other metadata remain in envelope_and_other.
    This telescopes exactly to the unchanged assistant_history estimate, with
    rounding assigned in that fixed order. No request messages are modified.
    """
    meter = getattr(token_meter, "heuristic", token_meter)
    baseline = meter.estimate([], [])
    categories = {}
    by_tool = {}
    assistant_breakdown = {
        name: {"count": 0, "estimated_tokens": 0, "serialized_chars": 0}
        for name in ("reasoning_content", "visible_content", "tool_calls",
                     "envelope_and_other")
    }
    last_assistant = max((i for i, m in enumerate(messages)
                          if m.get("role") == "assistant"), default=-1)

    def chars(value):
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))

    def add(target, key, tokens, length):
        bucket = target.setdefault(key, {"count": 0, "estimated_tokens": 0,
                                         "serialized_chars": 0})
        bucket["count"] += 1
        bucket["estimated_tokens"] += tokens
        bucket["serialized_chars"] += length

    calls = {}
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            calls = {}
            for call in message.get("tool_calls") or []:
                name = call.get("function", {}).get("name")
                call_id = call.get("id")
                if call_id in calls:
                    calls[call_id] = "unknown"
                else:
                    calls[call_id] = (name if isinstance(name, str) and
                                      re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
                                      else "unknown")
        category = PROJECTIONS.get(message.get("name"))
        if category is None:
            category = {"system": "system_runtime_guidance",
                        "developer": "system_runtime_guidance",
                        "user": "user_task_messages",
                        "assistant": "assistant_history"}.get(role, "other_messages")
        if role == "tool":
            category = ("fresh_tool_results" if index > last_assistant
                        else "historical_tool_results")
        tokens = meter.estimate([message], []) - baseline
        length = chars(message)
        add(categories, category, tokens, length)
        if category == "assistant_history":
            remaining = dict(message)
            remaining_tokens, remaining_chars = tokens, length
            for field, name in (("reasoning_content", "reasoning_content"),
                                ("content", "visible_content"),
                                ("tool_calls", "tool_calls")):
                if remaining.get(field):
                    del remaining[field]
                    next_tokens = meter.estimate([remaining], []) - baseline
                    next_chars = chars(remaining)
                    add(assistant_breakdown, name, remaining_tokens - next_tokens,
                        remaining_chars - next_chars)
                    remaining_tokens, remaining_chars = next_tokens, next_chars
            add(assistant_breakdown, "envelope_and_other", remaining_tokens, remaining_chars)
        if role == "tool":
            add(by_tool, calls.get(message.get("tool_call_id"), "unknown"), tokens, length)
        elif role != "assistant":
            calls = {}
    for tool in tools:
        add(categories, "tool_schemas", meter.estimate([], [tool]) - baseline, chars(tool))
    total = meter.estimate(messages, tools)
    return {
        "schema_version": 1,
        "estimated_tokens": total,
        "serialized_chars": chars({"messages": messages, "tools": tools}),
        "categories": categories,
        "tool_results_by_name": by_tool,
        "assistant_history_breakdown": assistant_breakdown,
        "envelope_and_rounding_tokens": total - sum(
            bucket["estimated_tokens"] for bucket in categories.values()),
    }
