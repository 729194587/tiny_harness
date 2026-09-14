"""Offline summaries of the pipeline's ordered events.jsonl artifacts.

Logical turns are keyed by file, run, child scope and turn; retries do not
inflate turn counts. Usage and attribution include every observed request,
including auxiliary model work. Missing usage is unknown, never zero.
"""

import json
from collections import Counter
from pathlib import Path

from tiny_harness.runtime.events import EventType


def analyze_run(run_dir: Path) -> dict:
    paths = sorted(run_dir.rglob("events.jsonl"))
    if not paths:
        raise ValueError(f"No events.jsonl artifacts in {run_dir}")
    turns = set()
    calls = Counter()
    usage = {name: [] for name in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")}
    request_contexts = []
    pre_prune_contexts = []
    prunes = []
    attribution = {"categories": {}, "tool_results_by_name": {},
                   "assistant_history_breakdown": {}}
    attributed_requests = responses = 0
    first_writes = []
    mutations = []
    started_tools = 0
    checkpoints = []
    summary_hit = []
    summary_miss = []
    for path in paths:
        current_turns = {}
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    data = event["data"]
                    kind = event["event_type"]
                    if not isinstance(data, dict):
                        raise ValueError("event data must be an object")
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"Invalid event at {path}:{line_number}") from error
                scope = (str(path.relative_to(run_dir)), event.get("run_id"),
                         data.get("parent_tool_call_id"))
                turn = data.get("turn", current_turns.get(scope))
                key = (*scope, turn)
                if kind == EventType.MODEL_REQUESTED:
                    if data.get("purpose", "main") == "main" and turn is not None:
                        turns.add(key)
                        current_turns[scope] = turn
                    if isinstance(data.get("context_tokens"), (int, float)):
                        request_contexts.append(data["context_tokens"])
                    detail = data.get("context_attribution")
                    if isinstance(detail, dict):
                        attributed_requests += 1
                        if not isinstance(data.get("context_tokens"), (int, float)):
                            if isinstance(detail.get("estimated_tokens"), (int, float)):
                                adjustment = detail.get("calibration_adjustment_tokens")
                                request_contexts.append(
                                    detail["estimated_tokens"]
                                    + (adjustment if isinstance(adjustment, (int, float)) else 0)
                                )
                        for group, buckets in attribution.items():
                            for name, bucket in detail.get(group, {}).items():
                                value = bucket.get("estimated_tokens")
                                if isinstance(value, (int, float)):
                                    entry = buckets.setdefault(name, {"sum_estimated_tokens": 0,
                                                                      "peak_estimated_tokens": 0})
                                    entry["sum_estimated_tokens"] += value
                                    entry["peak_estimated_tokens"] = max(entry["peak_estimated_tokens"], value)
                elif kind == EventType.MODEL_RESPONDED:
                    responses += 1
                    if data.get("purpose") == "summary":
                        for name, values in (("prompt_cache_hit_tokens", summary_hit),
                                             ("prompt_cache_miss_tokens", summary_miss)):
                            if isinstance(data.get(name), (int, float)):
                                values.append(data[name])
                    for name, values in usage.items():
                        value = data.get(name)
                        if name == "total_tokens" and value is None:
                            if all(isinstance(data.get(n), (int, float)) for n in ("prompt_tokens", "completion_tokens")):
                                value = data["prompt_tokens"] + data["completion_tokens"]
                        if isinstance(value, (int, float)):
                            values.append(value)
                elif kind == EventType.TOOL_CALLED:
                    calls[key] += 1
                elif kind == EventType.TOOL_STARTED:
                    started_tools += 1
                elif kind == EventType.WORKSPACE_OBSERVED:
                    mutations.append({"scope": list(scope), "turn": data.get("turn"),
                                      "root_turn": data.get("root_turn"),
                                      "workspace_changed": data.get("workspace_changed"),
                                      "observation_status": data.get("observation_status"),
                                      "changed_paths": data.get("changed_paths")})
                elif kind == EventType.TOOL_RESULT:
                    if data.get("tool_name") in {"write_file", "edit_file"} and data.get("outcome") == "returned":
                        if not any(item["scope"] == list(scope) for item in first_writes):
                            first_writes.append({"scope": list(scope), "turn": turn})
                elif kind == EventType.CONTEXT_COMPACTED and data.get("reason") == "working":
                    if data.get("strategy") == "llm_task_state_checkpoint":
                        checkpoints.append({"scope": list(scope), "turn": data.get("turn"),
                                            "before_tokens": data.get("before_tokens"),
                                            "after_tokens": data.get("after_tokens")})
                    prunes.append({"scope": list(scope), "turn": data.get("turn"), **{
                        name: data.get(name) for name in (
                            "before_tokens", "after_tokens", "pruned_results", "pruned_batches", "strategy",
                            "target_reached", "blocked_by_recent_protection")},
                        "transition": f"{data.get('before_tokens')} -> {data.get('after_tokens')}"})
                    if isinstance(data.get("before_tokens"), (int, float)):
                        pre_prune_contexts.append(data["before_tokens"])
    multi = sum(count > 1 for count in calls.values())
    complete = bool(mutations) and len(mutations) == started_tools and all(
        item["observation_status"] == "complete" for item in mutations)
    changed_turns = [item["root_turn"] for item in mutations
                     if item["workspace_changed"] and isinstance(item["root_turn"], int)]
    totals = {name: sum(values) if values else None for name, values in usage.items()}
    hit, miss = totals["prompt_cache_hit_tokens"], totals["prompt_cache_miss_tokens"]
    return {
        "checkpoint_count": len(checkpoints), "checkpoints": checkpoints,
        "summary_cache_hit_rate": (sum(summary_hit) / (sum(summary_hit) + sum(summary_miss))
                                   if summary_hit and summary_miss and sum(summary_hit) + sum(summary_miss) > 0
                                   else None),
        "total_prompt_tokens": totals["prompt_tokens"],
        "total_cache_hit_tokens": hit,
        "total_cache_miss_tokens": miss,
        "overall_cache_hit_rate": hit / (hit + miss) if hit is not None and miss is not None and hit + miss > 0 else None,
        "run_dir": str(run_dir), "event_files": len(paths), "turns": len(turns),
        **{name: sum(values) if values and len(values) == responses else None
           for name, values in usage.items()},
        "usage_observed_totals": {name: sum(values) if values else None for name, values in usage.items()},
        "usage_coverage": {name: len(values) for name, values in usage.items()},
        "model_responses": responses, "tool_calls": sum(calls.values()),
        "tool_call_turns": len(calls), "multi_tool_turns": multi,
        "multi_tool_rate": multi / len(calls) if calls else 0,
        "calls_per_tool_turn": sum(calls.values()) / len(calls) if calls else 0,
        "first_workspace_mutation_turn": min(changed_turns) if complete and changed_turns else None,
        "mutation_observability": "complete" if complete else "unknown: missing or incomplete workspace observations",
        "workspace_observations": mutations,
        "first_returned_file_mutation_tool_turns": first_writes,
        "peak_request_context_tokens": max(request_contexts) if request_contexts else None,
        "peak_pre_prune_context_tokens": max(pre_prune_contexts) if pre_prune_contexts else None,
        "working_context_prune_events": len(prunes), "prunes": prunes,
        "context_attribution_summary": {"requests": attributed_requests, **attribution},
    }


CORE_METRICS = (
    "total_prompt_tokens", "total_cache_hit_tokens", "total_cache_miss_tokens", "overall_cache_hit_rate",
    "turns", "prompt_tokens", "completion_tokens", "total_tokens", "tool_calls",
    "tool_call_turns", "multi_tool_turns", "multi_tool_rate", "calls_per_tool_turn",
    "first_workspace_mutation_turn", "peak_request_context_tokens",
    "peak_pre_prune_context_tokens", "working_context_prune_events",
)


def compare_runs(run_a: Path, run_b: Path) -> dict:
    a, b = analyze_run(run_a), analyze_run(run_b)
    metrics = {}
    for name in CORE_METRICS:
        left, right = a[name], b[name]
        delta = right - left if left is not None and right is not None else None
        metrics[name] = {"run_a": left, "run_b": right, "delta": delta,
                         "percent_change": (None if delta is None else 100 * delta / left if left else
                                            0 if delta == 0 else None)}
    return {"run_a": str(run_a), "run_b": str(run_b), "metrics": metrics,
            "conventions": "delta = B - A; percent = 100 * delta / A; null = unknown or undefined"}
