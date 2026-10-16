# Model request context attribution

[English](context-attribution.md) | [简体中文](context-attribution.zh-CN.md)

Context attribution helps explain which parts of a model request consume space:
conversation history, tool results, runtime guidance, or tool schemas. Use it to
investigate request growth, not to infer billed usage or choose a pruning policy.
The offline [run report](../evals/swe_bench_lite/README.md#inspect-existing-runs)
aggregates these estimates from event logs.

## What is measured

`MODEL_REQUESTED.data.context_attribution` (`schema_version: 1`) measures the final
messages and tool schemas immediately before each physical provider attempt in
`RecoveryExecutor`. Main/child turns and routed summary/memory calls are covered;
direct calls to a provider outside `RecoveryExecutor` are not instrumented.
Transient retries emit another measurement; reactive compaction rebuilds the
request before it is measured again. No request or canonical history is edited.

## Classification

- system/developer messages: `system_runtime_guidance`, including unmarked
  runtime state and efficiency/finalization guidance.
- user messages: `user_task_messages`; assistant messages, including function
  names/arguments and any reasoning fields: `assistant_history`.
- tool messages: `fresh_tool_results` if after the last assistant message,
  otherwise `historical_tool_results`. Fresh means no subsequent accepted
  assistant response, not proof that the provider has never read them:
  failed or retried calls may already have done so.
- For non-tool messages, exact known `name` markers override role classification:
  skill catalog, persistent memory catalog/relevant memory,
  context summary/archive, and todo state have separate projection categories.
- Tool schemas: `tool_schemas`; unrecognized roles: `other_messages`.

Tool-result counts are also aggregated in `tool_results_by_name` by matching
`tool_call_id` in the preceding assistant batch, not a global ID map. Missing or
ambiguous matches and unsafe names become `unknown`. Names are restricted to
64 ASCII letters/digits/underscores/hyphens. Aggregates contain no arguments,
IDs, paths, or bodies. Categories with no items are omitted (equivalent to zero).

Unmarked guidance cannot be split reliably by origin. Loaded skill bodies
returned by tools remain tool results. Summarized-away results cannot be
recovered or attributed to their original tools. Custom messages impersonating
reserved markers cannot be distinguished without separate provenance metadata.

## Reading the estimates

Each bucket has `count`, `estimated_tokens`, and `serialized_chars` (compact
JSON, Unicode characters, including message/schema structure; not UTF-8 bytes).
Tokens use the configured `TokenMeter`, or the underlying heuristic of a calibrated
meter. A bucket sums standalone item estimates minus the empty-request estimate.
The request total is measured separately. `envelope_and_rounding_tokens` is the
signed residual needed to reconcile bucket sums with `estimated_tokens`; for
non-additive custom meters it also includes cross-item effects.

When existing `context_tokens` metadata is present, the signed
`calibration_adjustment_tokens` reconciles that calibrated request estimate with
the attribution total. Calibration is not apportioned to categories and is not
modified by attribution. This is estimated input context usage, not exact
provider billing, cached-token accounting, output usage, or transport options
such as `tool_choice`. Use `MODEL_RESPONDED` usage fields for provider-reported
totals. Attribution appears in JSONL events and offline reports, not console
progress output.

### Assistant history breakdown

`assistant_history_breakdown` separates `reasoning_content`, `visible_content`,
`tool_calls`, and `envelope_and_other`. These are a breakdown of
`assistant_history`, not additional categories to add to the request total.
All four buckets are present even when empty.

The estimator removes nonempty reasoning, visible content, then the complete
tool-call payload from a copy of each assistant message. Each bucket records
the resulting decrease, including field serialization overhead; rounding follows
that order. Empty fields and remaining metadata belong to `envelope_and_other`.
The four estimates sum to `assistant_history`; they do not expose message bodies.
See the [implementation](../tiny_harness/context/attribution.py) and
[tests](../tests/test_context_attribution.py).

## Experimental use: bounded-history simulation

No experimental projection is implemented. To study possible context savings
offline, a fixed-trajectory simulation would need a scripted provider and canonical
history snapshots immediately before request projection. For each turn, fork the
snapshot: build the unchanged baseline request, then build a separate experimental
copy that preserves fresh results in full and bounds only historical results. Preserve call IDs, result order,
all non-result messages, runtime projections, tools and finalization state.
Estimate both assembled requests with the same uncalibrated meter and report
per-turn totals, fresh/historical totals, and differences for several bounds.
Do not infer savings by subtracting raw text lengths from calibrated totals.

In particular, current `model_context_messages()` bounds large `read_file` results
even when fresh. The experiment must start from canonical snapshots, not the
already-projected request, to recover those complete fresh results. Previously
compacted canonical history may already contain previews/summaries; replay from
pre-compaction fixtures or retained artifacts is needed to recover originals.

Counts-only JSONL events cannot reconstruct content or estimate arbitrary
head/tail projections exactly. Use controlled offline fixtures rather than
adding bodies to events. If simulating compaction thresholds as well, keep
separate branch histories and scripted summary outcomes. Actual changed model
decisions, downstream tool calls, and billed tokens cannot be predicted by a
fixed-trajectory simulation; a real-model comparison would be a separate step.
