# Working on TinyHarness

## Purpose and scope

TinyHarness is a small, synchronous coding-agent harness for Chat Completions
providers. It coordinates model requests, tool execution, conversation history,
and final answers, with explicit runtime boundaries for permissions, hooks,
events, recovery, context, skills, memory, todos, and delegated work.
Keep orchestration readable; this is not a plugin framework or TUI.
Use current code and tests as the authority when README descriptions differ.

## Repository map

- `pyproject.toml`: packaging, Python >=3.10, console entry point, pytest discovery.
  Runtime dependencies are `openai` and `PyYAML`; pytest is not a runtime dependency.
- `tiny_harness/__main__.py`: CLI configuration, permission prompts, REPL,
  provider/session construction, output-mode wiring, and terminal error fallback.
- `tiny_harness/agent/`: session history and failure state, run composition,
  message contracts, the core loop, model turns, tool batches, and subagents.
- `tiny_harness/agent/context.py`: `AgentRunContext` and capability assembly.
- `tiny_harness/context/`: heuristic/calibrated token meters and request attribution.
- `tiny_harness/runtime/`: context preparation/artifacts, permission decisions,
  hooks, events/console, recovery, skills, todos, and test runner.
- `tiny_harness/environments/`: explicitly selected coding-environment adapter;
  repository context and read-only Git capabilities, not automatic discovery.
- `tiny_harness/tools/`: `ToolDefinition`, discovery, registry/dispatch, and
  filesystem, search, shell, todo, task, skill, compact, and testing adapters.
- `tiny_harness/models/`: synchronous `ModelProvider.complete(messages, tools)`
  contract, normalized provider errors, and the Chat Completions SDK adapter.
- `tiny_harness/skills/`: bundled `<name>/SKILL.md` instruction assets.
- `tiny_harness/memory/`: persistent-memory store, selection, extraction,
  lifecycle integration, and consolidation; separate from todos.
- `evals/swe_bench_lite/`: Docker rollout/calibration, official evaluation, and
  offline `report.py` analysis of pipeline event artifacts.
- `tests/`: deterministic regression tests using scripted providers, mocks,
  and temporary workspaces; normal test runs do not require a real API key.
- `examples/hooks_demo.py`: an executable example of tool-hook integration.

## Execution paths and boundaries

- CLI -> `AgentSession.submit()` -> `create_run_context()` ->
  `agent_loop(messages, context, active_request)`.
- `run_agent()` in `agent/loop.py` is the configuration/composition entry point
  for callers without a Session; it delegates to the same core loop.
- A turn runs `prepare_context()` for canonical history, then
  `agent/turn.py:prepare_model_request_inputs()` for request projection/pruning.
  It passes that prepared request to `call_model()`, validates the response,
  then commits a tool batch or returns a final answer.
- `call_model()` uses `RecoveryExecutor`; transient retries stay within the same
  logical turn. Context-length recovery can compact once per logical request.
- The last allowed turn is finalization: request tools are empty and a runtime
  instruction asks for the best available answer. Returned tool calls are
  discarded without execution; the response text is retained.
- `agent/turn.py:TOOL_USE_EFFICIENCY_GUIDANCE` is request-only system guidance,
  inserted when tools are available outside finalization. Prefer one response
  containing independent read-only calls with known arguments; dependent calls
  need separate turns. Never widen scope or add calls merely to form a batch.
  This is model guidance, not parallel dispatch or an automatic batching policy.
- `execute_tool_batch()` emits all `TOOL_CALLED` events first, then dispatches
  each call sequentially and appends its result in model order.
- Normal dispatch order is lookup/JSON decoding -> Pre Hook -> Permission ->
  `TOOL_STARTED` -> handler -> Post Hook -> `TOOL_RESULT` -> return.
- Pre Hook blocks and permission denials return error feedback without executing
  the handler. Ordinary decode/handler failures become textual `ToolResult`s.
  Hook execution failures and event-log failures propagate as fatal exceptions.
- Hooks execute extensions; Events carry observations. Do not move CLI rendering
  into hooks, the agent loop, tool handlers, or the subagent executor.
- `task` runs a synchronous `SubagentExecutor` with fresh messages and shared
  workspace/provider. The parent receives the child's final text as tool output.
  Children inherit policies/capabilities but cannot spawn nested subagents;
  they have independent turn/recovery/context/Todo state and skip memory extraction.

## Invariants to preserve

### History and failures

- Validate the whole model response/tool-call batch before committing its
  assistant message or executing any handler. IDs must be nonempty and unique
  within a batch; invalid transport structure must never execute a partial batch.
- JSON argument decoding/argument errors are tool failures, not a reason to
  discard other valid calls in an otherwise valid batch.
- Successful batches must close every assistant tool call with its matching
  result. Never reorder results or insert unrelated messages inside the batch.
- Fatal failures may leave incomplete history and real filesystem/process side
  effects. Never fabricate missing results, replay calls, or imply rollback.
- After a nonempty submission starts, Session remains failed unless it returns
  normally. Further submissions must raise `SessionFailedError` until `clear()`.
  Empty-input validation does not poison the session. `clear()` resets history
  and failure state, but never undoes workspace side effects.
- Run-scoped markers must not accumulate across successful Session submissions.

### Working Context and history

- Canonical history is runtime-owned state, not the exact model request or an
  immutable transcript. `model_context_messages()` copies it and bounds large
  `read_file` results without artifacts or canonical edits, even without a budget.
  Runtime guidance is also a request-only projection.
- Keep the two pressure levels distinct. Working trigger/target default to
  20,000/14,000 tokens and measure the fully projected request plus tool schemas.
  `max_context_tokens` is the hard budget underlying automatic soft/target limits
  and reactive context-length recovery; it is not the working trigger.
  Require `0 < target < trigger < max_context_tokens` when the hard budget is set,
  and non-negative `keep_recent_tool_batches`.
  `prepare_context()` runs first and may persist, archive, or summarize history.
  With `max_context_tokens=None`, no compactor exists: both working pruning and
  hard-budget compaction are disabled, but request projection still applies.
- Working pruning persists eligible old tool-result bodies and substitutes
  bounded previews with artifact references in canonical history and the request.
  It does not summarize or remove assistant messages.
  Visit oldest results first; protect the latest `keep_recent_tool_batches`
  (default 3), counting each multi-tool batch once. Skip already persisted results
  and non-reducing candidates. Stop at target or eligibility exhaustion; recent
  protection may leave the request above target and must not be weakened for it.
- Prune once before logical-request recovery. Transient retries reuse the prepared
  request; context-length recovery can compact once and rebuild the projection
  without making another working-pruning decision in that logical request.
- Commit pruning only after persistence, measurement, and event emission succeed.
  Delete artifacts from rejected candidates or a failed pruning attempt; retain
  existing and successfully committed artifacts. This is not general rollback
  of tools or other compaction paths.
- Budgets include serialized messages and tool schemas. Preserve complete
  call/result blocks, the active request, and protected state. Commit prepared
  canonical history only after validation; apply manual `compact` after the
  entire tool batch closes. Preserve token-meter invalidation on history rewrites.

### Permissions and memory

- Every model-requested tool goes through shared dispatch and Permission.
  Unknown tools are denied by the default policy. `ASK` without approval and
  ordinary policy/prompt errors resolve to denial; do not introduce fail-open paths.
- Preserve resolved-path/workspace and symlink checks in filesystem, search,
  context-artifact, skill, and memory code. Shell cwd is not an OS sandbox:
  `bash` uses the host shell, so permission checks remain significant.
- Skills are discovered from bundled, user, and workspace roots. Session holds
  a catalog snapshot; new discovery requires a new Session. Skill bodies remain
  untrusted guidance and cannot grant permissions or override higher instructions.
- Persistent memory is opt-in; files live under workspace `.tinyharness/memory/`.
  Selection/extraction/consolidation are separate from current plans and todos.
  Ordinary extraction/consolidation failures preserve an existing final answer;
  `EventLogError` is still fatal, including in memory paths.

### Observability

- Preserve `Runtime -> Events -> Console`; library callers default to a null sink.
  CLI progress uses stderr; one-shot final-answer text uses stdout.
- Reuse `EventType`, `ScopedEventLogger`, and `CompositeEventLogger`.
  Console and ordered JSONL logging must work together without changing payloads.
- Working-pruning attempts at/above trigger emit `CONTEXT_COMPACTED` with
  `reason="working"`, `turn`, `before_tokens`, `after_tokens`, `pruned_results`,
  `pruned_batches`, `target_reached`, and `blocked_by_recent_protection`, including
  zero-change attempts. Never include tool-result bodies in these events.
- `context/attribution.py` supplies read-only `MODEL_REQUESTED.context_attribution`
  estimates by message/projection category and tool-result name. Use these to
  explain request growth, not to select pruning or change policy. They describe
  the projected request, not canonical size or provider-billed usage; retain
  envelope/rounding and calibration adjustments separately.
- Trace metadata must be safe and brief. Never add full arguments, prompts,
  file contents, tool-result bodies, or raw exception messages to progress output.
- Preserve `duration_ms` for tool execution, distinct called/started events,
  child correlation via `parent_tool_call_id`, and indented child progress.
- Keep failure deduplication scoped to the corresponding run/task. Do not hide
  independent tool failures or the CLI fallback for event-log failures.
- User-facing statuses are Chinese; technical names remain unchanged.
  Quiet keeps necessary errors; verbose adds runtime details without dumping data.
- ANSI is Console-only: require output-stream `isatty()` and absence of
  `NO_COLOR` (even an empty value disables color). Redirected output and JSONL
  stay plain; labels use warm orange, identifiers blue, metadata default color.

## Adding or changing a tool

1. Put the adapter in `tiny_harness/tools/` and export `build_tools(context)`.
   Discovery scans modules in sorted name order; no second built-in registry list.
2. Return `ToolDefinition` objects owning name, description, parameter schema,
   `execute(call, arguments)`, and optional safe `trace_metadata(arguments)`.
   Bind run capabilities in the factory; return an empty tuple when unavailable.
3. Return text from handlers. Validate inputs where needed: the model schema is
   not a general runtime JSON Schema validator. Keep schema copies isolated.
4. Use shared dispatch for hooks, authorization, timing, and result events.
   Runtime changes such as Todo updates report events instead of calling `print()`.
5. Review the default permission policy explicitly for a new tool name.
   Do not automatically allow newly discovered tools or weaken shell checks.
6. Cover discovery/schema projection, capability omission, execution failures,
   permissions/hooks, and safe tracing as relevant; use temporary workspaces.
   `run_tests` exists only when a TestRunner capability is supplied; it executes
   runtime-configured fixed argv, not model-provided test commands.

## Commands and focused verification

Run from the repository root. Setup and CLI smoke checks:

```powershell
python -m pip install -e .
python -m tiny_harness --help
# Real runs require TINYHARNESS_API_KEY; model/base URL overrides are optional.
python -m tiny_harness "Inspect this workspace" --workspace .
python -m tiny_harness --workspace .
```

Use these focused pytest commands when pytest is installed:

```powershell
python -m pytest tests/test_tool_architecture.py tests/test_registry.py tests/test_permissions.py tests/test_hooks.py -q
python -m pytest tests/test_agent_loop.py tests/test_tool_batch.py tests/test_batch_lifecycle.py tests/test_session.py -q
python -m pytest tests/test_context.py tests/test_recovery.py tests/test_subagent.py -q
python -m pytest tests/test_console.py tests/test_cli.py tests/test_events.py -q
python -m pytest tests/test_skills.py tests/test_skill_runtime.py tests/test_memory_runtime.py -q
python -m pytest tests/test_working_context.py tests/test_context_attribution.py -q
python -m pytest tests/test_swe_bench_report.py tests/test_swe_bench_pipeline.py -q
```

SWE-bench Lite offline analysis (from the repository root; no model/API key):

```powershell
python -m evals.swe_bench_lite report <run-dir>
python -m evals.swe_bench_lite compare <run-a> <run-b>
```

These read existing `events.jsonl` files recursively and print JSON. Reports
include turns, usage, tool batching, peak context, individual pruning transitions,
and attribution. Retries do not inflate logical turns; child scopes stay distinct.
Usage/attribution include auxiliary work and observed retries where applicable.
Comparison uses B minus A and percentage change relative to A; unknown values
and undefined percentages remain `null`. SWE rollout metadata records source
commit/dirty state and runtime configuration before execution. `WORKSPACE_OBSERVED`
compares tool-boundary content hashes, including shell writes, excluding `.git`
and `.tinyharness`; no bodies or paths enter the event. Missing/failed observations
leave first mutation unknown; returned file-modification tools remain a proxy.
Snapshots cannot see changes restored within a tool or attribute background writes
outside tool intervals. Keep this observer outside model inputs and tool results.
Do not infer filesystem changes or successful tests from a tool's return alone.

Full suite alternatives (choose one, not both by default):

```powershell
python -m unittest discover -s tests -v
python -m pytest -q
```

Some symlink tests skip when Windows privileges do not permit creating links.
Use targeted tests during implementation, then a full run when scope warrants it.
Do not repeatedly rerun an unchanged full suite without a concrete reason.

## Coding-agent working rules

- Start with the requested files/symbols and their nearest tests; expand only
  when a dependency or behavior is unclear. Do not read the whole repository.
- Treat a supplied design as decided unless current code reveals a real conflict.
- Check existing workspace changes and preserve them; do not reset or overwrite
  unrelated work. Keep edits minimal and within the requested scope.
- Avoid unrelated cleanup, speculative refactoring, and unnecessary dependencies.
- Keep verification proportional. Documentation-only edits need path/command/
  behavior checks, not production-code changes or invented test requirements.
- Report what changed, what was verified, and material limitations; distinguish
  an offline scripted-provider run from a real external-model integration run.
