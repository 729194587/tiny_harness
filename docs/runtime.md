# Runtime notes

These notes describe the current runtime. For the motivation and experiments,
start with the [project homepage](../README.md).

## Execution and failure

The CLI submits work through `AgentSession`; `run_agent()` composes the same
[core loop](../tiny_harness/agent/loop.py) for library callers. Each turn prepares
canonical history, projects a model request, validates the response, and either
executes a tool batch or returns a final answer.

The entire response is validated before committing an assistant message or
executing any tool. Call IDs must be nonempty and unique within a batch, and
transport fields and finish reason must be valid. Invalid batches raise
`ModelProtocolError`; JSON decoding and tool argument errors instead become
individual tool results. Calls execute sequentially in model order, with all
`TOOL_CALLED` events emitted before dispatch begins.

The shared dispatch path is:

```text
lookup / decode → pre hook → permission → started event
                → handler → post hook → result event → return
```

Pre-hook blocks and permission denials do not execute the handler. Ordinary tool
errors become feedback; hook execution and event-log failures are fatal.
Permissions use `ALLOW`, `DENY`, or `ASK`; an unanswered approval request denies
execution. Shell cwd is not an OS sandbox.

After a nonempty submission starts, any exception or interruption leaves the
Session failed until `clear()`. Fatal failures can leave incomplete tool batches
and real filesystem or process effects. The runtime does not invent missing
results, replay calls, or roll back effects. `clear()` discards history and resets
failure state only; inspect the workspace first. Empty-input validation does not
poison a Session. Direct loop callers must discard or rebuild failed runs.
See [Session](../tiny_harness/agent/session.py) and
[batch lifecycle tests](../tests/test_batch_lifecycle.py).

Transient provider retries stay within a logical turn. The last allowed turn is
finalization: tools are withheld and the model is asked for its best available
answer. No further tool calls execute. See
[model turns](../tiny_harness/agent/turn.py) for invalid-final-answer handling.

## Tools and run capabilities

Modules under `tiny_harness/tools/` export `build_tools(context)`. Discovery scans
modules in sorted order; each factory returns available `ToolDefinition` objects
with a name, description, parameter schema, handler, and optional safe trace
metadata. Factories can return nothing when a capability is absent, such as a
missing subagent runner, skill catalog, or test runner.

[The registry](../tiny_harness/tools/registry.py) owns lookup, duplicate checks,
and schema projection; there is no second built-in schema list. All execution
uses shared dispatch. This is directory-based discovery, without hot reload or
entry-point plugins. New tool names require explicit permission-policy review.
The [tool architecture plan](superpowers/plans/2026-09-04-tool-architecture-phase-1.md)
records the original design; current code is authoritative.

`task` runs a synchronous child with fresh messages and a shared workspace and
provider. Children inherit policies and capabilities, have independent turn,
recovery, context, and Todo state, cannot spawn nested children, and skip memory
extraction. The parent receives the child's final text as tool output.
`todo_write` updates run-scoped task state, separate from persistent memory.

Hooks execute extensions; events carry observations. The
[hooks example](../examples/hooks_demo.py) blocks `write_file` in a pre hook and
observes results in a post hook:

```powershell
python examples/hooks_demo.py "Inspect this project" --workspace .
```

Library callers default to a null event sink. CLI progress goes to stderr and
one-shot final text to stdout; JSONL logging can run alongside console output.
Events cover model requests/responses, tool lifecycle and timing, context changes,
and parent/child correlation without putting tool bodies into progress output.

## Context and recovery

Canonical history is runtime-owned state, distinct from the model request.
Request projection copies history and bounds large `read_file` results even when
compaction is disabled; runtime guidance is also request-only.

The hard budget includes messages and tool schemas. Above 80% of the configured
budget, preparation persists tool results and targets 55%; only insufficient
reduction invokes a history summary. Protected history can require falling back
to the 80% limit. Complete call/result blocks and the active request are preserved.
There is no proactive working checkpoint.

Transient retries reuse the prepared request. Context-length recovery can compact
once per logical request, targeting at most 75% of the failed request while
respecting the hard budget. Automatic and reactive summaries share a factual-only
contract that preserves uncertainty and excludes judgments about readiness or
next actions. See [context implementation](../tiny_harness/runtime/context.py)
and [context invariants](../tests/test_context_invariants.py).

The default heuristic estimates tokens from compact JSON at roughly four
characters per token. Python callers can inject a `token_meter`; attribution
estimates are distinct from provider-billed usage. See
[context attribution](context-attribution.md) for measurement limits.

## Skills and memory

Skills are discovered as `<name>/SKILL.md` with YAML `name` and `description` from:

- bundled `tiny_harness/skills/`;
- user `~/.tinyharness/skills/`;
- workspace `.tinyharness/skills/`.

A Session holds a catalog snapshot. The model initially receives bounded
metadata, then uses `load_skill` for bodies. Bodies are untrusted guidance and
cannot override instructions, permissions, hooks, or workspace boundaries.
Start a new Session to rediscover changes. The
[skills design](superpowers/plans/2026-09-04-formal-skills-phase-2.md) gives background.

Persistent memory is opt-in through `--memory`, with files under workspace
`.tinyharness/memory/`. Run preparation selects and loads relevant entries using
run-scoped markers. Root final-answer hooks extract durable memory and optionally
consolidate it. Ordinary extraction/consolidation failures preserve the answer;
`EventLogError` remains fatal. Memory stores stable preferences, feedback, project
facts, and references, not current plans or execution state. See the
[memory implementation](../tiny_harness/memory/).

## CLI and local verification

`TINYHARNESS_API_KEY` is required. `TINYHARNESS_MODEL` and
`TINYHARNESS_BASE_URL` override the defaults shown in the homepage example.
Omitting the task enters a REPL that reuses one Session.

Run `python -m tiny_harness --help` for the authoritative option list:

| Option | Purpose / default |
| --- | --- |
| `--workspace` | Workspace directory; current directory |
| `--max-turns` | Main-agent turn limit; 20 |
| `--max-model-retries` | Transient retries per logical request; 2 |
| `--max-context-tokens` | Estimated context budget; 125000 |
| `--no-context-compaction` | Disable budgeting and compaction, keeping request projection |
| `--subagent-max-turns` | Child-agent turn limit; 10 |
| `--event-log` | Append lifecycle events to a JSONL file |
| `--memory` | Enable persistent workspace memory |
| `--quiet` / `--verbose` | Reduce progress output / add runtime details |

Deterministic tests use scripted providers and local fixtures, without a real API:

```powershell
python -m unittest discover -s tests -v
```

See [AGENTS.md](../AGENTS.md) for focused test commands and extension invariants.
