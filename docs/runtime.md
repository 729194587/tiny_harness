# Runtime notes

[English](runtime.md) | [简体中文](runtime.zh-CN.md)

TinyHarness connects a model to tools in a workspace. Given a task, it sends the
model the conversation and available tools, checks any tool calls the model
returns, executes them, and adds their results to the conversation. The next
turn lets the model use those results to continue working or give a final answer.
The loop is synchronous: tool calls run one at a time.

This reference explains that flow and the rules around it, for developers
configuring a run or extending the harness. For setup and design trade-offs,
start with the [project README](../README.md).

## Execution and failure

The CLI uses `AgentSession` to keep a conversation across submissions. Each
submission adds the user's task and runs the [agent loop](../tiny_harness/agent/loop.py)
until it returns an answer. Python callers can use `run_agent()` to configure
and invoke the same loop with messages they manage themselves.

At the start of each turn, the runtime prepares the conversation for the model,
reducing its size if the context budget requires it. It then sends a request and
validates the response before adding it to history. If the response contains
tool calls, the runtime executes the batch and records the matching results
before requesting another response. Otherwise, the response text becomes the
final answer.

A run has a turn limit. The last allowed turn is reserved for answering: the
request contains no tools and asks the model for its best available answer.
Calls returned on that turn never execute. A response that passes protocol
validation but still contains tool calls, tool-protocol text, or no answer text
gets one retry; if that retry is also invalid, the runtime returns a fixed
failure message. See [model turns](../tiny_harness/agent/turn.py).

### When a run fails

An ordinary tool error is returned to the model as a tool result, so the model
can respond to it in a later turn. A temporary provider error can also be
retried within the same logical turn. Failures in hooks or event logging,
and invalid model response structure, stop the run instead.

Once a nonempty submission starts, any exception or interruption that escapes
it leaves the Session failed. Further submissions raise `SessionFailedError`
until the caller uses `clear()`. Rejected empty input does not put the Session
into this state.

Inspect the workspace before clearing a failed Session. A fatal failure can
leave a tool batch incomplete even though earlier calls changed files or
started processes. The runtime does not fabricate missing results, replay calls,
or undo those effects. `clear()` only discards conversation history and resets
the failure state. Callers that use the loop directly must discard or rebuild
the failed run themselves. See [Session](../tiny_harness/agent/session.py) and
[batch lifecycle tests](../tests/test_batch_lifecycle.py).

## Tools and run capabilities

### Tool discovery

The tools available to the model depend on what the run has been configured to
support. For example, a run without a test runner does not offer `run_tests`.
The same applies to tools that need a subagent runner or skill catalog.

To supply tools, a module under `tiny_harness/tools/` exports
`build_tools(context)`. Discovery visits these modules in sorted name order and
calls each function with the run context. The function returns the available
`ToolDefinition` objects, or no tools when a required capability is absent.
Each definition supplies the tool's name, description, parameter schema,
handler, and optional trace metadata that is safe to log.

[The registry](../tiny_harness/tools/registry.py) checks for duplicate names,
looks up tools, and produces the schemas sent to the model from those definitions.
There is no second built-in schema list. Discovery is directory-based; it does
not support hot reload or entry-point plugins. Adding a tool also requires
reviewing the permission policy for its name. The
[tool architecture plan](superpowers/plans/2026-09-04-tool-architecture-phase-1.md)
records the original design; current code is authoritative.

### Validation and execution

Before committing an assistant message or running any tool, the runtime validates
the entire model response. Call IDs must be nonempty and unique within the batch;
transport fields and the finish reason must also be valid. An invalid batch
raises `ModelProtocolError` without executing part of it. JSON decoding errors
and invalid tool arguments are handled separately: they become results for the
individual calls and do not discard other valid calls in the batch.

Every tool uses the same dispatch path:

```text
lookup / decode → pre hook → permission → started event
                → handler → post hook → result event → return
```

A pre hook can block a call before permission is checked. Permission decisions
are `ALLOW`, `DENY`, or `ASK`; `ASK` without approval denies execution. Neither
a blocked call nor a denied call reaches the handler. Setting a shell's working
directory does not make it an OS sandbox.

Calls execute sequentially in model order, and each result is appended in that
same order. All `TOOL_CALLED` events are emitted before dispatch begins;
`TOOL_STARTED` is emitted just before the handler runs. Ordinary handler errors become
tool results, while hook execution and event-log failures propagate and stop
the run.

### Reading and searching code

Use `code_outline(path, max_symbols=200)` to locate Python definitions before a
ranged `read_file`. It parses a workspace `.py` file with the standard library
AST parser, without importing or executing the source. The outline lists classes,
functions, async functions, and methods in source order, with indentation for
nested definitions. Line numbers are 1-based and inclusive; the start points
to `class` or `def`, excluding decorators.

The input limit is 2 MiB. Output defaults to 200 symbols, can be configured up to
500, and is capped at 30,000 characters. Truncation, empty outlines, syntax errors,
and invalid paths all receive explicit feedback. The tool uses shared dispatch
and workspace path checks and is allowed by the default permission policy.
It supports only Python source, with no symbol resolution, index, or call graph.

`search_code` and `grep` can limit content searches with an `include` glob
relative to `path`: `**/*.py` matches at all depths, `*.py` only direct children,
and `src/**/*.py` a subtree. Their recursive searches skip directory symlinks
and the directories `.git`, `.venv`, `venv`, `node_modules`, `__pycache__`,
`.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `.tox`, and `.nox`.

To search an excluded directory or file, set `path` to it explicitly; `include`
alone does not override the exclusions. An explicit file path bypasses the
`include` filter, but workspace boundary checks still apply. The directory-name
exclusions above do not apply to file discovery through `glob` and `list_files`.

### Delegated work and todos

The model can delegate a task with `task`. The parent waits while a child agent
runs with fresh messages, then receives the child's final text as the tool
result. The child shares the workspace and model provider, so its file changes
are visible to the parent.

Children inherit the parent's policies and capabilities, but have their own
turn budget, recovery state, context, and todos. They cannot create further
children and do not extract persistent memory. For task tracking within a run,
`todo_write` updates the current todo state; this is separate from persistent
memory.

## Context and recovery

As tools return results, the conversation grows. TinyHarness manages two related
forms of it: the history it keeps between turns, called *canonical history*, and
the request it prepares for a particular model call. History is runtime-owned
state and can be compacted; it is not an immutable transcript.

Preparing a request copies that history, bounds large `read_file` results, and
adds request-only runtime guidance. The code calls this *request projection*.
It does not shorten the stored history or create artifacts. This preparation
still happens when context compaction is disabled.

With a context budget configured, the runtime counts both messages and tool
schemas. Above 80% of the budget, it first saves tool results to files and replaces
them in history with shorter previews. It aims to reduce the context to 55% of
the budget. Only when that is insufficient does it ask for a history summary.
Protected history may prevent reaching 55%, in which case the limit falls back
to 80%. The active request and complete call/result blocks are preserved; if
required context cannot fit, preparation fails explicitly. Normal turns do not
create proactive working checkpoints.

Temporary provider failures retry the already-prepared request. A context-length
error needs different handling: when compaction is enabled, the runtime can
compact and rebuild the request once per logical request, targeting at most 75%
of the failed request while respecting the hard budget.

Both automatic and error-triggered summaries are instructed to keep factual
evidence and uncertainty. They must not judge readiness or prescribe next
actions. See the [context implementation](../tiny_harness/runtime/context.py)
and [context invariants](../tests/test_context_invariants.py).

Token counts are estimates. The default heuristic uses compact JSON at roughly
four characters per token, and Python callers can supply a `token_meter`.
[Context attribution](context-attribution.md) explains how request size is
broken down and why these estimates differ from provider-billed usage.

## Skills and memory

Skills supply instructions that the model can load when needed. TinyHarness
looks for `<name>/SKILL.md` files with YAML `name` and `description` fields in:

- bundled `tiny_harness/skills/`;
- user `~/.tinyharness/skills/`;
- workspace `.tinyharness/skills/`.

A Session keeps the skill catalog discovered when it was created. The model first
receives size-limited catalog metadata and can then use `load_skill` to read a
skill's body. Start a new Session to discover added or changed skills. Skill
bodies are untrusted guidance: they cannot override instructions, permissions,
hooks, or workspace boundaries. The
[skills design](superpowers/plans/2026-09-04-formal-skills-phase-2.md) gives background.

Persistent memory carries useful information between runs, such as stable
preferences, feedback, project facts, and references. It does not store the
current plan or execution state. Enable it with `--memory`; its files live under
`.tinyharness/memory/` in the workspace.

During run preparation, the runtime selects relevant memory entries and loads them into
history with markers scoped to that run. When the root agent produces a final
answer, final-answer hooks extract durable memory and may consolidate existing
entries. Ordinary extraction or consolidation failures preserve the answer;
`EventLogError` still stops the run. See the
[memory implementation](../tiny_harness/memory/).

## Hooks and events

Use hooks to run extension code before or after tools, and events to observe
what happened. The [hooks example](../examples/hooks_demo.py) blocks `write_file`
in a pre hook and observes results in a post hook. After installing the project
as described in the quick start, run it from the repository root:

```powershell
python examples/hooks_demo.py "Inspect this project" --workspace .
```

Library callers get a null event sink by default. The CLI uses events for progress
on stderr and writes a one-shot run's final answer to stdout. JSONL logging can
run alongside console output. Events record model requests and responses, tool
lifecycle and execution time, context changes, and parent/child correlation.
Console progress does not include tool-result bodies.

## CLI and local verification

`TINYHARNESS_API_KEY` is required. `TINYHARNESS_MODEL` and
`TINYHARNESS_BASE_URL` configure the model and provider endpoint; see
[`.env.example`](../.env.example) for example values.
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
