# Minimal SWE Baseline Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add only the observability, task packaging, isolated grading, and calibration machinery needed to run the unchanged TinyHarness Runtime on one calibrated repository-level coding task.

**Architecture:** Keep two deliberately different observation layers: enrich the public metadata-only Event Log with bounded safe summaries, and write a gitignored local research trace containing the model-visible assistant/tool interaction with full Tool Call arguments and Tool Results but no hidden reasoning. Add a new SWE task layer beside the existing synthetic Pilot: a strict manifest describes a pinned upstream repository and evaluator, a local bare-repository cache materializes a one-commit Agent workspace, evaluator-owned assets remain outside that workspace, and calibration proves the base/reference contracts before any real Agent run. The first SWE run uses one global frozen copy of the current Eval Permission/Bash configuration; existing Agent decisions, tool schemas, Permission semantics, and completion behavior remain unchanged.

**Tech Stack:** Python 3.10+, standard library (`dataclasses`, `hashlib`, `json`, `pathlib`, `subprocess`, `unittest`), existing OpenAI-compatible provider, existing TinyHarness Event/Hook/TestRunner/Eval abstractions, and the selected upstream repository's native test runner.

**Spec:** Conversation-approved revised design and constraints from 2026-09-04; no separate design file was created because the approved brainstorming round was explicitly kept read-only.

## Global Constraints

- Do not add or alter any Agent-facing Coding Tool.
- Do not change `EvalPermissionPolicy`, Permission decisions, denial feedback, context policy, recovery policy, system prompt, model sampling, completion behavior, or default Runtime capability availability.
- Observability is limited to tool lifecycle, bounded safe argument summaries, Tool Result size, model turn/context size, provider usage, file mutation, termination, and grader outcome.
- Do not implement intent classification, semantic fingerprints, edit-churn analysis, historical mutation grading, or a general trajectory query platform.
- Do not run hidden grading after each mutation. Keep only terminal and post-run grading.
- Real task selection must be based on task quality and reproducibility, not artificial coverage of Skills, Subagent, Context, or any other feature.
- The mechanism exposure matrix records only declared opportunities and observed triggers.
- Third-party repository contents and local bare caches are not committed to TinyHarness. Only task metadata, task text, a reference patch, hidden tests, and sanitized calibration evidence are committed.
- The per-task manifest must not contain a Bash allowlist. Every first-round SWE task uses the same task-independent `FROZEN_SWE_ALLOWED_BASH` tuple copied from the current public Eval configuration.
- Agent workspaces contain a shallow single-commit checkout with no configured remote. Evaluator assets, reference patches, calibration results, and task-package paths are never copied into the Agent workspace or messages.
- Evaluator isolation means outside the Agent workspace, messages, Tool schemas, and normal Harness interface. It is not OS-level sandboxing: the existing Bash Tool still runs without a process/container sandbox, and this limitation must be present in every SWE report.
- The public Event Log remains metadata-only. The local `research_trace.jsonl` is gitignored and may contain task text, assistant text, full Tool Call arguments, and full Tool Results needed for diagnosis; it must remove `reasoning_content` before persistence and must never contain evaluator assets or output.
- Existing synthetic eval commands and historical result readers remain backward compatible.
- Every task follows TDD: add a focused failing test, confirm the expected failure, implement the minimum change, run the focused test, then run the relevant regression set before committing.

---

## File Structure

### Runtime observability

- Modify `tiny_harness/agent/messages.py`: carry optional normalized provider token usage without changing the executable response contract.
- Modify `tiny_harness/models/chat_completions.py`: normalize usage fields when the provider returns them.
- Modify `tiny_harness/runtime/events.py`: add `tool_requested`, `workspace_changed`, and `workspace_observation_failed` event names; event payload policy remains metadata-only.
- Create `tiny_harness/runtime/observability.py`: produce bounded, content-safe summaries of requested Tool Call arguments.
- Modify `tiny_harness/runtime/recovery.py`: attach request context size and optional provider usage to existing model lifecycle events.
- Modify `tiny_harness/tools/registry.py`: emit attempted calls before validation and add UTF-8 byte size to Tool Result metadata.

### Eval-only trajectory and metrics

- Create `evals/trajectory.py`: use Git status plumbing to compare changed/untracked paths around existing observer Hooks, hash only already-changed files when needed to detect a second change, emit bounded `workspace_changed` events, and derive a minimal mechanism-exposure record.
- Create `evals/research_trace.py`: wrap the provider, persist model-visible requests/responses and the final message state locally, remove hidden reasoning from the persisted copy, and leave the live provider/messages untouched.
- Modify `evals/core.py`: aggregate attempted/executed/denied/errored calls, token usage availability, maximum model input size, and workspace-change counts.
- Modify `evals/pilot_metrics.py`: persist the new fields and an explicit termination reason while reading old event logs safely.

### Repository-level task foundation

- Create `evals/swe_cases.py`: strict SWE case/evaluator manifest, package validation, and local-cache resolution; no per-task Permission fields.
- Create `evals/swe_evaluator.py`: run a task-selected evaluator with fixed argv against a physical workspace copy, including optional SWE-bench-style test-patch and FAIL_TO_PASS/PASS_TO_PASS metadata.
- Create `evals/swe_runner.py`: one-commit workspace materialization, calibration, frozen-current-Runtime execution, terminal/post-run grading, research-trace persistence, and per-run ledger writing.
- Create `evals/swe.py`: `calibrate` and `run` CLI entry point.
- Create `evals/swe_tasks/README.md`: task-package contract and curation checklist.
- Modify `.gitignore`: ignore `evals/swe_cache/` and temporary SWE work directories.
- Modify `evals/README.md`: operator commands and explicit baseline limitations.

### Tests

- Create `tests/test_observability.py`.
- Modify `tests/test_chat_completions.py`.
- Modify `tests/test_events.py`.
- Modify `tests/test_evals.py`.
- Modify `tests/test_pilot_metrics.py`.
- Create `tests/test_trajectory.py`.
- Create `tests/test_research_trace.py`.
- Create `tests/test_swe_cases.py`.
- Create `tests/test_swe_evaluator.py`.
- Create `tests/test_swe_runner.py`.
- Create `tests/test_swe_cli.py`.

---

### Task 1: Add minimal model and Tool Call telemetry

**Files:**

- Create: `tiny_harness/runtime/observability.py`
- Modify: `tiny_harness/agent/messages.py`
- Modify: `tiny_harness/models/chat_completions.py`
- Modify: `tiny_harness/runtime/events.py`
- Modify: `tiny_harness/runtime/recovery.py`
- Modify: `tiny_harness/tools/registry.py`
- Create: `tests/test_observability.py`
- Modify: `tests/test_chat_completions.py`
- Modify: `tests/test_events.py`
- Modify: `tests/test_recovery.py`
- Modify: `tests/test_registry.py`

**Interfaces:**

- Produces: `ModelUsage(prompt_tokens: int | None, completion_tokens: int | None, total_tokens: int | None)`.
- Produces: optional `ModelResponse.usage: ModelUsage | None = None`; the last-position default preserves all existing positional constructors.
- Produces: `safe_tool_call_data(workspace: Path, call: ToolCall) -> dict[str, Any]`.
- Produces event: `tool_requested` with `tool_call_id`, `tool_name`, JSON size/hash, parse status, and a bounded `arguments` summary.
- Extends `model_requested` with `input_chars`, `message_count`, and `tool_schema_count`.
- Extends `model_responded` with optional token counts.
- Extends `tool_finished` with `content_bytes`; retain `content_length` as the character count for backward compatibility.
- Extends denied and hook-blocked events with the size of the Tool Result feedback returned to the model.
- Does not expose file content, edit text, task prompts, Todo text, reasoning, final answers, secrets, or raw unsafe Bash commands.

- [ ] **Step 1: Add failing safe-summary tests**

Create `tests/test_observability.py` with focused cases for every argument shape:

```python
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tiny_harness.agent.messages import ToolCall
from tiny_harness.runtime.observability import safe_tool_call_data


class SafeToolCallDataTest(unittest.TestCase):
    def test_file_write_keeps_relative_path_but_not_content(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            event = safe_tool_call_data(
                workspace,
                ToolCall(
                    "write-1",
                    "write_file",
                    '{"path":"src/app.py","content":"PRIVATE_BODY"}',
                ),
            )

        self.assertTrue(event["arguments_parseable"])
        self.assertEqual(
            event["arguments"],
            {"path": "src/app.py", "content_chars": 12},
        )
        self.assertNotIn("PRIVATE_BODY", repr(event))

    def test_outside_path_and_bash_values_are_not_recorded_verbatim(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            path_event = safe_tool_call_data(
                workspace,
                ToolCall("read-1", "read_file", '{"path":"../private.txt"}'),
            )
            bash_event = safe_tool_call_data(
                workspace,
                ToolCall(
                    "bash-1",
                    "bash",
                    '{"command":"TOKEN=PRIVATE_VALUE python script.py"}',
                ),
            )

        self.assertEqual(path_event["arguments"]["path_scope"], "outside_workspace")
        self.assertNotIn("private.txt", repr(path_event))
        self.assertEqual(bash_event["arguments"]["command_head"], "TOKEN")
        self.assertIn("command_sha256", bash_event["arguments"])
        self.assertNotIn("PRIVATE_VALUE", repr(bash_event))

    def test_invalid_json_is_still_a_bounded_attempt_record(self) -> None:
        with TemporaryDirectory() as directory:
            event = safe_tool_call_data(
                Path(directory),
                ToolCall("bad-1", "write_file", "NOT_JSON_PRIVATE"),
            )

        self.assertFalse(event["arguments_parseable"])
        self.assertEqual(event["arguments_json_chars"], 16)
        self.assertIn("arguments_sha256", event)
        self.assertNotIn("NOT_JSON_PRIVATE", repr(event))
```

- [ ] **Step 2: Run the new test and verify the expected import failure**

Run:

```powershell
python -m pytest tests/test_observability.py -q
```

Expected: collection fails because `tiny_harness.runtime.observability` does not exist.

- [ ] **Step 3: Implement the bounded summary contract**

Create `tiny_harness/runtime/observability.py` with the public functions `safe_tool_call_data(workspace: Path, call: ToolCall) -> dict[str, Any]` and `safe_workspace_path(workspace: Path, value: object) -> dict[str, str]`.

Use these summaries:

- `read_file`, `list_files`: workspace-relative `path`, or `path_scope=outside_workspace` plus a SHA-256 hash.
- `write_file`: safe path plus `content_chars`.
- `edit_file`: safe path plus `old_text_chars` and `new_text_chars`.
- `bash`: `command_chars`, `command_sha256`, first ASCII word as `command_head`, and booleans `has_control_operator`, `has_parent_reference`, `has_absolute_path`; never store the raw command.
- `run_tests`, `compact`: empty object.
- `load_skill`: `name` only when it is a bounded `[A-Za-z0-9_-]{1,80}` value; otherwise store length/hash.
- `task`: `prompt_chars` and `prompt_sha256`.
- `todo_write`: `todo_count` and status counts; never store Todo content.
- Unknown tools: sorted argument names and JSON value types only.
- Invalid JSON: `arguments_parseable=false`, JSON character count, and SHA-256 only.

Limit every emitted list to 50 items and every retained scalar string to 120 characters. Unit tests must assert that sentinels never appear in `repr(event)`.

- [ ] **Step 4: Add failing provider usage normalization tests**

In `tests/test_chat_completions.py`, add one response with:

```python
usage=SimpleNamespace(
    prompt_tokens=101,
    completion_tokens=23,
    total_tokens=124,
)
```

Assert:

```python
self.assertEqual(result.usage.prompt_tokens, 101)
self.assertEqual(result.usage.completion_tokens, 23)
self.assertEqual(result.usage.total_tokens, 124)
```

Add a second test asserting `result.usage is None` when the SDK response has no `usage` attribute.

- [ ] **Step 5: Implement optional usage without changing response validation**

In `tiny_harness/agent/messages.py`, add:

```python
@dataclass(frozen=True)
class ModelUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
```

Add `usage: ModelUsage | None = None` as the final `ModelResponse` field. In `chat_completions.py`, normalize only non-negative integer SDK values; bools and malformed values become `None`. If the entire usage object is absent, return `usage=None`.

- [ ] **Step 6: Add failing lifecycle-event tests**

Update `tests/test_events.py`, `tests/test_recovery.py`, and `tests/test_registry.py` to assert:

```python
self.assertEqual(tool_event_names, ["tool_requested", "tool_started", "tool_finished"])
self.assertEqual(denied_event_names, ["tool_requested", "tool_denied"])
self.assertEqual(invalid_event_names, ["tool_requested", "tool_finished"])
self.assertEqual(requested["input_chars"], context_char_count(messages, tools))
self.assertEqual(responded["prompt_tokens"], 101)
self.assertEqual(finished["content_bytes"], len(content.encode("utf-8")))
```

Retain and strengthen the existing payload-leak test so the complete JSON serialization contains none of the file-content, task-prompt, reasoning, final-answer, or unsafe-command sentinels.

- [ ] **Step 7: Emit the new telemetry at existing lifecycle boundaries**

- Add `TOOL_REQUESTED`, `WORKSPACE_CHANGED`, and `WORKSPACE_OBSERVATION_FAILED` to `EventType`.
- In `dispatch`, emit `TOOL_REQUESTED` before adapter lookup and argument validation.
- Do not move Pre Hook, Permission, Handler, or Post Hook ordering.
- Add request size fields in `RecoveryExecutor.complete` using the existing `context_char_count(messages, tools)` helper.
- Add optional usage keys only when `response.usage` exists; do not write misleading zeros.
- Add `content_bytes` to every `TOOL_FINISHED` path, including malformed calls and handler exceptions.
- Build denial and hook-block feedback once, then record its character/UTF-8 byte sizes on `TOOL_DENIED` or `TOOL_HOOK_BLOCKED` before returning the same unchanged text.

- [ ] **Step 8: Run focused and full regressions**

Run:

```powershell
python -m pytest tests/test_observability.py tests/test_chat_completions.py tests/test_events.py tests/test_recovery.py tests/test_registry.py -q
python -m pytest -q
```

Expected: all tests pass; existing event-order assertions are updated only by insertion of `tool_requested`.

- [ ] **Step 9: Commit the first independently reviewable unit**

```powershell
git add tiny_harness/agent/messages.py tiny_harness/models/chat_completions.py tiny_harness/runtime/events.py tiny_harness/runtime/observability.py tiny_harness/runtime/recovery.py tiny_harness/tools/registry.py tests/test_observability.py tests/test_chat_completions.py tests/test_events.py tests/test_recovery.py tests/test_registry.py
git commit -m "feat: add minimal runtime trajectory telemetry"
```

---

### Task 2: Add the local research trace, Git-based mutations, and minimal run metrics

**Files:**

- Create: `evals/trajectory.py`
- Create: `evals/research_trace.py`
- Modify: `evals/core.py`
- Modify: `evals/pilot_metrics.py`
- Create: `tests/test_trajectory.py`
- Create: `tests/test_research_trace.py`
- Modify: `tests/test_evals.py`
- Modify: `tests/test_pilot_metrics.py`

**Interfaces:**

- Produces: `GitPathState(status: str, path: str, content_sha256: str | None)`.
- Produces: `capture_git_state(workspace: Path) -> dict[str, GitPathState]` using Git status plumbing rather than a full repository walk.
- Produces: `GitWorkspaceMutationTracker(workspace: Path, event_logger: EventLogger)` with `pre(context) -> None`, `post(context, result) -> None`, and read-only `failed: bool` Hook-observer state.
- Produces: `ResearchTraceRecorder(path: Path, *, workspace: Path)` with read-only `failed: bool`, `ResearchTraceProvider(provider, recorder)`, and `strip_reasoning(value: Any) -> Any`.
- Produces: `observed_mechanisms(events) -> dict[str, bool]` for direct event presence only.
- Extends `RunMetrics` with attempted/executed/denied/errored calls, token sums and missing-usage count, maximum input chars, and workspace-change count.
- Does not copy historical workspace snapshots or invoke any grader. Full local trace data never enters the public Event Log or report.

- [ ] **Step 1: Add failing manifest-diff and observer tests**

Create `tests/test_trajectory.py`:

```python
def test_mutation_tracker_uses_git_paths_without_persisting_content(self) -> None:
    logger = RecordingEventLogger()
    tracker = GitWorkspaceMutationTracker(self.workspace, logger)
    context = ToolHookContext(
        tool_call_id="write-1",
        tool_name="write_file",
        arguments=MappingProxyType({"path": "src/app.py", "content": "SECRET"}),
    )

    tracker.pre(context)
    path = self.workspace / "src" / "app.py"
    path.parent.mkdir()
    path.write_text("PRIVATE_FILE_BODY", encoding="utf-8")
    tracker.post(context, ToolResult("write-1", "Wrote file"))

    changed = logger.events[-1]
    self.assertEqual(changed["event_type"], "workspace_changed")
    self.assertEqual(changed["data"]["added"], ["src/app.py"])
    self.assertNotIn("PRIVATE_FILE_BODY", repr(changed))
```

The test setup initializes and commits a Git repository before calling `tracker.pre`. Also test tracked modification, deletion, untracked addition, a second edit to an already-modified path, no-change suppression, rename detection disabled, and maximum 200 reported paths with `omitted_path_count`. Patch `subprocess.run` to fail and assert the observer records `workspace_observation_failed` metadata, sets `failed=True`, and does not raise into the Agent Loop.

- [ ] **Step 2: Run the new test and verify it fails**

```powershell
python -m pytest tests/test_trajectory.py -q
```

Expected: import failure because `evals.trajectory` does not exist.

- [ ] **Step 3: Implement the low-overhead Git mutation observer**

Run this fixed command with `shell=False`, workspace cwd, captured bytes, and a short timeout:

```text
git -c status.renames=false status --porcelain=v1 -z --untracked-files=all
```

Parse only status codes and relative paths. To detect a second edit while a path remains `M` or `??`, calculate SHA-256 only for non-deleted paths already returned by Git status. Do not recursively hash clean tracked files and do not capture full `git diff` output. Snapshot only around tools that can reasonably mutate the workspace:

```python
MUTATION_CAPABLE_TOOLS = frozenset(
    {"write_file", "edit_file", "bash", "run_tests"}
)
```

Do not observe around `task`; child Tool Calls use the same hooks and generate their own records. Compare pre/post `GitPathState` values and emit one `workspace_changed` event only when paths changed during that Tool Call. Record bounded `added`, `modified`, `deleted`, and `untracked` relative paths; never record file bytes or hashes in the public event. This deliberately observes only Git-visible tracked/untracked paths at the pre/post boundaries: ignored files and a path created and deleted within one Tool Call are not guaranteed to appear. Record that limitation in the run report instead of adding a filesystem watcher. Catch Git/status/hash failures inside the observer, emit `workspace_observation_failed` with only stage and exception type, and set `failed=True`; the SWE evaluator later marks that run invalid without interrupting or steering the Agent.

- [ ] **Step 4: Add failing local research-trace tests**

Create `tests/test_research_trace.py` with a fake provider response containing assistant text, `reasoning_content`, and a Tool Call with a full Bash command. Record a subsequent model request containing the resulting full Tool Result, then call `record_final_messages`.

Assert:

```python
self.assertIn("python -m pytest tests/test_widget.py -q", trace_text)
self.assertIn("FULL_TOOL_RESULT_SENTINEL", trace_text)
self.assertIn("assistant explanation", trace_text)
self.assertNotIn("PRIVATE_REASONING_SENTINEL", trace_text)
self.assertIs(returned_response, original_response)
self.assertEqual(messages, original_messages)
```

Add a separate test passing a public `RecordingEventLogger` and prove the full Bash command and Tool Result appear only in `research_trace.jsonl`, never in serialized public events.

- [ ] **Step 5: Implement the gitignored research trace without touching live messages**

`ResearchTraceProvider.complete` must:

1. deep-copy the model-visible request messages and Tool schemas, recursively remove every `reasoning_content` key, and write the result as a `model_request` record;
2. call the wrapped provider with the original `messages` and `tools` objects;
3. write `model_response` with assistant `content`, full Tool Call IDs/names/arguments, finish reason, and usage, but omit `reasoning_content`;
4. on provider failure, write only exception type and re-raise the same exception;
5. return the original `ModelResponse` unchanged.

`record_final_messages` writes a reasoning-stripped deep copy so the final Tool Result is preserved even when max turns are exhausted immediately after a tool batch. JSONL writes flush synchronously. The recorder path must resolve outside the Agent workspace; constructor validation rejects a path inside it. All persistence methods catch local serialization/I/O errors, set `failed=True`, and return without changing the live call. The SWE runner marks such a run invalid after the Agent stops; trace failure must not alter prompts, Tool Results, Permission decisions, model inputs, provider exceptions, or returned responses.

- [ ] **Step 6: Add failing metric aggregation tests**

Extend the `collect_metrics` fixture in `tests/test_evals.py` with requested, denied, returned, errored, model-usage, context-size, and workspace-change events. Assert exact values:

```python
self.assertEqual(metrics.tool_attempts, 4)
self.assertEqual(metrics.tool_executions, 3)
self.assertEqual(metrics.tool_denials, 1)
self.assertEqual(metrics.tool_errors, 1)
self.assertEqual(metrics.prompt_tokens, 120)
self.assertEqual(metrics.completion_tokens, 30)
self.assertEqual(metrics.total_tokens, 150)
self.assertEqual(metrics.usage_missing_responses, 1)
self.assertEqual(metrics.max_input_chars, 18_000)
self.assertEqual(metrics.workspace_change_events, 2)
```

- [ ] **Step 7: Implement backward-compatible metrics and termination**

Add these fields to `RunMetrics`, all defaulting to zero:

```python
tool_attempts: int = 0
tool_executions: int = 0
tool_denials: int = 0
tool_errors: int = 0
prompt_tokens: int = 0
completion_tokens: int = 0
total_tokens: int = 0
usage_responses: int = 0
usage_missing_responses: int = 0
max_input_chars: int = 0
workspace_change_events: int = 0
workspace_observation_failures: int = 0
```

Derive termination in `build_run_ledger` as exactly one of:

- `natural_final_answer`
- `max_turns_exceeded`
- `runtime_error`
- `invalid`

Keep all existing ledger fields. Add `usage_complete = usage_missing_responses == 0 and usage_responses > 0`. Old event logs must summarize successfully with zero counts and `usage_complete=false`.

Replace the misleading `bash_test_denied` field with additive fields `bash_denied` and `tool_denied`; retain `bash_test_denied` only as a deprecated alias when reading/writing schema version 2 records. Increment the ledger schema version to 3.

- [ ] **Step 8: Add minimal mechanism exposure output**

Implement direct boolean observation only:

```python
{
    "permission": any(tool_denied),
    "context": any(context_trimmed or context_compacted),
    "recovery": any(model_retry_scheduled),
    "skills": any(tool_requested where tool_name == "load_skill"),
    "verification": any(tool_requested where tool_name == "run_tests"),
    "subagent": any(tool_requested where tool_name == "task"),
    "todo": any(tool_requested where tool_name == "todo_write"),
}
```

Do not infer phases, intent, causality, quality, or usefulness.

- [ ] **Step 9: Run focused, offline, and full regressions**

```powershell
python -m pytest tests/test_trajectory.py tests/test_research_trace.py tests/test_evals.py tests/test_pilot_metrics.py -q
python -m evals.run --suite offline --results-dir .\evals\results\foundation-offline
python -m pytest -q
```

Expected: tests and offline suite pass; the generated offline report contains no private payloads.

- [ ] **Step 10: Commit the eval observability unit**

```powershell
git add evals/trajectory.py evals/research_trace.py evals/core.py evals/pilot_metrics.py tests/test_trajectory.py tests/test_research_trace.py tests/test_evals.py tests/test_pilot_metrics.py
git commit -m "feat: capture minimal eval trajectory evidence"
```

---

### Task 3: Define and validate repository-level SWE task packages

**Files:**

- Create: `evals/swe_cases.py`
- Create: `evals/swe_tasks/README.md`
- Modify: `.gitignore`
- Create: `tests/test_swe_cases.py`

**Interfaces:**

- Produces: immutable `SWECase` and `SWECaseBundle` dataclasses.
- Produces: `load_swe_case(case_root: Path) -> SWECaseBundle`.
- Produces: `resolve_repository_cache(case: SWECase, cache_root: Path) -> Path`.
- A task package contains `case.json`, `task.md`, `reference.patch`, and an evaluator-owned `evaluator/` directory. A SWE-bench-style package also contains its original `test.patch` and FAIL_TO_PASS/PASS_TO_PASS identifiers.
- Repository source is a local bare Git cache at `evals/swe_cache/{repository_cache_key}.git`, not a committed source copy.

- [ ] **Step 1: Add failing manifest contract tests**

Create `tests/test_swe_cases.py` covering one valid package and explicit failures for every unsafe condition. The valid manifest shape is:

```json
{
  "schema_version": 1,
  "id": "owner_repo_issue_123",
  "repository_url": "https://github.com/owner/repo.git",
  "repository_cache_key": "owner-repo",
  "base_commit": "0123456789abcdef0123456789abcdef01234567",
  "max_turns": 20,
  "max_context_chars": null,
  "public_test_argv": ["python", "-m", "pytest", "-q"],
  "evaluator": {
    "kind": "command",
    "argv": ["python", "evaluator/grade.py"],
    "timeout_seconds": 120
  },
  "mechanism_opportunities": []
}
```

Tests must reject:

- unknown fields or schema versions;
- duplicate or unsafe IDs/cache keys;
- non-HTTPS GitHub provenance URLs;
- commits that are not exactly 40 lowercase hexadecimal characters;
- any `allowed_bash` field, which is not part of the per-task schema;
- empty, non-string, or NUL-containing argv entries;
- absolute/traversing paths in package members;
- missing or empty `task.md`, `reference.patch`, or evaluator entry point;
- evaluator/reference paths that resolve through symlinks;
- `kind="swe_bench"` without `test.patch`, a non-empty FAIL_TO_PASS list, and an explicit PASS_TO_PASS list;
- command evaluators that incorrectly declare SWE-bench-only fields;
- unsupported mechanism names.

- [ ] **Step 2: Run the manifest tests and verify import failure**

```powershell
python -m pytest tests/test_swe_cases.py -q
```

Expected: collection fails because `evals.swe_cases` does not exist.

- [ ] **Step 3: Implement immutable package contracts**

Use these exact dataclasses:

```python
@dataclass(frozen=True)
class SWEEvaluator:
    kind: str
    argv: tuple[str, ...]
    timeout_seconds: int
    fail_to_pass: tuple[str, ...] = ()
    pass_to_pass: tuple[str, ...] = ()


@dataclass(frozen=True)
class SWECase:
    id: str
    repository_url: str
    repository_cache_key: str
    base_commit: str
    task: str
    max_turns: int
    max_context_chars: int | None
    public_test_argv: tuple[str, ...]
    evaluator: SWEEvaluator
    mechanism_opportunities: tuple[str, ...]


@dataclass(frozen=True)
class SWECaseBundle:
    root: Path
    case: SWECase
    reference_patch: Path
    evaluator_root: Path
    test_patch: Path | None
```

Allowed mechanism names are exactly `tools`, `permission`, `context`, `recovery`, `skills`, `verification`, `subagent`, and `todo`. The list is descriptive only and must never change Runtime configuration. `load_swe_case` must reject `allowed_bash` as an unknown field rather than silently ignoring it.

- [ ] **Step 4: Document the package and cache boundary**

In `evals/swe_tasks/README.md`, document:

- how a curator chooses a real merged task;
- the exact package files and manifest fields;
- how to create the bare cache with `git clone --mirror` outside Agent execution;
- why future commits/remotes/evaluator assets are absent from Agent workspaces;
- licensing/provenance requirements;
- the base-public-pass/base-evaluator-fail/reference-public-pass/reference-evaluator-pass calibration contract;
- the evaluator contract: fixed argv with `shell=False`, private output discarded, exit code 0 PASS, exit code 1 task FAIL, all other exits/timeouts evaluator-invalid;
- how a `command` evaluator invokes the selected repository's native test framework rather than converting it to unittest;
- how a `swe_bench` evaluator retains `test.patch`, FAIL_TO_PASS, and PASS_TO_PASS semantics;
- that `mechanism_opportunities` describes natural opportunities only and must not influence task selection.

Add `evals/swe_cache/`, `evals/swe_work/`, and `**/research_trace.jsonl` to `.gitignore`. The trace normally lives under the already ignored `evals/results/`, but the explicit pattern prevents accidental staging if a custom result root is used inside the repository.

- [ ] **Step 5: Run focused and full tests**

```powershell
python -m pytest tests/test_swe_cases.py -q
python -m pytest -q
```

Expected: all tests pass; no real repository or cache is required by the unit suite.

- [ ] **Step 6: Commit the task-package contract**

```powershell
git add .gitignore evals/swe_cases.py evals/swe_tasks/README.md tests/test_swe_cases.py
git commit -m "feat: define calibrated SWE task packages"
```

---

### Task 4: Add isolated materialization, calibration, and current-Runtime execution

**Files:**

- Create: `evals/swe_evaluator.py`
- Create: `evals/swe_runner.py`
- Create: `evals/swe.py`
- Modify: `evals/README.md`
- Create: `tests/test_swe_evaluator.py`
- Create: `tests/test_swe_runner.py`
- Create: `tests/test_swe_cli.py`

**Interfaces:**

- Produces: `PreparedSWECase`, `SWEGradeResult`, and `SWEFinalAnswerGradingEventLogger` without changing the existing synthetic grader.
- Produces: `evaluate_swe_snapshot(prepared: PreparedSWECase) -> SWEGradeResult` using the selected task's fixed evaluator argv and native test semantics.
- Produces: `prepare_swe_case(bundle, cache_root, results_root, repetition) -> PreparedSWECase`.
- Produces: `calibrate_swe_case(bundle, cache_root, output_path, repetitions=3) -> dict[str, Any]`.
- Produces: `run_swe_case(bundle, cache_root, results_root, repetition, provider) -> EvalResult`.
- Produces CLI subcommands `python -m evals.swe calibrate` and `python -m evals.swe run` with the arguments specified in Step 10.
- Defines the task-independent constant `FROZEN_SWE_ALLOWED_BASH = ("python -m unittest discover -s tests -v",)`, exactly matching every current public Pilot case.
- Reuses the current `_configured_test_runner(FROZEN_SWE_ALLOWED_BASH)`, unmodified `EvalPermissionPolicy(FROZEN_SWE_ALLOWED_BASH)`, trajectory hooks, and Pilot ledger/report functions.

- [ ] **Step 1: Add failing one-commit materialization tests**

In `tests/test_swe_runner.py`, build a temporary bare Git repository with two commits. Set the manifest base to the first commit and assert that `prepare_swe_case`:

- checks out exactly the pinned tree;
- leaves `HEAD` detached;
- makes the second commit unreachable from workspace refs;
- leaves `git remote` empty;
- creates only `run_root/workspace`, `events.jsonl` when execution begins, and grade/ledger artifacts after execution;
- keeps `evaluator_root`, optional `test.patch`, and `reference.patch` outside `run_root` and outside `workspace`;
- never writes the evaluator source path into workspace files.

- [ ] **Step 2: Run the materialization test and verify it fails**

```powershell
python -m pytest tests/test_swe_runner.py::SWEPreparationTest -q
```

Expected: import failure because `evals.swe_runner` does not exist.

- [ ] **Step 3: Implement local-cache materialization with fixed argv**

Use `subprocess.run` with an argv list, `shell=False`, and `check=True` for these operations, where the named values come directly from the validated bundle and CLI paths:

```text
git init $workspacePath
git -C $workspacePath fetch --depth 1 --no-tags $bareCachePath $baseCommit
git -C $workspacePath checkout --detach FETCH_HEAD
```

Do not call the network and do not derive commands through shell strings. Reject a missing cache or commit before creating a run directory. Point `PreparedSWECase.evaluator_root` directly at the evaluator-owned package directory; do not copy it into the run directory.

- [ ] **Step 4: Add failing evaluator-adapter tests**

Create `tests/test_swe_evaluator.py` with temporary workspace/evaluator packages for both supported kinds:

- `command`: the evaluator runner invokes that repository's native test command, writes a private sentinel to stdout, and exits 0/1; assert only pass/exit/digest metadata returns and the sentinel is discarded.
- `swe_bench`: the evaluator receives the copied `test.patch`, FAIL_TO_PASS, and PASS_TO_PASS metadata through an evaluator-only context file, applies the patch only to the physical grading snapshot, and evaluates the declared tests; assert the live Agent workspace is unchanged.
- evaluator exit 0 means PASS, exit 1 means task FAIL, exit 2 or timeout means evaluator-invalid.
- evaluator asset mutation or symlink is evaluator-invalid.

The adapter always copies the Agent workspace and evaluator assets to a temporary grading root and writes `evaluator_context.json` there. That context contains only the evaluator kind, grading-snapshot-relative paths, an optional test-patch-relative path, and the exact FAIL_TO_PASS/PASS_TO_PASS lists. Export its path and the snapshot path only to the evaluator subprocess, run the manifest argv with `shell=False`, and return no evaluator stdout/stderr. The file and environment variables do not exist while the Agent is running and are never copied into the live workspace, messages, public ledger, or research trace.

- [ ] **Step 5: Implement evaluator-owned terminal/post-run grading**

Implement `SWEFinalAnswerGradingEventLogger` with the same root-only `RUN_FINISHED` trigger and observation-only behavior as the existing `FinalAnswerGradingEventLogger`, but call `evaluate_swe_snapshot`. It must never append evaluator results to Agent messages or research trace. `run_swe_post_run_grade` repeats the same evaluator after Agent termination and writes sanitized `post_run_grade.json`.

- [ ] **Step 6: Add failing calibration tests**

Create deterministic temporary repositories and fake test/grader executors for these outcomes:

1. base public PASS three times, base evaluator FAIL three times, reference public PASS three times, reference evaluator PASS three times → calibration valid;
2. flaky base public result → invalid with `base_public_unstable`;
3. base evaluator PASS → invalid with `base_evaluator_not_discriminating`;
4. reference patch fails `git apply --check` → invalid with `reference_patch_invalid`;
5. reference evaluator FAIL → invalid with `reference_evaluator_failed`;
6. evaluator assets change during calibration → invalid with `evaluator_changed`.

Assert that `calibration.json` contains only exit codes, booleans, elapsed time, repository/test/patch/grader digests, Python/platform metadata, and failure codes—never grader stdout/stderr or source content.

- [ ] **Step 7: Implement three-pass calibration**

Calibration order is fixed:

```text
validate package and cache
materialize base
run public_test_argv 3 times             -> all PASS
run selected evaluator 3 times           -> all valid FAIL
git apply --check reference.patch
git apply reference.patch
run public_test_argv 3 times             -> all PASS
run selected evaluator 3 times           -> all valid PASS
verify evaluator/test-patch digest unchanged
write sanitized calibration.json
```

Calibration never calls the model. Any timeout, inconsistent exit code, source mutation by tests, or evaluator invalidity makes calibration invalid. A SWE-bench-style calibration additionally records the test-patch digest and exact FAIL_TO_PASS/PASS_TO_PASS lists.

- [ ] **Step 8: Add failing frozen-current-Runtime runner tests**

Mock `agent_loop` and assert exact configuration:

```python
self.assertEqual(call.kwargs["max_turns"], case.max_turns)
self.assertEqual(call.kwargs["max_context_chars"], case.max_context_chars)
self.assertTrue(call.kwargs["allow_subagent"])
self.assertEqual(call.kwargs["subagent_max_turns"], case.max_turns)
self.assertEqual(call.kwargs["recovery_policy"].max_retries, 2)
self.assertEqual(
    call.kwargs["test_runner"].argv,
    ("python", "-m", "unittest", "discover", "-s", "tests", "-v"),
)
self.assertIs(
    call.kwargs["permission_policy"].decide("run_tests", {}),
    PermissionDecision.ALLOW,
)
self.assertIs(
    call.kwargs["permission_policy"].decide(
        "bash", {"command": "python -m pytest -q"}
    ),
    PermissionDecision.DENY,
)
```

Also assert that `EvalPermissionPolicy.__init__` is unchanged, the system prompt is byte-for-byte `_system_prompt()` from the current eval, no Memory flag is enabled, no new Tool schema is injected, and observer Hooks do not block or rewrite calls. The task's `public_test_argv` and evaluator argv must never configure the Agent-facing `run_tests` Tool or Permission policy.

- [ ] **Step 9: Implement the SWE runner by composing existing Runtime pieces**

Mirror `run_real_case` outcome handling rather than introducing a second definition of verified/premature/explicit/invalid. Wrap the provider with `ResearchTraceProvider`, construct `GitWorkspaceMutationTracker`, register its observer callbacks on a fresh `ToolHooks`, and pass those hooks through the existing `tool_hooks=` parameter. In a `finally` block, call `record_final_messages(messages)` so a max-turn final tool batch is retained. If either observer failed, mark the evaluator result invalid while preserving the Agent's original termination and artifacts. Do not attach these observers to the legacy Pilot in this phase. Required per-run artifacts are:

```text
runs/{case-id}-harness-{repetition}/
├── workspace/
├── events.jsonl
├── research_trace.jsonl        # gitignored local diagnostic artifact
├── terminal_grade.json
├── post_run_grade.json
├── run.json
└── mechanism_exposure.json
```

`mechanism_exposure.json` contains the manifest's declared opportunity list and direct observed booleans only. It must not contain a human or model-generated diagnosis.

Use only `FROZEN_SWE_ALLOWED_BASH` to build both `EvalPermissionPolicy` and the existing unittest-based `run_tests` capability. This is intentionally independent of the selected task's native test framework. A denied pytest command, an irrelevant unittest run, or inability to verify is baseline evidence and must not trigger a fallback policy or task-specific allowlist.

Every summary/report must include: `Evaluator assets are outside the Harness interface, but Bash is not OS-sandboxed; filesystem confidentiality is not guaranteed against a deliberately escaping shell command.` It must also state that mutation telemetry uses Git-visible pre/post state and may miss ignored files or paths created and deleted within one Tool Call.

- [ ] **Step 10: Add and test the CLI**

Implement:

```powershell
python -m evals.swe calibrate `
  --task-dir .\evals\swe_tasks\owner_repo_issue_123 `
  --repo-cache .\evals\swe_cache `
  --output .\evals\swe_tasks\owner_repo_issue_123\calibration.json

python -m evals.swe run `
  --task-dir .\evals\swe_tasks\owner_repo_issue_123 `
  --repo-cache .\evals\swe_cache `
  --repetitions 1 `
  --results-dir .\evals\results\swe-smoke
```

CLI exit codes:

- `0`: valid calibration, or all requested runs produced valid evaluator outcomes regardless of task success;
- `1`: invalid calibration/run;
- `2`: argparse/configuration error.

The real `run` command requires `TINYHARNESS_API_KEY`; `calibrate` never does.

- [ ] **Step 11: Run focused, offline, and full tests**

```powershell
python -m pytest tests/test_swe_evaluator.py tests/test_swe_runner.py tests/test_swe_cli.py tests/test_evals.py -q
python -m evals.run --suite offline --results-dir .\evals\results\foundation-offline-2
python -m pytest -q
```

Expected: all tests pass; no test accesses the network or requires an API key.

- [ ] **Step 12: Commit the runnable foundation**

```powershell
git add evals/swe_evaluator.py evals/swe_runner.py evals/swe.py evals/README.md tests/test_swe_evaluator.py tests/test_swe_runner.py tests/test_swe_cli.py tests/test_evals.py
git commit -m "feat: add calibrated SWE baseline runner"
```

---

### Task 5: Curate, calibrate, and smoke-run the first real repository task

**Files:**

- Create: `evals/swe_tasks/{case_id}/case.json`
- Create: `evals/swe_tasks/{case_id}/task.md`
- Create: `evals/swe_tasks/{case_id}/reference.patch`
- Create: `evals/swe_tasks/{case_id}/evaluator/grade.py` and the selected evaluator's private assets
- Create if selected instance uses SWE-bench semantics: `evals/swe_tasks/{case_id}/test.patch`
- Create: `evals/swe_tasks/{case_id}/calibration.json`
- Modify: `evals/README.md`
- Modify: `tests/test_swe_cases.py`

`{case_id}` is not a placeholder committed to the repository. During this task it is replaced everywhere with the concrete lowercase `{owner}_{repository}_{issue-or-pr-number}` selected by the curation gate before any file is staged.

**Interfaces:**

- Consumes the Task 3 package contract and Task 4 CLI.
- Produces one concrete, provenance-pinned, calibrated task package.
- Produces one non-committed smoke-run directory under `evals/results/` proving the unchanged Runtime can execute the task end to end.

- [ ] **Step 1: Select one task using the curation gate**

Review real merged upstream issues/PRs and accept the first candidate only when all conditions hold:

- OSI-compatible repository license and stable HTTPS GitHub provenance;
- exact base commit and reference commit are available;
- base repository test suite passes offline in the chosen environment;
- task requires repository navigation and at least two production-file relationships, not necessarily two changed files;
- reference patch is bounded to 1–5 production files and at most 200 changed lines excluding tests/docs;
- no network, service, database, GUI, compiler toolchain, generated tree, or platform-specific behavior is needed;
- public tests finish within 120 seconds;
- the evaluator can exercise the repository's native test framework deterministically without revealing the reference implementation;
- if the source is a SWE-bench-style instance, its original test patch and FAIL_TO_PASS/PASS_TO_PASS semantics can be retained without translation to unittest;
- task text is self-contained and does not mention Skills, Subagent, Context, or any preferred Harness behavior;
- task was not selected to trigger a specific mechanism.

Record the upstream issue/PR URL in `task.md` provenance, but present only the normalized task statement to the Agent.

- [ ] **Step 2: Create the local cache and concrete package**

Create the bare cache outside the committed task package. Resolve the one newly curated package directory and read its concrete URL/cache key from the already validated manifest:

```powershell
$caseDir = Get-ChildItem .\evals\swe_tasks -Directory |
  Where-Object { Test-Path (Join-Path $_.FullName 'case.json') } |
  Sort-Object Name |
  Select-Object -Last 1
$case = Get-Content (Join-Path $caseDir.FullName 'case.json') -Raw |
  ConvertFrom-Json
$cachePath = ".\evals\swe_cache\$($case.repository_cache_key).git"
git clone --mirror $case.repository_url $cachePath
```

Write the concrete manifest and task/evaluator assets. The manifest has no `allowed_bash`. Set `max_context_chars` to `null` for this first run to preserve the current public Eval configuration. Keep `allow_subagent=True`, Recovery retries at 2, Memory disabled, and the current Eval system prompt through runner code rather than manifest switches. The runner—not the manifest—uses the task-independent `FROZEN_SWE_ALLOWED_BASH` tuple.

Set `mechanism_opportunities` only after inspecting the untouched base repository. An empty list is valid. Do not add a Skill or modify the repository to manufacture an opportunity.

- [ ] **Step 3: Add a package regression test before calibration**

In `tests/test_swe_cases.py`, load the concrete package and assert:

```python
self.assertEqual(bundle.case.id, bundle.root.name)
self.assertEqual(len(bundle.case.base_commit), 40)
self.assertIsNone(bundle.case.max_context_chars)
self.assertTrue(bundle.reference_patch.is_file())
self.assertTrue((bundle.evaluator_root / "grade.py").is_file())
self.assertNotIn("allowed_bash", json.loads((bundle.root / "case.json").read_text()))
```

Run:

```powershell
python -m pytest tests/test_swe_cases.py -q
```

Expected: PASS before invoking any model.

- [ ] **Step 4: Calibrate and inspect the sanitized result**

```powershell
python -m evals.swe calibrate `
  --task-dir $caseDir.FullName `
  --repo-cache .\evals\swe_cache `
  --output (Join-Path $caseDir.FullName 'calibration.json')
```

Expected: exit code 0 and `valid=true`; all four three-run groups satisfy the calibration contract. Manually inspect `calibration.json` and confirm it contains no stdout/stderr, source text, absolute cache path, username, API key, or grader failure details.

- [ ] **Step 5: Run the unchanged Runtime once**

With the configured model credentials:

```powershell
python -m evals.swe run `
  --task-dir $caseDir.FullName `
  --repo-cache .\evals\swe_cache `
  --repetitions 1 `
  --results-dir .\evals\results\swe-foundation-smoke
```

Acceptance is an end-to-end **valid run**, not a verified solution. The command must produce ordered public events, gitignored `research_trace.jsonl`, terminal/post-run grades, ledger, mechanism exposure, usage availability, Tool Result sizes, and any workspace mutation events. A natural failure or max-turn result is valid baseline evidence; evaluator/configuration leakage or infrastructure failure is not.

- [ ] **Step 6: Perform the no-leak and no-policy-change audit**

Search the public result metadata for the concrete absolute evaluator path, reference patch contents, and a unique evaluator sentinel. All searches must return no matches in `events.jsonl`, `run.json`, `terminal_grade.json`, `post_run_grade.json`, and `mechanism_exposure.json`.

Inspect `research_trace.jsonl` separately. It must contain the actual Tool Call arguments, Tool Results, and assistant/tool interaction needed to explain the run; it must not contain `reasoning_content`, the evaluator path/assets/output, reference patch contents, API credentials, or environment secrets.

Compare the run configuration recorded at `run_started` with the global frozen configuration and current Eval defaults. Confirm that no per-task Bash allowance, extra Tool schema, prompt instruction, Permission allowance, completion gate, or evaluator feedback was introduced. Explicitly verify that a repository-native pytest command is still denied unless it happens to match the existing global policy, and that no fallback is applied.

- [ ] **Step 7: Run the final regression suite**

```powershell
python -m pytest -q
python -m evals.run --suite offline --results-dir .\evals\results\foundation-final-offline
python -m evals.pilot summarize .\evals\results\pilot-final-no-goal-20260831
```

Expected: all tests pass, offline invariants pass, and the existing historical Pilot can still be summarized.

- [ ] **Step 8: Document the first task without claiming success-rate evidence**

Update `evals/README.md` with:

- concrete cache, calibration, and one-run commands;
- selected upstream provenance and task ID;
- statement that one smoke run validates the foundation, not Runtime reliability;
- explicit note that no historical mutation grading, intent analytics, policy changes, per-task Bash allowlists, or new tools are present;
- explicit limitation that evaluator assets are hidden from the normal Harness interface but Bash is not OS-sandboxed;
- distinction between public metadata-only `events.jsonl` and gitignored full local `research_trace.jsonl`;
- location of gitignored run artifacts.

- [ ] **Step 9: Commit the first calibrated task**

Stage the concrete task directory, sanitized calibration record, tests, and docs. Do not stage `evals/swe_cache/`, `evals/results/`, `research_trace.jsonl`, prepared workspaces, provider responses, or raw evaluator output.

```powershell
git add evals/swe_tasks tests/test_swe_cases.py evals/README.md
git diff --cached --check
git commit -m "eval: add first calibrated repository task"
```

---

## Acceptance Criteria

The phase is complete only when all conditions are true:

1. Every model request records logical turn, physical attempt, message/tool counts, and character-sized input context.
2. Provider token usage is recorded when available and explicitly marked incomplete when absent; missing usage is never represented as zero usage.
3. Every Tool Call produces a `tool_requested` event, including unknown tools and malformed JSON.
4. Executed, denied, and errored calls can be distinguished using existing and new event types without inspecting Tool Result text.
5. Public safe argument summaries contain useful relative paths/counts/hashes but no file bodies, edit text, task prompts, Todo text, reasoning, final answers, raw unsafe commands, or outside-workspace paths.
6. Gitignored local research traces contain the full model-visible assistant/tool interaction, Tool Call arguments, and Tool Results needed for diagnosis, while removing `reasoning_content` only from the persisted copy.
7. Tool Result character and UTF-8 byte sizes are recorded in public metadata.
8. SWE Eval runs use Git status plumbing and hash only already-changed files when needed; they record bounded path changes without full repository manifests or historical grading.
9. Natural finish, max-turn exhaustion, runtime error, and invalid evaluator run have distinct termination values.
10. Terminal and post-run evaluation remain evaluator-only and never feed information back to the Agent.
11. One concrete task package passes three-run base/reference calibration and records immutable repository, patch, native test, evaluator, and optional SWE-bench test-patch digests.
12. One real model run reaches a valid evaluator outcome and emits all required artifacts; verified success is not required.
13. The first Agent workspace contains only a shallow checkout of the pinned base commit, has no remote, and contains no evaluator assets, reference patch, test patch, or calibration file.
14. The per-task manifest contains no Bash allowlist; the runner uses only the global frozen current-Eval allowlist and unmodified `EvalPermissionPolicy`.
15. Every SWE report states that evaluator assets are interface-hidden but Bash is not OS-sandboxed.
16. Existing synthetic Pilot and offline reliability suites remain runnable and semantically unchanged.
17. No Agent-facing Tool, Permission decision, Runtime policy, system prompt, or completion mechanism changes in this phase.

## Explicitly Deferred

- Remaining three repository-level tasks and the 8-run frozen baseline.
- Tool intent taxonomy and semantic command fingerprints.
- Centralized trace indexing, a trajectory query UI, public full-trace artifacts, or long-term archival policy for local research traces.
- Edit churn, phase inference, repeated-read analysis, and causal scoring.
- Per-mutation workspace copies and historical hidden grading.
- Any new search, glob, diff, test-selection, or repository-navigation Tool.
- Permission, Context, Recovery, Skills, Todo, Subagent, Verification, Memory, or stop-behavior optimization.
- Synthetic reproducer construction and Runtime A/B experiments.

## Commit Boundaries

1. `feat: add minimal runtime trajectory telemetry`
2. `feat: capture minimal eval trajectory evidence`
3. `feat: define calibrated SWE task packages`
4. `feat: add calibrated SWE baseline runner`
5. `eval: add first calibrated repository task`

Each commit must pass its focused tests and `python -m pytest -q` before the next task begins. The first four commits must not depend on an API key or network. The fifth commit requires successful offline calibration and one valid, non-committed real-model smoke run.
