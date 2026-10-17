# TinyHarness

[English](README.md) | [简体中文](README.zh-CN.md)

TinyHarness is a lightweight, synchronous Coding Agent Runtime built on the Chat Completions API. It supports tool execution with permission checks, context management, skills, optional persistent memory, and subagents.

The project uses coding tasks and SWE-bench Lite Dev to evaluate runtime behavior. Console output and JSONL logs share the same event stream for inspecting agent execution. See the [runtime documentation](docs/runtime.md) for architecture details.

## Design and trade-offs

- **Context and Prefix Cache.** TinyHarness keeps message history mostly append-only and compacts it under context pressure to preserve prefix-cache reuse. See the [cache strategy changes](https://github.com/729194587/tiny_harness/commit/df1fd53dfd5655ff2a94e91f37570c7b630d62f1).
- **Summary reliability.** Context compaction can lose information or turn unverified hypotheses into apparent facts. Summaries are designed to retain factual evidence and uncertainty. See [context and recovery](docs/runtime.md#context-and-recovery).
- **Evaluation and execution traces.** SWE-bench Lite Dev checks baseline and reference-patch test results before agent execution. Event logs capture turns, cache usage, context changes, and workspace modifications. See the [evaluation workflow](evals/swe_bench_lite/README.md).
- **Over-exploration.** The agent sometimes repeatedly searches or reads files without producing a patch. This remains an open problem; see the [trace inspection notes](evals/swe_bench_lite/README.md#inspect-existing-runs).

## Quick start

Requires Python 3.10+ and an API key for a model provider. From the repository root (PowerShell):

```powershell
python -m pip install -e .
Copy-Item .env.example .env
# Edit .env to set your API key, model, and base URL
python -m tiny_harness "Inspect this project and explain its test setup" --workspace .
```

Bash:

```bash
python -m pip install -e .
cp .env.example .env
# Edit .env to set your API key, model, and base URL
python -m tiny_harness "Inspect this project and explain its test setup" --workspace .
```

Omit the task argument to start an interactive session; see `--help` for other options.

Each run (including each interactive submission) writes a separate JSONL log to
`.tinyharness/events/` under the startup directory, named with a UTC timestamp and
a unique ID. Use `--no-event-log` to disable file logging or `--event-log PATH` to
append to a custom file; these options are mutually exclusive.

## Evaluation

TinyHarness uses SWE-bench Lite Dev to evaluate agent behavior. The workflow covers environment calibration, agent execution, and patch grading with the official evaluator.

The repository includes a candidate task pool and a four-task smoke selection for development and pipeline verification, not a published benchmark score. See the [evaluation documentation](evals/swe_bench_lite/README.md) for task selection, metrics, reproduction, and evidence limitations.

## Further reading

- [Agent Loop](tiny_harness/agent/loop.py): implementation entry point for the execution loop.
- [Runtime documentation](docs/runtime.md): execution boundaries, tools, context, skills, memory, and CLI configuration.
- [Development conventions and test commands](AGENTS.md): contribution rules and focused verification.
