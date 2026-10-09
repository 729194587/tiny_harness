# TinyHarness

[English](README.md) | [简体中文](README.zh-CN.md)

TinyHarness is a synchronous Coding Agent Runtime built on the Chat Completions API. It supports tool calling, permission checks, hooks, context compaction and recovery, skills, optional persistent memory, todos, and synchronous subagents.

The project uses coding tasks and SWE-bench Lite Dev to evaluate runtime design. Console output and JSONL logs share the same event stream, making it possible to inspect agent execution. See the [runtime documentation](docs/runtime.md) for architecture details.

The [offline regression tests](tests) use scripted providers, mocks, and temporary workspaces. [GitHub Actions](.github/workflows/ci.yml) runs the tests and CLI checks on Python 3.10 and 3.12.

## Design and trade-offs

- **Context and Prefix Cache.** Rewriting message history can reduce prefix cache reuse. TinyHarness therefore keeps the history mostly append-only and compacts it when context pressure reaches a threshold. Proactive checkpoints were later removed. See the [cache strategy changes](https://github.com/729194587/tiny_harness/commit/df1fd53dfd5655ff2a94e91f37570c7b630d62f1) and [checkpoint removal](https://github.com/729194587/tiny_harness/commit/9a7635b5ebba3a921ce74f51523f18fad6075ef5).
- **Summary reliability.** The agent continues working from the summary produced by compaction. To avoid promoting unverified hypotheses to facts, the summarization rules were revised to retain factual information only. Pressure-driven compaction and error recovery share these [rules](docs/runtime.md#context-and-recovery).
- **Evaluation and execution traces.** SWE-bench Lite Dev checks baseline and reference-patch test results before running the agent. Event logs record turns, cache usage, context changes, and workspace modifications for trace analysis. See the [evaluation workflow](evals/swe_bench_lite/README.md).
- **Over-exploration.** The agent sometimes repeatedly reads files and searches, using up turns without producing a patch. The [execution trace and workspace observation notes](evals/swe_bench_lite/README.md#inspect-existing-runs) can help analyze this behavior. It remains an open problem.

## Quick start

Requires Python 3.10+ and an API key for a model provider. From the repository root (PowerShell):

```powershell
python -m pip install -e .
$env:TINYHARNESS_API_KEY = "..."
$env:TINYHARNESS_MODEL = "deepseek-v4-flash"
$env:TINYHARNESS_BASE_URL = "https://api.deepseek.com"
python -m tiny_harness "Inspect this project and explain its test setup" --workspace .
```

Bash:

```bash
python -m pip install -e .
export TINYHARNESS_API_KEY="..."
export TINYHARNESS_MODEL="deepseek-v4-flash"
export TINYHARNESS_BASE_URL="https://api.deepseek.com"
python -m tiny_harness "Inspect this project and explain its test setup" --workspace .
```

Set the model name and base URL for your Chat Completions provider. Omit the task argument to start an interactive session; see `--help` for other options.

Offline tests:

```bash
python -m pip install pytest
python -m pytest -q
python -m tiny_harness --help
```

## Evaluation

The SWE-bench Lite Dev evaluation pipeline first checks baseline and reference-patch test results, then runs the agent on the tasks, and finally grades generated patches with the official evaluator.

The repository includes a candidate task pool and a four-task smoke selection for development and pipeline verification. The [evidence inventory and reproduction guide](evals/swe_bench_lite/README.md#evidence-and-provenance) links the source data and lists the task selection, source revision, configuration, official grading results, and token, cache, context, and tool metrics needed for experiments.

`report` and `compare` analyze saved execution events offline; `finalize` produces official grading results and a summary for a single-task run.

## Further reading

- [Agent Loop](tiny_harness/agent/loop.py): implementation entry point for the execution loop.
- [Runtime documentation](docs/runtime.md): execution boundaries, tools, context, skills, memory, and CLI configuration.
- [Development conventions and test commands](AGENTS.md): contribution rules and focused verification.
