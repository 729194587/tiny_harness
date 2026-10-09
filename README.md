# TinyHarness

TinyHarness is a small, synchronous coding-agent runtime for Chat Completions
providers. The aim is to keep it small enough to understand, instrument, and
change.

This is a place to test runtime ideas against real coding tasks and SWE-bench:
add a mechanism, inspect what happened, compare runs, then revise or remove it.
The interesting part is how the design changes under that scrutiny.

The runtime implements sequential tool calling, permission checks and hooks,
context compaction and recovery, skills, opt-in persistent memory, todos, and
synchronous subagents. A shared event stream supports console progress and JSONL
analysis. See the [runtime boundaries](docs/runtime.md) for the architecture.

The [offline regression suite](tests) exercises these mechanisms with scripted
providers, mocks, and temporary workspaces. [CI](.github/workflows/ci.yml) runs it
on Python 3.10 and 3.12, plus CLI help checks, without API keys or Docker. This
verifies runtime behavior; it does not measure an external model's coding ability.
The [evaluation evidence](#evaluation) below separates checked-in task inputs
from benchmark results that are not yet published.

## Questions shaping the runtime

- **Does less context mean lower cost?** Rewriting history can sacrifice prefix
  cache reuse. The design moved toward append-mostly history and compaction under
  actual context pressure, eventually removing proactive checkpoints.
  See the [cache-oriented changes](https://github.com/729194587/tiny_harness/commit/df1fd53dfd5655ff2a94e91f37570c7b630d62f1)
  and [checkpoint removal](https://github.com/729194587/tiny_harness/commit/9a7635b5ebba3a921ce74f51523f18fad6075ef5).
- **What does a summary change besides token count?** A summary becomes the
  agent's working evidence; turning a hypothesis into a fact can steer later
  decisions. The contract evolved to preserve uncertainty and then became
  factual-only. Pressure and recovery summaries retain that
  [contract](tiny_harness/runtime/context.py), without claiming that summaries
  alone explain any particular failed task.
- **Can we trust the measurement?** Evaluation checks the environment against
  baseline and reference patches before model rollout. Event reports connect
  turns, cache usage, context changes, and workspace mutations so failures can be
  investigated beyond a score. Start with the
  [evaluation workflow](evals/swe_bench_lite/README.md).
- **When is there enough evidence to act?** Repeated reads and searches can use
  up a turn budget without producing a patch. Over-exploration remains an open
  problem; removing proactive compaction does not solve it. The
  [request attribution notes](docs/context-attribution.md) explain what we can
  measure, and what those measurements cannot establish.

## Try it

From a checkout, with Python 3.10+ and a provider API key (PowerShell):

```powershell
python -m pip install -e .
$env:TINYHARNESS_API_KEY = "..."
$env:TINYHARNESS_MODEL = "deepseek-v4-flash"
$env:TINYHARNESS_BASE_URL = "https://api.deepseek.com"
python -m tiny_harness "Inspect this project and explain its test setup" --workspace .
```

Or in Bash:

```bash
python -m pip install -e .
export TINYHARNESS_API_KEY="..."
export TINYHARNESS_MODEL="deepseek-v4-flash"
export TINYHARNESS_BASE_URL="https://api.deepseek.com"
python -m tiny_harness "Inspect this project and explain its test setup" --workspace .
```

Set the model and base URL for your Chat Completions provider. Omit the task to
start an interactive session; use `--help` for options.

To verify the checkout offline after installation (PowerShell or Bash):

```bash
python -m pip install pytest
python -m pytest -q
python -m tiny_harness --help
```

## Evaluation

The SWE-bench Lite Dev pipeline calibrates baseline and reference-patch tests,
runs the agent on calibrated tasks, then grades generated patches with the
official evaluator. Its [candidate pool](evals/swe_bench_lite/dev.jsonl) and
[four-task smoke selection](evals/swe_bench_lite/selected_tasks.jsonl) are checked
in. The smoke selection is a development check, not a full-benchmark score.

No complete historical 15-task selection or graded experiment bundle is checked
in, so this README makes no resolve-rate, cache-improvement, or ranking claim.
The [evidence inventory and reproduction guide](evals/swe_bench_lite/README.md#evidence-and-provenance)
explain what is available and how to retain task selection, source revision,
configuration, official verdicts, and token/cache/context/tool metrics together.
Existing `report` and `compare` commands analyze saved events offline;
`finalize` adds official grading and a summary for a single-task run.

## Read further

The [agent loop](tiny_harness/agent/loop.py) is the implementation entry point.
[Runtime notes](docs/runtime.md) cover execution boundaries, tools, context,
skills, memory, and CLI configuration. [Working on TinyHarness](AGENTS.md)
covers contribution rules and focused tests.
