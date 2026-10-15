# Single-configuration SWE-bench Lite Test60

The objective is the solving ability of the current complete TinyHarness. There
are no profiles, tool switches or comparison groups. `test60-protocol.json`
supersedes obsolete planning instructions in the immutable `test60-v1` bundle.
Its 60 IDs, order and checksums remain unchanged. `freeze.py` remains the historical
byte-reproduction utility, not an execution plan. Do not regenerate Test data.

## Offline preparation

Use the WSL evaluation environment, from the repository root:

```bash
.venv/bin/python -B -m evals.swe_bench_lite.restore_frozen
.venv/bin/python -B -m evals.swe_bench_lite.benchmark --help
.venv/bin/python -B -m unittest evals.test_swe_benchmark evals.test_swe_freeze evals.test_swe_restore -v
```

After choosing an exact model/version and endpoint, create a **new** campaign:

```bash
.venv/bin/python -B -m evals.swe_bench_lite.benchmark prepare \
  evals/results/test60-new --model MODEL --model-version VERSION \
  --base-url https://PROVIDER/v1 --max-turns 40 \
  --subagent-max-turns 10 --max-context-tokens 125000
.venv/bin/python -B -m evals.swe_bench_lite.benchmark report evals/results/test60-new
```

Prepare never instantiates an API client or contacts Docker. It validates frozen
artifacts, copies actual execution source (including uncommitted/untracked Python
and bundled skills), and records dependencies, installed grader/provider source
fingerprints, original system Prompt, complete discovered tools/skills, model
declaration and budgets. Execution rechecks actual source/dependency/tool/skill
identity. A mutable hosted model alias cannot establish server model identity.
Credentials are not written to configuration. The provider's request defaults and
Runtime recovery remain unchanged; no Prompt or decoding override is added.

## Isolation and local environments

Only allowlisted `SweTask` data cross the rollout interface. Evaluator records,
calibration outputs, snapshots and the frozen bundle are outside the disposable
task workspace. The sole bind mount is `/testbed`. Calibration and rollout each
use fresh workspaces; calibration uses official TestSpec script semantics.

Images must already be local and are pinned by full ID, digests, architecture and
layer chain. Calibration/rollout use `--pull never`. Grading runs the installed
official module with explicit `test` split and a process-local SDK guard denying
automatic image pulls. It does not edit the installed package or scoring logic.

Read-only capacity and manual cleanup-plan interfaces:

```bash
.venv/bin/python -B -m evals.swe_bench_lite.benchmark capacity CAMPAIGN \
  --host-disk /mnt/d --instance-id INSTANCE --additional-bytes ESTIMATE
.venv/bin/python -B -m evals.swe_bench_lite.benchmark cleanup-plan CAMPAIGN \
  --image EXACT_IMAGE_REFERENCE
```

Use the actual VHDX backing drive, not virtual WSL capacity. The first command
checks local availability and free reserve after an operator-supplied incremental
estimate (default reserve 20 GiB). Zero estimate only checks current reserve.
Missing image/layer costs remain unknown. The second emits a plan only, **never
deletes anything**, protecting the designated three Dev images, unfinished or
unverified campaign pins and container references. Only a completed, validly graded
resolved/unresolved task releases its campaign pin for a cleanup proposal. Other
campaigns must also be checked manually. Shared-layer reclaim size remains unknown. It provides no
VHDX operation. A missing protected image fails closed for manual inspection.

## Later execution requires separate authorization

These interfaces are not actions authorized in the current implementation phase.
`run` may incur API charges; `grade` starts the official Docker evaluator:

```bash
.venv/bin/python -B -m evals.swe_bench_lite.benchmark run CAMPAIGN \
  --instance-id INSTANCE --execute
.venv/bin/python -B -m evals.swe_bench_lite.benchmark grade CAMPAIGN \
  --instance-id INSTANCE --attempt 1 --execute
```

Invoke tasks individually in frozen order within operator-defined image batches.
Missing images fail before formal receipt/API construction. Exclusive receipts
precede calibration/model construction. Each task's prediction is persisted as
single-line JSONL. Existing attempts are never silently replayed. Saved partial
artifacts are evidence, not a resumable transcript; ordinary environment cleanup
still removes temporary task workspaces.

Attempt 0001 is the default. An established infrastructure failure permits a new
numbered attempt only after a separate operator/evidence approval:

```bash
.venv/bin/python -B -m evals.swe_bench_lite.benchmark approve-retry CAMPAIGN \
  --instance-id INSTANCE --operator OPERATOR --reason EVIDENCE
```

Then an explicit `run ... --execute` creates the next attempt. Ordinary unresolved
or protocol failures cannot be retried. After unclear interruption, confirm all
worker/container operations stopped before manual review:

```bash
.venv/bin/python -B -m evals.swe_bench_lite.benchmark recover CAMPAIGN \
  --instance-id INSTANCE --operator OPERATOR --reason INFRASTRUCTURE_EVIDENCE \
  --confirm-process-stopped
```

Recovery archives stale coordination locks and records a separate review; it never
restarts work or rewrites prior results. A separate retry approval is still needed.
An unfinished grading directory is also never replayed automatically; preserve it
for investigation. Grading reuse requires matching prediction, dataset, campaign,
grader/image identity and unchanged saved report artifacts.

## Reporting

`report CAMPAIGN` is read-only JSON. The primary denominator is always 60; pending
tasks mark results provisional. Every repo keeps its frozen denominator. The
latest approved infrastructure attempt is authoritative, never the best verdict.
Costs/time include **all** attempts. Raw official evidence is preserved alongside
resolved, unresolved (including empty patch), infrastructure failure, official
evaluation error, harness/protocol failure and pending classifications.

Reports include token totals/coverage, physical requests versus logical turns,
tools by name, attempt duration, per-task first mutation root turn, service failures
and repo outcomes. Missing billed usage stays unknown, particularly on failed
requests. Never-mutated differs from incomplete observations. Service rates use
observed event records, not billing records. Attempt duration includes calibration
and rollout; official logs separately retain test timing.

Offline tests do not certify live-model reliability, all Test image compatibility,
or actual official Test results. Those require separately approved execution.
