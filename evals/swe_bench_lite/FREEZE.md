# Lite Test 60: phase 1 data freeze

`frozen/test60-v1/` is a fixed repo-stratified subset of the official
`SWE-bench/SWE-bench_Lite` Test split. It is not the full 300-task score.
The exact dataset revision is `b0dde1093fe417d83b7184254edf8199c1f0dff5`.
The seed is `20261015`; sampling uses only repo/instance identity and preserves
the source row order. `protocol.json` specifies allocation, SHA-256 ranking,
historical planning metadata, failure accounting and the no-replacement rule.
The authoritative execution plan is now `test60-protocol.json`: one complete,
unchanged TinyHarness configuration. All old profile/paired-order instructions
inside the immutable bundle are void and ignored. Data and checksums are unchanged.

The pinned parquet contains 300 unique instances across **12 repositories**;
the dataset card's prose says 11, so the actual rows are authoritative.

## Files and isolation

- `agent/agent_tasks.jsonl`: the only agent-visible input, with the five fields
  allowed by `SweTask`. No gold/test patches, test IDs, scripts, hints, difficulty
  or evaluator configuration are projected into it.
- `evaluator/selection.jsonl`: all original fields for the selected 60 tasks.
- `evaluator/population.jsonl`: all 300 original records, deterministically encoded.
- `evaluator/source/`: original official parquet, README, pinned API metadata and
  a source descriptor containing download URLs and checksums.
- `selection.csv`: source index, repo rank and identity. Its historical paired-order
  column is inert archival metadata, never read by the execution controller.
- `manifest.json`: revision, schema, repo counts, selected IDs, generator/decoder
  identity and artifact checksums.
- `SHA256SUMS`: every bundle file except itself, including the manifest.

Never mount the bundle root, evaluator directory, controller manifests or
calibration artifacts into the task workspace. Only the agent projection may
cross that boundary. This directory separation requires enforcement in phase 2;
it does not create an OS access boundary by itself. Public issue text is retained
verbatim, including any hints already present in that text.

Both fetch and freeze refuse to overwrite an existing output directory. A failed
operation may leave its own partial new directory; inspect it rather than replaying
into it. The checksums make changes detectable; they are not cryptographic signatures
or filesystem write protection. Changes require a new versioned freeze.

## Restoring omitted public data

Git stores the fixed selection and small original metadata (about 0.95 MB), not
the redundant population JSONL and parquet (about 7.11 MB). Before offline tests
in a fresh checkout, run:

```bash
python -B -m evals.swe_bench_lite.restore_frozen
# Or use --parquet-cache /path/to/original/public/test.parquet to avoid network.
```

The utility downloads only the revision-pinned parquet or reads an explicit
cache, checks SHA-256 and reconstructs population JSONL. All bytes must match the
unchanged original inventory. It never overwrites files or resamples tasks.
Original API metadata remain in Git and are not refetched.

## Offline reproduction

Run from the repository root in the existing evaluation Python environment
(`pyarrow==25.0.1`; no model credentials or Docker required):

```bash
python -B -m evals.swe_bench_lite.freeze freeze \
  --source evals/swe_bench_lite/frozen/test60-v1/evaluator/source \
  --output /tmp/tinyharness-test60-reproduction-new
python -B -m unittest evals.test_swe_freeze -v
(cd evals/swe_bench_lite/frozen/test60-v1 && sha256sum -c SHA256SUMS)
```

The output path must not exist. The integration test generates two independent
bundles in fresh subprocesses with different `PYTHONHASHSEED` values and compares
every byte against each other and the frozen bundle. Synthetic tests verify
checksums, forbidden-field canaries, quota behavior, source ordering, ignored
oracle fields, validation, corruption rejection and overwrite refusal.

To independently download the pinned official source into a **new** directory:

```bash
python -B -m evals.swe_bench_lite.freeze fetch \
  --revision b0dde1093fe417d83b7184254edf8199c1f0dff5 \
  --output /tmp/tinyharness-official-test-source-new
```

Fetch uses only the official Hugging Face metadata/data URLs. Freeze and tests
use local data and never import model, Docker or official grading entrypoints.

## Superseded implementation plan

The list below is an archived phase-1 plan, not an implementation requirement.
All ablation/profile/paired-order work is cancelled. Use `TEST60.md` and
`test60-protocol.json` for the implemented single-configuration controller.

1. Accept separate agent input, evaluator selection and controller manifest;
   validate all hashes and identities before running. Fix the single-instance
   calibration CLI's unconditional Dev source, and pass/record `split=test`.
2. Freeze the actual Harness source, including uncommitted/untracked files or a
   clean approved revision; lock dependencies, skills, prompts, model/version,
   endpoint, sampling parameters, recovery and turn/context/token budgets.
   A mutable hosted model alias cannot guarantee an unchanged server model.
3. Cancelled. Preserve the complete existing tool set and Agent behavior.
4. Pin image digests, batch environments without changing membership, enforce
   oracle-artifact isolation, and validate calibration against the official script
   semantics. A calibration failure remains in the denominator, without replacement.
5. Persist per-task predictions/status and exclusive launch receipts before work;
   inspect interrupted states instead of automatic replay. Preserve every attempt,
   apply the protocol's bounded service retries, and add global API/cost limits.
6. Add batch official grading/reporting bound to prediction, dataset, evaluator,
   image and configuration fingerprints. Preserve raw verdicts and separately
   classify failures/empty patches; report resolved/60 and coverage.
7. Aggregate tokens with coverage and physical retries, tools and per-task mutation
   observations. Distinguish never-mutated from unknown, and report automatic/reactive
   compaction rather than only historical working-context events.

No model execution, image preparation or official grading is authorized by this
phase-1 bundle. A concrete model configuration still requires later approval.
