# Test60 phase 2 acceptance — 2026-10-15

Objective: one unchanged, complete TinyHarness configuration on the existing
60-task Test subset. No ablation/profile/switch implementation is present.
`test60-protocol.json` is authoritative; historical paired-order instructions
in the immutable phase-1 data are void. No frozen file was rewritten.

## Implemented boundaries

| Requirement | Implementation |
| --- | --- |
| Explicit Test data/calibration/grading | Pinned bundle loader, official TestSpec script projection, explicit `split=test`; legacy Dev defaults retained |
| Private evaluator input isolation | Strict five-field agent projection; only SweTask enters rollout; private artifacts outside the sole `/testbed` mount |
| Data integrity | Pinned SHA256SUMS digest, every artifact verified, extra files rejected, ordered 60-ID agreement checked |
| Attempts/interruption | Exclusive receipt and per-task lock; immediate prediction persistence; no replay; separately recorded human review/approval for infrastructure retries |
| Official result identity | Prediction, private grading dataset, campaign, installed grader and pinned image fingerprints; saved report/log hashes checked before reuse/reporting |
| Fixed denominator | Always 60 and frozen per-repo counts; latest approved infrastructure retry is authoritative, never best-of; all attempts retained/accounted |
| Actual configuration | Source copies/hashes, dependency versions, installed provider/grader code hashes, Prompt hash, complete tool schemas/skills, model declaration and execution budgets |
| Image management preparation | Actual-host-drive capacity/reserve check; local-image inspection; exact human cleanup proposal only, with protected references and unknown shared-layer reclaim |

No Runtime source or Prompt was changed. No model client was instantiated during
actual preparation verification. Agent assembly retains the full tool set and
existing budgets/recovery semantics. The grading subprocess denies image-pull
fallback without changing installed official scoring code.

## Verification performed

- WSL Python 3.12 evaluation environment: **22 new offline acceptance tests passed**.
- Related regression group: **73 tests passed** (overlaps earlier controller tests;
  these counts are not additive). Covers freeze reproduction, pipeline, calibration,
  finalizer, reporting, Docker cleanup and installed official API/CLI compatibility.
- Frozen checksum inventory digest unchanged:
  `ed1383ec5ee43849c3de4b332266319d6e0102bf378ca373a0fd00fada5c9a87`.
  All **10** listed artifact checksums verified.
- Real dependency/source/tool preparation and revalidation tested in a temporary
  campaign, with SDK construction and Docker access forbidden by test guards.
- Model, Docker execution and official evaluation paths otherwise use controlled
  fixtures/mocks. Official script transformation is tested without running it.
- Interrupted and SIGKILL-style receipts, exhausted service failure, ordinary
  protocol failure, missing image, changed environment/prediction/report, empty
  patch, conflicting verdict, fixed denominator and mutation unknown/never cases
  covered. Cleanup proposals never execute their suggested commands.

## Limits and follow-up

This is offline infrastructure acceptance, not a Test score or live-provider
integration result. No paid request, image download/build/deletion, actual task
execution or official Test grading occurred. No historical result was overwritten;
no commit or push was made.

The supported acceptance environment is WSL. An auxiliary run in the restricted
Windows sandbox encountered directory-rename permission denials and is not claimed
as passing Windows execution support.

Choose/approve the concrete model/version before preparing a real campaign.
Mutable hosted aliases cannot be verified as immutable server models locally.
Validate actual Test images and official grading only under later authorization.
An unfinished grading directory requires manual investigation; automatic regrading
or replay is deliberately absent. Capacity estimates must come from the real host
drive/shared-layer evidence, and cleanup requires manual review of other campaigns.

Operational commands and artifacts: see `TEST60.md`.
