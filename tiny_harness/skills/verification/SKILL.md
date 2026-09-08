---
name: verification
description: Verify completed code changes with proportionate evidence before claiming success, using targeted checks first, broader validation only when justified, and clearly reporting anything that could not be verified.
---

# Verification

Use this workflow when a coding task appears complete and you need evidence that the requested change actually works.

Verification should increase confidence without becoming an endless search for certainty.

## 1. Identify the claim

Before running commands, state what you are trying to prove.

Examples:

- the reported bug is fixed;
- a new tool is discovered and callable;
- a refactor preserved existing behavior;
- a package still builds;
- relevant tests pass.

Choose verification that directly supports that claim.

Do not run unrelated checks merely because they are available.

## 2. Start with targeted evidence

Run the smallest check that directly exercises the changed behavior.

Examples:

- changed one tool → run that tool's tests;
- changed permission handling → run permission and dispatch tests;
- fixed one failing test → rerun that test first;
- changed a CLI option → exercise that CLI path.

Inspect the actual result, including failures, skips, warnings, and output when they matter.

Do not infer success from the absence of an obvious error.

## 3. Expand only when justified

Broaden verification when the change affects shared behavior or multiple modules.

Possible broader checks include:

- directly dependent test modules;
- integration tests;
- compilation or static checks;
- the full test suite.

Do not automatically run every available validation command after every small change.

The scope of verification should be proportional to the scope and risk of the change.

## 4. Verify important runtime behavior directly

When practical, prefer evidence from the real execution path over only structural assertions.

Examples:

- confirm a newly discovered Tool is actually visible to the model;
- confirm a Skill is visible in a new Session;
- confirm a failure can no longer be reproduced;
- confirm the command or feature behaves as expected in its normal entry point.

Unit tests and runtime checks complement each other; neither must be repeated unnecessarily.

## 5. Treat blocked verification as a result

Verification can be limited by:

- permission denial;
- missing dependencies;
- unsupported operating-system features;
- unavailable credentials or services;
- unrelated existing failures.

If a check is blocked:

1. record what prevented it;
2. preserve the evidence from checks that did run;
3. try an alternative only if it provides genuinely new evidence;
4. stop rather than repeatedly attempting equivalent commands.

Do not turn verification failure into an infinite tool loop.

## 6. Distinguish failures from environment limitations

When a test or command fails, determine what the result actually means before claiming the implementation is broken.

For example:

- a regression test failure caused by the changed code is relevant;
- a skipped symlink test on a platform without symlink permission is an environment limitation;
- an unrelated missing package may prevent a full suite from running without invalidating targeted tests.

Report these distinctions explicitly.

Do not hide real failures as environment problems, and do not report environment limitations as code regressions without evidence.

## 7. Make the final claim match the evidence

Only claim what the completed checks support.

Prefer statements such as:

- "The targeted tests passed."
- "The full suite passed with 6 platform-specific skips."
- "The runtime behavior was verified directly."
- "The change is implemented, but the full suite could not be run because dependency X is unavailable."

Avoid stronger claims than the evidence allows.

## Completion

Before saying the task is complete, be able to answer:

- What behavior was changed?
- What evidence directly verifies it?
- Were broader checks necessary?
- Did any checks fail or get skipped?
- Is anything still unverified?

Once sufficient evidence has been collected, report the result and stop.