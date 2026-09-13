---
name: systematic-debugging
description: Diagnose bugs, failing tests, errors, and unexpected behavior by reproducing the problem, gathering evidence, identifying the root cause, testing a focused hypothesis, and verifying a minimal fix.
---

# Systematic Debugging

Use this workflow when investigating a bug, failing test, runtime error, regression, or other unexpected behavior.

The goal is to understand the cause before changing code. Avoid speculative fixes that are not supported by evidence.

## 1. Reproduce the problem

Establish the smallest reliable reproduction you can.

- Read the reported error or failing assertion carefully.
- Run the directly relevant test or command when practical.
- Record the concrete failure: error message, unexpected value, stack trace, or observable behavior.
- If reproduction is blocked, state what evidence is available instead of pretending the failure was reproduced.

Do not begin editing code merely because a likely fix comes to mind.

## 2. Locate the relevant code

Narrow the search before reading large parts of the repository.

Prefer:

1. `glob` to locate files by filename or directory pattern.
2. `grep` for quick location of exact symbols, error strings, or literal occurrences.
3. `search_code` for literal search when nearby context is needed; it can avoid an extra search-to-read round trip.
4. Ranged `read_file` with `start_line`/`end_line` to read only the necessary range once a relevant location is known.
5. `bash` primarily for execution, tests, git, or queries repository tools cannot express.

Choose the tool that best answers the current question. Each exploration should answer a concrete unresolved question. Follow dependencies only as far as necessary to explain the failure, and stop confirming once evidence supports the next action.

## 3. Gather evidence

Trace the failing behavior through the relevant code path.

Look for:

- where the unexpected value or state first appears;
- assumptions made by callers and callees;
- boundary conditions and error handling;
- recent or nearby code that interacts with the failure;
- tests that define the intended behavior.

Distinguish observed facts from hypotheses.

## 4. Form one focused hypothesis

State the most likely root cause based on the available evidence.

A useful hypothesis explains:

- why the observed failure occurs;
- which code is responsible;
- what small observation or experiment could confirm or reject it.

Do not make several unrelated changes at once.

If the evidence contradicts the hypothesis, discard it and form a new one.

## 5. Test the hypothesis

Use the smallest practical experiment.

Examples:

- inspect one additional call site;
- run one focused test;
- check an intermediate value;
- reproduce the behavior with a minimal input.

The purpose is to determine whether the proposed cause is real before modifying production code.

## 6. Fix the root cause minimally

Once the cause is supported by evidence:

- change only the code necessary to correct it;
- preserve existing behavior outside the affected path;
- avoid unrelated cleanup or architecture changes;
- add or update a regression test when it provides useful protection.

Do not hide the failure with retries, broad exception handling, or special cases unless those behaviors are actually part of the intended design.

## 7. Verify the fix

Re-run the original failing scenario first.

Then run the smallest set of directly relevant regression checks needed to support the claim that the issue is fixed.

Broaden verification only when the scope or risk of the change justifies it.

If verification is blocked by permissions, unavailable tools, environment limitations, or unrelated failures:

- report the limitation clearly;
- report what was successfully verified;
- do not repeatedly try equivalent commands without new evidence.

## Completion

Before concluding, be able to explain:

- what failed;
- what the root cause was;
- what changed;
- what evidence supports the fix;
- what remains unverified, if anything.
