# Formal Skills Phase 2 Implementation Plan

> **For Codex:** Follow this plan with test-driven development. Keep the work limited to Skills and stop after verification and reporting.

**Goal:** Add bundled, user, and workspace Skill sources with deterministic precedence, per-source safety boundaries, progressive disclosure, and Session-lifetime catalog snapshots.

**Architecture:** Keep `runtime/skills.py` as the small Skill runtime. Represent each root explicitly as a source and retain that source on every manifest so discovery and delayed loading share the same boundary. Merge independently discovered source catalogs in fixed `user > workspace > bundled` order. Discover once when an `AgentSession` is created and pass that immutable catalog through run and subagent context construction.

**Tech stack:** Python dataclasses/enums, pathlib, setuptools package data, pytest.

---

### Task 1: Specify multi-source discovery and loading

**Files:**
- Modify: `tests/test_skills.py`
- Modify: `tests/test_skill_runtime.py`

1. Replace the legacy `<workspace>/skills` fixtures with `<workspace>/.tinyharness/skills`.
2. Add failing tests for bundled, user, and workspace discovery and a unified catalog.
3. Add failing tests for fixed cross-source precedence and same-source duplicate isolation.
4. Parameterize boundary/symlink tests across all three sources.
5. Preserve tests proving metadata-only discovery, delayed body loading, malformed isolation, exact lookup, bounded loading, and catalog budgets.
6. Add source-neutral catalog and tool wording assertions.

### Task 2: Implement explicit Skill sources

**Files:**
- Modify: `tiny_harness/runtime/skills.py`
- Modify: `tiny_harness/tools/skill.py`

1. Add a minimal source/origin data structure carrying the source root and allowed boundary.
2. Locate bundled Skills relative to the installed `tiny_harness` package, user Skills below `~/.tinyharness/skills`, and workspace Skills below `<workspace>/.tinyharness/skills`.
3. Discover each source deterministically and isolate malformed entries and same-source duplicates.
4. Merge valid manifests with fixed `user > workspace > bundled` precedence.
5. Revalidate a manifest against its retained source root and boundary during `load_skill`.
6. Keep full Skill bodies out of discovery and catalog formatting.
7. Make model-facing wording source-neutral without granting any Skill source extra authority.
8. Run the focused Skill tests.

### Task 3: Freeze Skill catalogs per Session

**Files:**
- Modify: `tests/test_session.py`
- Modify: `tests/test_subagent.py`
- Modify: `tiny_harness/agent/session.py`
- Modify: `tiny_harness/agent/context.py`
- Modify: `tiny_harness/agent/loop.py`
- Modify: `tiny_harness/agent/subagent.py`

1. Replace the old per-submit refresh expectation with failing tests for one Session snapshot and next-Session refresh.
2. Update runtime integration fixtures to the formal workspace root.
3. Discover the catalog in `AgentSession.__init__` and pass it to each run context.
4. Add a narrow optional catalog seam to one-shot and child runs so a subagent in the same run uses the same snapshot.
5. Keep Tool discovery unchanged except for receiving the already-built Skill catalog through context composition.
6. Run Session, context, Tool discovery, agent-loop, and subagent tests.

### Task 4: Add and package the bundled review Skill

**Files:**
- Add: `tiny_harness/skills/review/SKILL.md`
- Modify: `pyproject.toml`
- Modify or add: packaging-focused test as appropriate

1. Copy the existing review Skill into the formal bundled directory without cleaning examples.
2. Configure setuptools package data for `skills/*/SKILL.md`.
3. Verify installed-package lookup does not depend on cwd or repository-root discovery.
4. Build a wheel and inspect it for `tiny_harness/skills/review/SKILL.md`.

### Task 5: Verify and review

**Files:** all changed files

1. Run the focused Skill and directly affected runtime tests.
2. Run the complete test suite if practical.
3. Build and inspect the wheel artifact.
4. Request a code review subagent as required by the Superpowers review workflow; address verified Phase 2 findings only.
5. Re-run fresh verification after any review fixes.
6. Capture `git diff --stat`, `git status --short`, final paths, and call-chain evidence for the Phase 2 report.
