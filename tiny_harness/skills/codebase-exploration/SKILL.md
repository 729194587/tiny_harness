---
name: codebase-exploration
description: Explore an unfamiliar codebase efficiently when locating implementations, tracing behavior, understanding architecture, or identifying files relevant to a coding task, using focused search and reading instead of broad repository scans.
---

# Codebase Exploration

Use this workflow when you need to understand unfamiliar code, locate an implementation, trace behavior across modules, or determine which files are relevant to a task.

The goal is to gather enough evidence to act confidently without reading more of the repository than necessary.

## 1. Start from the question

Identify exactly what you need to understand.

Examples:

- Where is this behavior implemented?
- Which files participate in this feature?
- Where is this symbol defined and used?
- What code path leads to this error?
- Which tests cover this behavior?
- How are these components connected?

Do not begin by reading the entire repository.

## 2. Locate candidate files

Prefer repository-aware search tools before shell commands.

Use:

1. `glob` to locate files by name, extension, or directory pattern.
2. `grep` for quick location of exact symbols, error strings, or literal occurrences.
3. `search_code` for literal search when nearby context is needed; it can avoid an extra search-to-read round trip.
4. `list_files` when the immediate contents of a known directory are useful.

Choose the tool that best answers the current question.

Examples:

- `glob("**/*.py", path="tiny_harness")`
- `grep("MemoryRuntime", include="**/*.py")`
- `grep("create_run_context", include="**/*.py")`

Use `bash` primarily for execution, tests, git, or queries repository tools cannot express.

## 3. Narrow before reading

From the search results, choose the smallest set of files likely to answer the question.

Prefer files that contain:

- the primary definition;
- direct callers or consumers;
- relevant tests;
- configuration or composition code connecting the components.

Do not read every file that merely contains a common term.

## 4. Read the relevant path

Once a relevant location is known, prefer `read_file` with `start_line`/`end_line` to read only the necessary range.

While reading, identify:

- important types and functions;
- inputs and outputs;
- state that crosses module boundaries;
- direct dependencies;
- where important decisions are made.

Separate implementation details from public or subsystem boundaries.

## 5. Follow dependencies only when needed

If the current files do not fully explain the behavior, search for the next specific dependency.

Typical follow-ups include:

- callers of a function;
- implementations of an interface;
- construction sites for an object;
- tests exercising the behavior;
- event, hook, or permission paths that affect execution.

Continue with focused search or ranged reading only while each step answers a concrete unanswered question. If search context already provides enough evidence for the next action, stop exploring and act.

Avoid recursively exploring unrelated neighboring modules.

## 6. Use tests as behavioral evidence

When implementation intent is unclear, inspect directly relevant tests.

Tests can reveal:

- expected behavior;
- edge cases;
- compatibility requirements;
- intentionally unsupported behavior;
- subsystem boundaries.

Do not assume that implementation code alone completely defines the intended contract.

## 7. Build a concise mental model

Before making changes or explaining the architecture, summarize the relevant path in a small call chain or dependency map.

For example:

`AgentSession → create_run_context → discover_tools → ToolRegistry → agent_loop`

or:

`model tool call → tool batch → registry dispatch → permission → execute → ToolResult`

The model should contain only components relevant to the current task.

## 8. Stop when the evidence is sufficient

Exploration is complete when you can answer the original question or identify the files that must change with reasonable confidence.

Do not continue searching merely because more code exists.

If uncertainty remains, state the specific unresolved question and gather only the evidence needed to resolve it.

## When making changes

If exploration leads to an implementation task:

- modify only the files supported by the evidence gathered;
- avoid unrelated cleanup discovered during exploration;
- expand the search again only if the implementation exposes a previously unknown dependency.

Exploration should reduce uncertainty, not expand the scope of the task.
