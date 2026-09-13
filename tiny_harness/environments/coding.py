"""Bounded repository facts and read-only Git tools for coding runs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tiny_harness.tools.definition import ToolDefinition

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext


MAX_CONTEXT_CHARS = 4000
MAX_GIT_OUTPUT_CHARS = 16000
MAX_ROOT_ENTRIES = 60
_SKIP = frozenset({
    ".git", ".tinyharness", "node_modules", ".venv", "venv", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build", "target",
})
_CONFIG_HINTS = {
    "pyproject.toml": "Python build/project configuration",
    "setup.py": "Python packaging configuration",
    "setup.cfg": "Python tooling configuration",
    "pytest.ini": "pytest configuration",
    "tox.ini": "tox configuration",
    "package.json": "JavaScript/TypeScript package configuration",
    "Cargo.toml": "Rust package configuration",
    "go.mod": "Go module configuration",
    "pom.xml": "Maven configuration",
    "build.gradle": "Gradle configuration",
    "Makefile": "Make build configuration",
    "CMakeLists.txt": "CMake configuration",
}
_GIT = "git --no-pager --no-optional-locks -c color.ui=false "
_STATUS = _GIT + "status --porcelain=v2 --untracked-files=normal --ignore-submodules=all"
_DIFF = _GIT + "diff --no-ext-diff --no-textconv --ignore-submodules=all HEAD --"


def _bounded(text: str, limit: int) -> str:
    suffix = "\n... truncated"
    return text if len(text) <= limit else text[:limit - len(suffix)] + suffix


class CodingEnvironmentAdapter:
    """Add repository facts and Git queries without replacing built-in tools.

    Callers explicitly authorize git_status/git_diff using their existing
    PermissionPolicy. This adapter grants no permissions and performs no setup.
    """

    def build_tools(self, context: AgentRunContext) -> tuple[ToolDefinition, ...]:
        workspace = context.workspace
        runner = context.shell_runner

        def definition(name: str, description: str, command: str, note: str):
            def execute(call, arguments):
                if arguments:
                    raise TypeError(f"{name} does not accept arguments")
                report = runner.run(workspace, command)
                return _bounded(note + "\n" + report, MAX_GIT_OUTPUT_CHARS)

            return ToolDefinition(
                name, description,
                {"type": "object", "properties": {}, "additionalProperties": False},
                execute,
            )

        return (
            definition(
                "git_status", "Read Git worktree status, including untracked paths.",
                _STATUS,
                "Git status snapshot (porcelain v2: XY=index/worktree, ?=untracked; "
                "untracked directories may be summarized; submodules omitted).",
            ),
            definition(
                "git_diff", "Read tracked-file changes relative to HEAD, including staged and unstaged changes.",
                _DIFF,
                "Tracked changes relative to HEAD; untracked contents and submodules omitted. "
                "This is a review snapshot, not an evaluation patch export.",
            ),
        )

    def initial_context(self, context: AgentRunContext) -> str:
        workspace = context.workspace.resolve()
        lines = [
            "Coding environment: initial repository snapshot; file contents have not been read.",
            "Search and narrow before broad reading; each exploration should answer a concrete unresolved question. "
            "Stop exploring when evidence is sufficient for the next action.",
            "Choose the tool that fits the question: glob locates files by filename/directory pattern; "
            "grep quickly locates exact symbols, error strings, or literal occurrences; "
            "search_code performs literal search with nearby context, reducing extra search-to-read round trips. "
            "Once a relevant location is known, prefer read_file with start_line/end_line for the necessary range. "
            "Use bash primarily for execution, tests, git, or queries repository tools cannot express. "
            "git_status/git_diff require policy authorization.",
            ("run_tests is configured; it runs the caller-provided fixed test suite."
             if context.test_runner is not None else
             "run_tests is not configured. Test commands must use the authorized shell environment; none are inferred."),
            "Root entries (JSON-quoted names; caches/dependencies and outside-workspace links omitted):",
        ]
        try:
            entries = sorted(workspace.iterdir(), key=lambda p: (p.name.casefold(), p.name))
            visible = []
            hints = []
            for entry in entries:
                if entry.name in _SKIP:
                    continue
                try:
                    resolved = entry.resolve()
                    if not resolved.is_relative_to(workspace):
                        continue
                    directory = resolved.is_dir()
                    if not directory and not resolved.is_file():
                        continue
                except (OSError, RuntimeError):
                    continue
                name = json.dumps(entry.name + ("/" if directory else ""), ensure_ascii=False)
                visible.append(name)
                if not directory:
                    hint = _CONFIG_HINTS.get(entry.name)
                    if hint:
                        hints.append(f"{name}: {hint} (filename evidence only)")
                    elif entry.name.casefold().startswith(("readme", "contributing")):
                        hints.append(f"{name}: repository documentation path")
            lines.extend(visible[:MAX_ROOT_ENTRIES])
            if len(visible) > MAX_ROOT_ENTRIES:
                lines.append("... root entries truncated")
            lines.append("Configuration/documentation clues (no test commands inferred):")
            lines.extend(hints or ["(none detected at repository root)"])
        except OSError:
            lines.append("Repository listing unavailable.")
        return _bounded("\n".join(lines), MAX_CONTEXT_CHARS)
