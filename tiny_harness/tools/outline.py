"""Bounded Python source outlines without importing or executing source."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

from tiny_harness.tools.definition import ToolDefinition
from tiny_harness.tools.filesystem import _resolve_path

if TYPE_CHECKING:
    from tiny_harness.agent.context import AgentRunContext

MAX_OUTLINE_FILE_BYTES = 2 * 1024 * 1024
MAX_OUTLINE_SYMBOLS = 500
MAX_OUTLINE_OUTPUT_CHARS = 30_000


def code_outline(workspace: Path, path: str, max_symbols: int = 200) -> str:
    """List definitions in source order, indented by their enclosing definitions."""
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a nonempty string")
    if isinstance(max_symbols, bool) or not isinstance(max_symbols, int):
        raise TypeError("max_symbols must be an integer")
    if not 1 <= max_symbols <= MAX_OUTLINE_SYMBOLS:
        raise ValueError(f"max_symbols must be between 1 and {MAX_OUTLINE_SYMBOLS}")
    target = _resolve_path(workspace, path)
    if target.suffix != ".py":
        raise ValueError("code_outline only supports .py files")
    if not target.is_file():
        raise ValueError(f"Python file does not exist or is not a file: {path}")
    with target.open("rb") as stream:
        source = stream.read(MAX_OUTLINE_FILE_BYTES + 1)
    if len(source) > MAX_OUTLINE_FILE_BYTES:
        raise ValueError("Python file exceeds the 2 MiB outline limit")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"Python syntax error at line {error.lineno}: {error.msg}") from None
    rows: list[str] = []
    size = 0
    stack = [(tree, 0, False)]
    while stack:
        node, depth, in_class = stack.pop()
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if isinstance(node, ast.ClassDef):
                kind = "class"
            else:
                kind = "method" if in_class else "function"
                if isinstance(node, ast.AsyncFunctionDef):
                    kind = "async " + kind
            row = f"{'  ' * depth}{kind} {node.name} [{node.lineno}-{node.end_lineno}]"
            if len(rows) == max_symbols:
                rows.append(f"... truncated after {max_symbols} symbols")
                break
            if size + len(row) + 1 > MAX_OUTLINE_OUTPUT_CHARS - 100:
                rows.append("... truncated at output character limit")
                break
            rows.append(row)
            size += len(row) + 1
            depth += 1
            in_class = isinstance(node, ast.ClassDef)
        stack.extend((child, depth, in_class) for child in reversed(list(ast.iter_child_nodes(node))))
    return "\n".join(rows) or "(no definitions)"


def build_tools(context: AgentRunContext) -> tuple[ToolDefinition, ...]:
    return (ToolDefinition(
        name="code_outline",
        description=(
            "Outline a workspace .py file without executing it: classes, functions, async "
            "functions and methods, indented by enclosing definitions, with inclusive "
            "1-based start/end lines (starting at def/class, excluding decorators). "
            "Use read_file with those lines for details. Maximum 2 MiB input, 500 symbols "
            "and 30000 output characters."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_symbols": {"type": "integer", "minimum": 1,
                                "maximum": MAX_OUTLINE_SYMBOLS, "default": 200},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        execute=lambda call, arguments: code_outline(context.workspace, **arguments),
    ),)
