"""Minimal permission decisions for tool execution."""

import os
import re
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, Protocol


class PermissionDecision(str, Enum):
    """A policy decision made before a tool handler runs."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionPolicy(Protocol):
    """Decide how one parsed tool call should be handled."""

    def decide(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> PermissionDecision:
        """Return ALLOW, DENY, or ASK for one tool call."""
        ...


PermissionPrompt = Callable[[str, Mapping[str, Any]], bool]


_READ_ONLY_SHELL_COMMANDS = frozenset(
    {
        "cat",
        "cd",
        "comp",
        "diff",
        "dir",
        "echo",
        "fc",
        "file",
        "find",
        "findstr",
        "grep",
        "head",
        "ls",
        "pwd",
        "realpath",
        "rg",
        "sort",
        "stat",
        "tail",
        "tree",
        "type",
        "uniq",
        "wc",
        "where",
        "which",
    }
)
_READ_ONLY_GIT_SUBCOMMANDS = frozenset(
    {
        "describe",
        "diff",
        "grep",
        "log",
        "ls-files",
        "rev-parse",
        "show",
        "status",
    }
)
_WINDOWS_SWITCH_COMMANDS = frozenset(
    {"comp", "dir", "fc", "find", "findstr", "tree", "where"}
)
_MUTATING_SHELL_ARGUMENTS = frozenset(
    {
        "-delete",
        "-exec",
        "-execdir",
        "-fprint",
        "-fprintf",
        "-fls",
        "-o",
        "-ok",
        "-okdir",
        "--output",
        "--pre",
    }
)
_MUTATING_SHELL_ARGUMENT_PREFIXES = (
    "--output=",
    "--pre=",
    "--pre-glob=",
)
_SHELL_SEPARATOR = re.compile(r"\|\||&&|[|;&\r\n]")
_DYNAMIC_OR_EXTERNAL_PATH = re.compile(
    r"(?:\$|`|\^|"
    r"%[A-Za-z_][A-Za-z0-9_]*(?::[^%\r\n]*)?%|"
    r"![A-Za-z_][A-Za-z0-9_]*!|"
    r"(?:^|[\s\"'])[A-Za-z]:|"
    r"(?:^|[\s\"'])\\+|"
    r"(?:^|[\\/\s\"'])\.\.(?:[\\/\s\"']|$)|"
    r"(?:^|[\s\"'])~(?:[\\/\s\"']|$))"
)


def is_read_only_shell_command(command: object) -> bool:
    """Recognize a small, conservative set of local inspection commands."""

    if not isinstance(command, str) or not command.strip():
        return False
    if any(character in command for character in ("<", ">", "(", ")")):
        return False
    if _DYNAMIC_OR_EXTERNAL_PATH.search(command):
        return False

    segments = _SHELL_SEPARATOR.split(command)
    if not segments or any(not segment.strip() for segment in segments):
        return False
    for segment in segments:
        words = segment.strip().lstrip("@").split()
        if not words:
            return False
        executable = words[0].strip("\"'").casefold()
        if executable.endswith(".exe"):
            executable = executable[:-4]
        for argument in words[1:]:
            argument = argument.strip("\"'")
            normalized_argument = argument.casefold()
            if normalized_argument in _MUTATING_SHELL_ARGUMENTS or any(
                normalized_argument.startswith(prefix)
                for prefix in _MUTATING_SHELL_ARGUMENT_PREFIXES
            ):
                return False
            if not argument.startswith("/"):
                continue
            if (
                os.name == "nt"
                and executable in _WINDOWS_SWITCH_COMMANDS
                and re.fullmatch(r"/[A-Za-z][A-Za-z0-9:-]*", argument)
            ):
                continue
            return False
        if executable in _READ_ONLY_SHELL_COMMANDS:
            continue
        if executable == "git" and len(words) >= 2:
            subcommand = words[1].strip("\"'").casefold()
            if subcommand in _READ_ONLY_GIT_SUBCOMMANDS:
                continue
        return False
    return True


class DefaultPermissionPolicy:
    """Allow bounded tools and clear shell inspection; ask otherwise."""

    _ALLOWED_TOOLS = frozenset(
        {
            "read_file",
            "write_file",
            "edit_file",
            "list_files",
            "todo_write",
            "task",
            "compact",
        }
    )

    def decide(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> PermissionDecision:
        if tool_name in self._ALLOWED_TOOLS:
            return PermissionDecision.ALLOW
        if tool_name == "bash":
            return (
                PermissionDecision.ALLOW
                if is_read_only_shell_command(arguments.get("command"))
                else PermissionDecision.ASK
            )
        return PermissionDecision.DENY


DEFAULT_PERMISSION_POLICY = DefaultPermissionPolicy()


def resolve_permission(
    policy: PermissionPolicy,
    tool_name: str,
    arguments: Mapping[str, Any],
    prompt: PermissionPrompt | None = None,
) -> PermissionDecision:
    """Resolve ASK through a prompt and fail closed on ordinary errors."""

    try:
        decision = policy.decide(tool_name, arguments)
        if decision is PermissionDecision.ALLOW:
            return PermissionDecision.ALLOW
        if decision is PermissionDecision.DENY:
            return PermissionDecision.DENY
        if decision is PermissionDecision.ASK and prompt is not None:
            return (
                PermissionDecision.ALLOW
                if prompt(tool_name, arguments)
                else PermissionDecision.DENY
            )
    except Exception:
        pass
    return PermissionDecision.DENY
