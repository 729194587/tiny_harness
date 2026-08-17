"""Minimal permission decisions for tool execution."""

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


class DefaultPermissionPolicy:
    """Allow file/planning tools and ask before every shell command."""

    _ALLOWED_TOOLS = frozenset(
        {
            "read_file",
            "write_file",
            "edit_file",
            "list_files",
            "todo_write",
        }
    )

    def decide(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> PermissionDecision:
        del arguments
        if tool_name in self._ALLOWED_TOOLS:
            return PermissionDecision.ALLOW
        if tool_name == "bash":
            return PermissionDecision.ASK
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
