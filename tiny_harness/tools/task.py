"""Synchronous subagent delegation tool."""

from collections.abc import Callable

SubagentRunner = Callable[[str, str], str]


def task(
    runner: SubagentRunner,
    tool_call_id: str,
    prompt: str,
) -> str:
    """Run one delegated prompt and return only the child final text."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    return runner(prompt.strip(), tool_call_id)
