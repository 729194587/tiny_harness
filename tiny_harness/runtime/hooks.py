"""Minimal synchronous tool hooks for one TinyHarness run."""

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from tiny_harness.agent.messages import ToolResult


@dataclass(frozen=True)
class ToolHookContext:
    """Parsed tool-call metadata exposed to Pre/Post Tool hooks."""

    tool_call_id: str
    tool_name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class HookBlock:
    """An explicit PreToolUse decision to skip one tool call."""

    reason: str


class HookExecutionError(RuntimeError):
    """Raised when a registered hook violates its contract or fails."""

    def __init__(self, stage: str, hook_index: int, error: Exception) -> None:
        self.stage = stage
        self.hook_index = hook_index
        self.error_type = type(error).__name__
        super().__init__(
            f"{stage} tool hook {hook_index} failed: "
            f"{self.error_type}: {error}"
        )


PreToolHook = Callable[[ToolHookContext], HookBlock | None]
PostToolHook = Callable[[ToolHookContext, ToolResult], None]


@dataclass(frozen=True)
class StopHookContext:
    """模型提出结束时交给受信任 Harness Hook 的运行边界。"""

    messages: list[dict[str, Any]]
    candidate_answer: str
    turn: int
    has_next_turn: bool


@dataclass(frozen=True)
class StopDecision:
    """Stop Hook 对候选最终回答作出的类型化决定。"""

    action: Literal["allow", "block"]
    reason: str = ""


StopHook = Callable[[StopHookContext], StopDecision]


def run_stop_hook(
    hook: StopHook | None,
    messages: list[dict[str, Any]],
    candidate_answer: str,
    *,
    turn: int,
    has_next_turn: bool,
) -> StopDecision:
    """运行一个可选 Stop Hook，并拒绝模糊或非法返回值。"""

    if hook is None:
        return StopDecision("allow")
    decision = hook(
        StopHookContext(
            messages=messages,
            candidate_answer=candidate_answer,
            turn=turn,
            has_next_turn=has_next_turn,
        )
    )
    if not isinstance(decision, StopDecision):
        raise TypeError("Stop hook must return StopDecision")
    if decision.action not in {"allow", "block"}:
        raise TypeError("StopDecision action must be 'allow' or 'block'")
    if not isinstance(decision.reason, str):
        raise TypeError("StopDecision reason must be a string")
    return decision


class ToolHooks:
    """Ordered, run-scoped PreToolUse and PostToolUse callbacks."""

    def __init__(self) -> None:
        self._pre_tool_use: list[PreToolHook] = []
        self._post_tool_use: list[PostToolHook] = []

    def register_pre(self, callback: PreToolHook) -> None:
        if not callable(callback):
            raise TypeError("PreToolUse hook must be callable")
        self._pre_tool_use.append(callback)

    def register_post(self, callback: PostToolHook) -> None:
        if not callable(callback):
            raise TypeError("PostToolUse hook must be callable")
        self._post_tool_use.append(callback)

    def run_pre(
        self,
        context: ToolHookContext,
    ) -> tuple[int, HookBlock] | None:
        """Run PreToolUse hooks until one explicitly blocks the call."""

        callbacks = tuple(self._pre_tool_use)
        for hook_index, callback in enumerate(callbacks, start=1):
            try:
                result = callback(context)
                if result is not None and not isinstance(result, HookBlock):
                    raise TypeError(
                        "PreToolUse hook must return HookBlock or None"
                    )
                if isinstance(result, HookBlock) and not isinstance(
                    result.reason, str
                ):
                    raise TypeError("HookBlock reason must be a string")
            except Exception as error:
                raise HookExecutionError("pre", hook_index, error) from error
            if result is not None:
                return hook_index, result
        return None

    def run_post(
        self,
        context: ToolHookContext,
        result: ToolResult,
    ) -> None:
        """Run PostToolUse hooks in registration order as observers."""

        callbacks = tuple(self._post_tool_use)
        for hook_index, callback in enumerate(callbacks, start=1):
            try:
                hook_return = callback(context, copy.deepcopy(result))
                if hook_return is not None:
                    raise TypeError("PostToolUse hook must return None")
            except Exception as error:
                raise HookExecutionError("post", hook_index, error) from error
