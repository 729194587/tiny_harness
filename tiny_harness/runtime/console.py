"""Small, synchronous Chinese progress renderer; never render raw payloads."""

import json
import os
import sys
from collections.abc import Mapping
from typing import Any, TextIO

from tiny_harness.runtime.events import EventLogError, EventType


def short(value: Any, limit: int = 120) -> str:
    """Bound scalar text and escape terminal controls and line separators."""
    if not isinstance(value, (str, int, float, bool)):
        return ""
    text = str(value)
    text = "".join(c if c.isprintable() else repr(c)[1:-1] for c in text[:limit])
    return text[:limit] + ("…" if len(text) > limit or len(str(value)) > limit else "")


def trace_summary(data: Mapping[str, Any]) -> str:
    """Only known, short trace fields may reach the terminal."""
    parts = []
    for key in ("path", "query", "pattern", "offset", "limit"):
        if key in data and (value := short(data[key])):
            parts.append(value if key == "path" else f"{key}={json.dumps(value, ensure_ascii=False)}")
    return " ".join(parts)[:240]


_WARM_ORANGE = "\x1b[38;5;208m"
_BLUE = "\x1b[38;5;75m"
_RESET = "\x1b[0m"


def _use_color(stream: TextIO) -> bool:
    if "NO_COLOR" in os.environ:
        return False
    try:
        return stream.isatty()
    except (AttributeError, OSError, ValueError):
        return False


class _LegacyConsoleEventLogger:
    """Progress goes to stderr so stdout remains the final answer channel."""

    def __init__(self, *, quiet: bool = False, verbose: bool = False,
                 stream: TextIO | None = None) -> None:
        self.quiet = quiet
        self.verbose = verbose
        self.stream = stream
        self.run_failure_reported = False
        self._failed_subagents: set[str] = set()
        self._calls: dict[tuple[Any, ...], dict[str, Any]] = {}

    def emit(self, event_type: EventType, data: Mapping[str, Any] | None = None) -> None:
        stream = self.stream if self.stream is not None else sys.stderr
        color = _use_color(stream)

        def identifier(value: Any) -> str:
            text = short(value)
            return f"{_BLUE}{text}{_RESET}" if color and text else text

        def status(label: str, name: str | None = None) -> str:
            if not color:
                return f"[{label}：{name}]" if name is not None else f"[{label}]"
            if name is not None:
                return f"{_WARM_ORANGE}[{label}：{_BLUE}{name}{_WARM_ORANGE}]{_RESET}"
            return f"{_WARM_ORANGE}[{label}]{_RESET}"

        d = dict(data or {})
        key = (d.get("agent_scope"), d.get("parent_tool_call_id"), d.get("tool_call_id"))
        if event_type == EventType.TOOL_CALLED:
            self._calls[key] = {k: short(d[k]) for k in
                                ("name", "path", "query", "pattern", "offset", "limit") if k in d}
            return
        metadata = self._calls.get(key, {})
        tool = short(d.get("tool_name", ""))
        skill = short(metadata.get("name", ""))
        line = ""
        error = False
        if event_type == EventType.TOOL_STARTED:
            if tool == "load_skill":
                line = status("正在装载 Skill", skill)
            elif tool != "task":
                line = status("正在调用工具", tool)
                if summary := trace_summary(metadata):
                    line += " " + summary
        elif event_type == EventType.TOOL_RESULT:
            self._calls.pop(key, None)
            child_failure_reported = tool == "task" and d.get("tool_call_id") in self._failed_subagents
            self._failed_subagents.discard(d.get("tool_call_id"))
            if d.get("outcome") == "error" and not child_failure_reported:
                line = status("工具调用失败", tool) + " " + identifier(d.get("error_type", "Error"))
                error = True
            elif d.get("outcome") == "returned":
                if tool == "load_skill":
                    line = status("Skill 装载完成", skill)
                elif tool != "task":
                    line = status("工具调用完成", tool)
            if line and self.verbose and "duration_ms" in d:
                line += f" 耗时 {short(d['duration_ms'])} ms"
        elif event_type in (EventType.TOOL_DENIED, EventType.TOOL_HOOK_BLOCKED, EventType.TOOL_HOOK_FAILED):
            label = {EventType.TOOL_DENIED: "工具调用被权限策略拒绝",
                     EventType.TOOL_HOOK_BLOCKED: "工具调用被 Hook 阻止",
                     EventType.TOOL_HOOK_FAILED: "工具 Hook 执行失败"}[event_type]
            line = status(label, tool)
            error = True
            if event_type == EventType.TOOL_HOOK_FAILED:
                self._calls.pop(key, None)
                line += " " + identifier(d.get("error_type", "Error"))
        elif event_type in (EventType.SUBAGENT_STARTED, EventType.SUBAGENT_FINISHED, EventType.SUBAGENT_FAILED):
            line = {EventType.SUBAGENT_STARTED: status("子 Agent 已启动"),
                    EventType.SUBAGENT_FINISHED: status("子 Agent 已完成"),
                    EventType.SUBAGENT_FAILED: status("子 Agent 执行失败")}[event_type]
            error = event_type == EventType.SUBAGENT_FAILED
            if error:
                if d.get("parent_tool_call_id") is not None:
                    self._failed_subagents.add(d["parent_tool_call_id"])
                if "error_type" in d:
                    line += " " + identifier(d["error_type"])
        elif event_type == EventType.CONTEXT_COMPACTED:
            line = status(f"上下文已压缩：{short(d.get('before_tokens', '?'))} → {short(d.get('after_tokens', '?'))} tokens")
        elif event_type == EventType.TODO_UPDATED:
            line = status(f"任务进度：已完成 {short(d.get('completed', 0))}/{short(d.get('total', 0))}，进行中 {short(d.get('in_progress', 0))}")
        elif event_type == EventType.MODEL_RETRY_SCHEDULED:
            line = status("模型调用将重试")
            if self.verbose:
                line += f" 尝试 {short(d.get('attempt', '?'))}，等待 {short(d.get('delay_ms', 0))} ms，原因 {short(d.get('error_kind', ''))}"
        elif event_type in (EventType.RUN_FAILED, EventType.MODEL_RETRY_EXHAUSTED,
                            EventType.MEMORY_EXTRACTION_FAILED, EventType.MEMORY_CONSOLIDATION_FAILED):
            label = {EventType.RUN_FAILED: "运行失败", EventType.MODEL_RETRY_EXHAUSTED: "模型重试次数已用尽",
                     EventType.MEMORY_EXTRACTION_FAILED: "记忆提取失败", EventType.MEMORY_CONSOLIDATION_FAILED: "记忆整合失败"}[event_type]
            line = (status(label) + " " + identifier(d.get("error_type", d.get("error_kind", "")))).rstrip()
            error = True
        elif self.verbose:
            labels = {EventType.CONTEXT_PREPARED: "上下文已准备", EventType.MODEL_REQUESTED: "正在请求模型",
                      EventType.MODEL_RESPONDED: "模型已响应", EventType.MODEL_REQUEST_FAILED: "模型请求失败",
                      EventType.CONTEXT_SUMMARY_REQUESTED: "正在生成上下文摘要",
                      EventType.CONTEXT_SUMMARY_RESPONDED: "上下文摘要已生成"}
            if event_type in labels:
                details = " ".join(f"{label}={identifier(d[k]) if k == 'error_type' else short(d[k])}" for k, label in
                                   (("turn", "轮次"), ("context_tokens", "上下文 tokens"), ("attempt", "尝试"),
                                    ("purpose", "用途"), ("error_type", "异常")) if k in d)
                line = (status(labels[event_type]) + " " + details).rstrip()
        if event_type == EventType.RUN_STARTED and d.get("agent_scope") != "subagent":
            self.run_failure_reported = False
            self._failed_subagents.clear()
        if event_type == EventType.RUN_FAILED and d.get("agent_scope") == "subagent":
            # The executor reports this failure at the delegated-task boundary.
            line = ""
        if event_type in (EventType.RUN_FINISHED, EventType.RUN_FAILED):
            self._calls = {k: v for k, v in self._calls.items() if k[:2] != key[:2]}
        if line and (not self.quiet or error):
            lifecycle = event_type in (EventType.SUBAGENT_STARTED, EventType.SUBAGENT_FINISHED, EventType.SUBAGENT_FAILED)
            indent = "  " if d.get("agent_scope") == "subagent" and not lifecycle else ""
            try:
                print(indent + line, file=stream, flush=True)
            except (OSError, ValueError) as error:
                raise EventLogError("Failed to write console progress") from error
            if event_type == EventType.RUN_FAILED:
                self.run_failure_reported = True


# Console Renderer v2 replaces the legacy per-tool renderer while keeping the
# public import stable for library and CLI callers.
from tiny_harness.runtime.console_v2 import ConsoleEventLogger
