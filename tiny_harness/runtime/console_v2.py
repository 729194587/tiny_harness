"""High-signal synchronous console rendering for TinyHarness events."""

from __future__ import annotations

import os
import sys
import unicodedata
from collections import Counter
from contextlib import contextmanager
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any, TextIO

from tiny_harness.runtime.events import EventLogError, EventType

_DIM_GRAY = "\x1b[2;90m"
_DIM_CYAN = "\x1b[2;36m"
_YELLOW = "\x1b[33m"
_RED = "\x1b[31m"
_RESET = "\x1b[0m"


def _short(value: Any, limit: int = 120) -> str:
    if not isinstance(value, (str, int, float, bool)):
        return ""
    text = str(value)
    text = "".join(c if c.isprintable() else repr(c)[1:-1] for c in text[:limit])
    return text[:limit] + ("…" if len(text) > limit or len(str(value)) > limit else "")


def _use_color(stream: TextIO) -> bool:
    if "NO_COLOR" in os.environ:
        return False
    try:
        return stream.isatty()
    except (AttributeError, OSError, ValueError):
        return False


def _human_number(value: int | float | None) -> str:
    if value is None:
        return "?"
    number = float(value)
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.1f}".rstrip("0").rstrip(".") + "m"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.1f}".rstrip("0").rstrip(".") + "k"
    return str(round(number))


def _human_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{round(seconds * 1000)}ms"
    if seconds < 60:
        return f"{seconds:.1f}".rstrip("0").rstrip(".") + "s"
    minutes, remainder = divmod(round(seconds), 60)
    return f"{minutes}m {remainder}s"


class ConsoleRenderer:
    """Aggregate the complete event stream into high-value terminal updates."""

    _TOOL_EVENTS = frozenset({
        EventType.TOOL_CALLED, EventType.TOOL_STARTED, EventType.TOOL_RESULT,
        EventType.TOOL_DENIED, EventType.TOOL_HOOK_BLOCKED, EventType.TOOL_HOOK_FAILED,
    })

    def __init__(self, *, quiet: bool = False, verbose: bool = False,
                 stream: TextIO | None = None,
                 clock: Callable[[], float] = perf_counter) -> None:
        self.quiet = quiet
        self.verbose = verbose
        self.stream = stream
        self._clock = clock
        self.run_failure_reported = False
        self._failed_subagents: set[str] = set()
        self._calls: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._tool_burst: Counter[str] = Counter()
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._turns = 0
        self._tool_calls = 0
        self._compactions = 0
        self._input_tokens = 0
        self._estimated_input_tokens = 0
        self._output_tokens = 0
        self._cache_hit_tokens = 0
        self._cache_miss_tokens = 0
        self._has_input_tokens = False
        self._has_output_tokens = False
        self._context_warning_shown = False
        self._run_finished = False
        self._live_text = ""
        self._live_style = ""
        self._live_visible = False
        self._suspended = False
        self._persistent = False

    def _is_live(self) -> bool:
        stream = self.stream if self.stream is not None else sys.stderr
        try:
            return stream.isatty()
        except (AttributeError, OSError, ValueError):
            return False

    def clear_live(self) -> None:
        """Erase the current status before any persistent output or stdin prompt."""
        if self._live_visible:
            stream = self.stream if self.stream is not None else sys.stderr
            try:
                stream.write("\r\x1b[2K")
                stream.flush()
            except (OSError, ValueError) as exc:
                raise EventLogError("Failed to clear console progress") from exc
            self._live_visible = False

    @contextmanager
    def suspend_live(self):
        self.clear_live()
        self._suspended = True
        try:
            yield
        finally:
            self._suspended = False
            if self._live_text:
                self._write(self._live_text, style=self._live_style, error=True)

    def _write(self, text: str, *, style: str = "", error: bool = False) -> None:
        if self.quiet and not error:
            return
        stream = self.stream if self.stream is not None else sys.stderr
        if self._is_live() and not self._persistent:
            self._live_text, self._live_style = text, style
            if self._suspended:
                return
            # Stay on one physical row, including on narrow/CJK terminals. Leave
            # the last column unused to avoid terminal autowrap and scrollback.
            try:
                columns = os.get_terminal_size(stream.fileno()).columns
            except (AttributeError, OSError, ValueError):
                columns = 80
            remaining = max(0, columns - 1)
            clipped = ""
            for char in " · ".join(line for line in text.splitlines() if line):
                width = 0 if unicodedata.combining(char) else (2 if unicodedata.east_asian_width(char) in "WF" else 1)
                if width > remaining:
                    break
                clipped += char
                remaining -= width
            rendered = f"{style}{clipped}{_RESET}" if style and _use_color(stream) else clipped
            try:
                stream.write("\r\x1b[2K" + rendered)
                stream.flush()
            except (OSError, ValueError) as exc:
                raise EventLogError("Failed to write console progress") from exc
            self._live_visible = True
            return
        rendered = f"{style}{text}{_RESET}" if style and _use_color(stream) else text
        try:
            print(rendered, file=stream, flush=True)
        except (OSError, ValueError) as exc:
            raise EventLogError("Failed to write console progress") from exc

    def _flush_tool_burst(self) -> None:
        if not self._tool_burst:
            return
        lines = ["🔎 Exploring", ""]
        lines.extend(f"{name} ×{count}" for name, count in self._tool_burst.items())
        self._tool_burst.clear()
        self._write("\n".join(lines), style=_DIM_CYAN)

    def _reset_run(self) -> None:
        self.run_failure_reported = False
        self._failed_subagents.clear()
        self._calls.clear()
        self._tool_burst.clear()
        self._started_at = self._clock()
        self._finished_at = None
        self._turns = self._tool_calls = self._compactions = 0
        self._input_tokens = self._estimated_input_tokens = self._output_tokens = 0
        self._cache_hit_tokens = self._cache_miss_tokens = 0
        self._has_input_tokens = self._has_output_tokens = False
        self._context_warning_shown = self._run_finished = False

    @staticmethod
    def _key(data: Mapping[str, Any]) -> tuple[Any, ...]:
        return (data.get("agent_scope"), data.get("parent_tool_call_id"), data.get("tool_call_id"))

    def _record_tokens(self, event_type: EventType, data: Mapping[str, Any]) -> None:
        # Summary/compaction events also carry heuristic input_tokens; those
        # are neither another API request nor actual provider usage.
        if event_type not in (EventType.MODEL_REQUESTED, EventType.MODEL_RESPONDED):
            return
        if event_type == EventType.MODEL_RESPONDED:
            hit = data.get("prompt_cache_hit_tokens")
            miss = data.get("prompt_cache_miss_tokens")
            if type(hit) is int and type(miss) is int:
                self._cache_hit_tokens += hit
                self._cache_miss_tokens += miss
        for key in ("input_tokens", "prompt_tokens"):
            if isinstance(data.get(key), (int, float)):
                self._input_tokens += int(data[key])
                self._has_input_tokens = True
                break
        else:
            if event_type == EventType.MODEL_REQUESTED and isinstance(data.get("context_tokens"), (int, float)):
                self._estimated_input_tokens += int(data["context_tokens"])
        for key in ("output_tokens", "completion_tokens"):
            if isinstance(data.get(key), (int, float)):
                self._output_tokens += int(data[key])
                self._has_output_tokens = True
                break

    def _emit_verbose(self, event_type: EventType, data: Mapping[str, Any]) -> None:
        safe = []
        for key in ("turn", "tool_name", "outcome", "error_type", "duration_ms",
                    "context_tokens", "before_tokens", "after_tokens",
                    "prompt_tokens", "completion_tokens", "total_tokens",
                    "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "cache_hit_rate"):
            if key in data and (value := _short(data[key])):
                safe.append(f"{key}={value}")
        error = event_type in {EventType.RUN_FAILED, EventType.TOOL_DENIED,
                               EventType.TOOL_HOOK_BLOCKED, EventType.TOOL_HOOK_FAILED}
        self._write(event_type.name + (" " + " ".join(safe) if safe else ""),
                    style=_RED if error else _DIM_GRAY, error=error)

    def emit(self, event_type: EventType, data: Mapping[str, Any] | None = None) -> None:
        d = dict(data or {})
        root_end = event_type in (EventType.RUN_FINISHED, EventType.RUN_FAILED) and d.get("agent_scope") != "subagent"
        if event_type == EventType.RUN_STARTED and d.get("agent_scope") != "subagent":
            self.clear_live()
            self._live_text = ""
        self._persistent = root_end and event_type == EventType.RUN_FAILED
        if self._persistent:
            self.clear_live()
            self._tool_burst.clear()
        try:
            self._emit(event_type, d)
            if self._is_live() and not self.verbose:
                if event_type == EventType.MODEL_REQUESTED:
                    self._write(f"正在请求模型 · 轮次 {_short(d.get('turn', '?'))} · 工具 {self._tool_calls} · 上下文 {_human_number(d.get('context_tokens'))}", style=_DIM_GRAY)
                elif event_type == EventType.TOOL_STARTED:
                    self._write(f"正在调用工具 · {_short(d.get('tool_name', 'tool'))} · 工具 {self._tool_calls}", style=_DIM_CYAN)
        finally:
            self._persistent = False
            if root_end:
                self.clear_live()
                self._live_text = ""

    def _emit(self, event_type: EventType, data: Mapping[str, Any] | None = None) -> None:
        d = dict(data or {})
        key = self._key(d)
        if event_type == EventType.RUN_STARTED and d.get("agent_scope") != "subagent":
            self._reset_run()
        self._record_tokens(event_type, d)
        if event_type == EventType.TOOL_CALLED:
            self._tool_calls += 1
            self._calls[key] = {k: _short(d[k]) for k in ("tool_name", "name") if k in d}
        if self.verbose:
            self._emit_verbose(event_type, d)
            self._update(event_type, d, key, aggregate=False)
            return
        if event_type not in self._TOOL_EVENTS:
            self._flush_tool_burst()
        if event_type == EventType.MODEL_RESPONDED:
            usage = " / ".join(
                f"实际{label} {d[key]} tokens"
                for key, label in (("prompt_tokens", "输入"), ("completion_tokens", "输出"))
                if type(d.get(key)) is int
            )
            cache = " / ".join(
                f"cache {label} {d[key]}"
                for key, label in (("prompt_cache_hit_tokens", "hit"), ("prompt_cache_miss_tokens", "miss"))
                if type(d.get(key)) is int
            )
            if isinstance(d.get("cache_hit_rate"), (int, float)):
                cache += f" / 命中率 {d['cache_hit_rate']:.1%}"
            usage = " / ".join(part for part in (usage, cache) if part)
            if usage:
                self._write(usage, style=_DIM_GRAY)
        self._update(event_type, d, key, aggregate=True)

    def _update(self, event_type: EventType, d: Mapping[str, Any], key: tuple[Any, ...],
                *, aggregate: bool) -> None:
        metadata = self._calls.get(key, {})
        tool = _short(d.get("tool_name", "")) or _short(metadata.get("tool_name", "")) or "tool"
        if event_type == EventType.TOOL_RESULT:
            self._calls.pop(key, None)
            child_failure = tool == "task" and d.get("tool_call_id") in self._failed_subagents
            self._failed_subagents.discard(d.get("tool_call_id"))
            if aggregate and d.get("outcome") == "returned":
                self._tool_burst[tool] += 1
            elif aggregate and d.get("outcome") == "error" and not child_failure:
                self._flush_tool_burst()
                self._write(f"✗ {tool} failed: {_short(d.get('error_type', 'Error'))}", style=_RED, error=True)
        elif event_type in (EventType.TOOL_DENIED, EventType.TOOL_HOOK_BLOCKED, EventType.TOOL_HOOK_FAILED):
            if aggregate:
                self._flush_tool_burst()
                reason = {EventType.TOOL_DENIED: "permission denied",
                          EventType.TOOL_HOOK_BLOCKED: "blocked by hook",
                          EventType.TOOL_HOOK_FAILED: f"hook failed: {_short(d.get('error_type', 'Error'))}"}[event_type]
                self._write(f"✗ {tool} {reason}", style=_RED, error=True)
            if event_type == EventType.TOOL_HOOK_FAILED:
                self._calls.pop(key, None)
        elif event_type == EventType.SUBAGENT_FAILED:
            if d.get("parent_tool_call_id") is not None:
                self._failed_subagents.add(str(d["parent_tool_call_id"]))
            if aggregate:
                self._write(f"✗ Subagent failed: {_short(d.get('error_type', 'Error'))}", style=_RED, error=True)
        elif event_type == EventType.SUBAGENT_STARTED and aggregate:
            self._write("Subagent started", style=_DIM_GRAY)
        elif event_type == EventType.SUBAGENT_FINISHED and aggregate:
            self._write("Subagent finished", style=_DIM_GRAY)
        elif event_type == EventType.CONTEXT_PREPARED:
            pressure = d.get("pressure")
            if aggregate and isinstance(pressure, (int, float)) and pressure >= .8 and not self._context_warning_shown:
                self._context_warning_shown = True
                self._write(f"⚠ Context nearing limit\n\n{_human_number(d.get('context_tokens'))} / {_human_number(d.get('hard_limit'))} tokens", style=_YELLOW)
        elif event_type == EventType.CONTEXT_COMPACTED:
            self._compactions += 1
            if aggregate:
                self._write(f"⚡ Context compacted\n\n{_human_number(d.get('before_tokens'))} → {_human_number(d.get('after_tokens'))} tokens", style=_DIM_GRAY)
        elif event_type == EventType.MODEL_RETRY_SCHEDULED and aggregate:
            self._write("Model request will retry", style=_YELLOW)
        elif event_type in (EventType.RUN_FAILED, EventType.MODEL_RETRY_EXHAUSTED,
                            EventType.MEMORY_EXTRACTION_FAILED, EventType.MEMORY_CONSOLIDATION_FAILED):
            if event_type == EventType.RUN_FAILED and d.get("agent_scope") == "subagent":
                return
            if aggregate:
                label = {EventType.RUN_FAILED: "Run failed",
                         EventType.MODEL_RETRY_EXHAUSTED: "Model retries exhausted",
                         EventType.MEMORY_EXTRACTION_FAILED: "Memory extraction failed",
                         EventType.MEMORY_CONSOLIDATION_FAILED: "Memory consolidation failed"}[event_type]
                detail = _short(d.get("error_type", d.get("error_kind", "")))
                self._write((f"✗ {label}: {detail}").rstrip(": "), style=_RED, error=True)
            if event_type == EventType.RUN_FAILED:
                self.run_failure_reported = True
        if event_type in (EventType.RUN_FINISHED, EventType.RUN_FAILED):
            if d.get("agent_scope") != "subagent":
                self._finished_at = self._clock()
                self._turns = int(d.get("turns", d.get("turn", self._turns)) or 0)
                self._run_finished = event_type == EventType.RUN_FINISHED
            scope = key[:2]
            self._calls = {call_key: value for call_key, value in self._calls.items() if call_key[:2] != scope}

    def summary(self, *, model: str, stream: TextIO | None = None) -> str:
        """Return a faint summary for the CLI to append after the final answer."""
        if self.quiet or not self._run_finished:
            return ""
        if self._started_at is not None and self._finished_at is not None:
            duration = self._finished_at - self._started_at
        else:
            now = self._clock()
            duration = (self._finished_at if self._finished_at is not None else now) - (self._started_at if self._started_at is not None else now)
        effective_input = self._input_tokens if self._has_input_tokens else self._estimated_input_tokens
        input_tokens = _human_number(effective_input if self._has_input_tokens or self._estimated_input_tokens else None)
        if not self._has_input_tokens and self._estimated_input_tokens:
            input_tokens += " (预估)"
        output_tokens = _human_number(self._output_tokens if self._has_output_tokens else None)
        cache_total = self._cache_hit_tokens + self._cache_miss_tokens
        cache = f"{self._cache_hit_tokens / cache_total:.1%} hit" if cache_total > 0 else "n/a"
        text = "\n".join(("────────────────", "", "Run summary", f"model: {_short(model)}",
                          f"turns: {self._turns}", f"tool calls: {self._tool_calls}",
                          f"tokens: {input_tokens} input / {output_tokens} output",
                          f"cache: {cache}",
                          f"compactions: {self._compactions}",
                          f"duration: {_human_duration(max(0.0, duration))}"))
        target = stream if stream is not None else sys.stdout
        return f"{_DIM_GRAY}{text}{_RESET}" if _use_color(target) else text


# Preserve the existing event-logger construction API while making the display
# responsibility explicit in the implementation name.
ConsoleEventLogger = ConsoleRenderer
