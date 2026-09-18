"""Small, synchronous Chinese progress renderer; never render raw payloads."""

import json
from collections.abc import Mapping
from typing import Any


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


# Keep the public Console import stable for library and CLI callers.
from tiny_harness.runtime.console_v2 import ConsoleEventLogger
