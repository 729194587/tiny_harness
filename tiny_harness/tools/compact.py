"""Manual context-compaction tool."""

from tiny_harness.runtime.context import CompactionRequest


def compact(manager: CompactionRequest) -> str:
    """Request compaction after the current tool batch is fully closed."""

    return manager.request()
