"""Opt-in payload tracing using existing event data dictionaries."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolTraceConfig:
    """Debug text may be sensitive. Preview limits count Unicode characters."""

    enabled: bool = False
    result_preview_chars: int = 200

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if type(self.result_preview_chars) is not int or self.result_preview_chars < 0:
            raise ValueError("result_preview_chars must be a nonnegative integer")

    def result_metadata(self, content: str) -> dict[str, str]:
        if not self.enabled or not self.result_preview_chars:
            return {}
        return {"content_preview": content[:self.result_preview_chars]}
