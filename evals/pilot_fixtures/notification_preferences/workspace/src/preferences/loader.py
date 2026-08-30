"""Load preferences from stored payloads."""

from collections.abc import Mapping
from typing import Any

from preferences.model import NotificationPreferences


def load_preferences(payload: Mapping[str, Any]) -> NotificationPreferences:
    return NotificationPreferences(
        email_enabled=payload.get("email_enabled", True),
    )
