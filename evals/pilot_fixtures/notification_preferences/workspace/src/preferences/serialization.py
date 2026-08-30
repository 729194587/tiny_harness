"""Serialize notification preferences."""

from preferences.model import NotificationPreferences


def preferences_to_dict(
    preferences: NotificationPreferences,
) -> dict[str, bool]:
    return {"email_enabled": preferences.email_enabled}
