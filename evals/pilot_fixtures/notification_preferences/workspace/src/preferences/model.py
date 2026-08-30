"""Preference model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NotificationPreferences:
    email_enabled: bool = True

    def __post_init__(self) -> None:
        if type(self.email_enabled) is not bool:
            raise ValueError("email_enabled must be a bool")
