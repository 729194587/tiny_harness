"""Small weather client configuration object."""

from collections.abc import Mapping

from weather.loader import load_settings


class WeatherClient:
    def __init__(
        self,
        timeout_seconds: int | None = None,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        settings = load_settings(environment)
        self.timeout_seconds = (
            settings.timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
