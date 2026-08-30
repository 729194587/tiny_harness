"""Runtime settings loader."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from weather.defaults import DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class Settings:
    timeout_seconds: int


def load_settings(environment: Mapping[str, str] | None = None) -> Settings:
    values = os.environ if environment is None else environment
    raw_timeout = values.get("WEATHER_TIMEOUT")
    timeout = (
        DEFAULT_TIMEOUT_SECONDS
        if raw_timeout is None
        else int(raw_timeout)
    )
    return Settings(timeout_seconds=timeout)
