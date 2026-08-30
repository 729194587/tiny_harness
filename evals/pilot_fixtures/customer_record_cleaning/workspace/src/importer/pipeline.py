"""Customer import pipeline."""

from collections.abc import Iterable
from typing import Any


def import_customers(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [dict(record) for record in records]
