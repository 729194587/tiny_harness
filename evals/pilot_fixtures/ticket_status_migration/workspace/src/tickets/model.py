"""Ticket model."""

from dataclasses import dataclass

from tickets.statuses import VALID_STATUSES


@dataclass(frozen=True)
class Ticket:
    status: str

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Unsupported ticket status: {self.status}")
