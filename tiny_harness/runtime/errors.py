"""Runtime termination errors."""


class MaxTurnsExceededError(RuntimeError):
    """The Agent exhausted its logical-turn budget before it could stop."""
