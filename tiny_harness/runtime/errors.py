"""Runtime termination errors that are independent of Goal evaluation."""


class MaxTurnsExceededError(RuntimeError):
    """The Agent exhausted its logical-turn budget before it could stop."""
