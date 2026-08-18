import json
import unittest

from tiny_harness.agent.messages import ModelResponse
from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.runtime.recovery import (
    RecoveryExecutor,
    RecoveryPolicy,
    RecoveryState,
)


class SequenceProvider:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        if not self.outcomes:
            raise AssertionError("SequenceProvider has no outcome left")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class RecordingEventLogger:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event_type, data=None) -> None:
        self.events.append(
            {"event_type": event_type.value, "data": dict(data or {})}
        )


def model_error(kind, *, retry_after=None):
    return ModelProviderError(kind, retry_after_seconds=retry_after)


class RecoveryExecutorTest(unittest.TestCase):
    def executor(self, logger, delays, *, max_retries=2):
        return RecoveryExecutor(
            RecoveryPolicy(
                max_retries=max_retries,
                base_delay_seconds=1,
                max_delay_seconds=10,
                jitter_ratio=0,
            ),
            event_logger=logger,
            sleep=delays.append,
        )

    def test_transient_failures_retry_with_physical_attempt_events(self) -> None:
        logger = RecordingEventLogger()
        delays = []
        provider = SequenceProvider(
            [
                model_error(ModelErrorKind.RATE_LIMIT),
                model_error(ModelErrorKind.SERVER_UNAVAILABLE),
                ModelResponse("done", None, [], "stop"),
            ]
        )
        state = RecoveryState()

        response = self.executor(logger, delays).complete(
            provider,
            [],
            [],
            purpose="main",
            turn=3,
            state=state,
        )

        self.assertEqual(response.content, "done")
        self.assertEqual(provider.calls, 3)
        self.assertEqual(delays, [1, 2])
        requested = [
            event["data"]
            for event in logger.events
            if event["event_type"] == "model_requested"
        ]
        self.assertEqual(
            requested,
            [
                {"purpose": "main", "turn": 3, "attempt": 1},
                {"purpose": "main", "turn": 3, "attempt": 2},
                {"purpose": "main", "turn": 3, "attempt": 3},
            ],
        )
        responded = next(
            event["data"]
            for event in logger.events
            if event["event_type"] == "model_responded"
        )
        self.assertEqual(responded["purpose"], "main")
        self.assertEqual(responded["turn"], 3)
        self.assertEqual(responded["attempt"], 3)
        self.assertEqual(state.transient_retries_used, 2)

    def test_transient_counter_survives_a_reactive_recovery_boundary(self) -> None:
        logger = RecordingEventLogger()
        delays = []
        provider = SequenceProvider(
            [
                model_error(ModelErrorKind.SERVER_UNAVAILABLE),
                model_error(ModelErrorKind.CONTEXT_LENGTH),
                model_error(ModelErrorKind.CONNECTION),
                ModelResponse("done", None, [], "stop"),
            ]
        )
        state = RecoveryState()
        executor = self.executor(logger, delays)

        with self.assertRaises(ModelProviderError) as raised:
            executor.complete(
                provider,
                [],
                [],
                purpose="main",
                turn=1,
                state=state,
                context_recovery_available=True,
            )
        self.assertEqual(raised.exception.kind, ModelErrorKind.CONTEXT_LENGTH)
        state.reactive_compact_used = True

        response = executor.complete(
            provider,
            [],
            [],
            purpose="main",
            turn=1,
            state=state,
        )

        self.assertEqual(response.content, "done")
        self.assertEqual(state.attempt, 4)
        self.assertEqual(state.transient_retries_used, 2)
        self.assertEqual(delays, [1, 2])

    def test_retry_after_is_capped_by_policy(self) -> None:
        logger = RecordingEventLogger()
        delays = []
        provider = SequenceProvider(
            [
                model_error(ModelErrorKind.RATE_LIMIT, retry_after=30),
                ModelResponse("done", None, [], "stop"),
            ]
        )

        self.executor(logger, delays).complete(
            provider,
            [],
            [],
            purpose="summary",
            turn=2,
            state=RecoveryState(),
        )

        self.assertEqual(delays, [10])
        retry = next(
            event
            for event in logger.events
            if event["event_type"] == "model_retry_scheduled"
        )
        self.assertEqual(retry["data"]["purpose"], "summary")
        self.assertEqual(retry["data"]["delay_ms"], 10_000)

    def test_transient_exhaustion_and_fatal_errors_do_not_loop(self) -> None:
        logger = RecordingEventLogger()
        delays = []
        transient_provider = SequenceProvider(
            [
                model_error(ModelErrorKind.CONNECTION),
                model_error(ModelErrorKind.CONNECTION),
            ]
        )

        with self.assertRaises(ModelProviderError):
            self.executor(logger, delays, max_retries=1).complete(
                transient_provider,
                [],
                [],
                purpose="main",
                turn=1,
                state=RecoveryState(),
            )

        self.assertEqual(transient_provider.calls, 2)
        self.assertEqual(delays, [1])
        exhausted = [
            event
            for event in logger.events
            if event["event_type"] == "model_retry_exhausted"
        ]
        self.assertEqual(len(exhausted), 1)

        fatal_provider = SequenceProvider([model_error(ModelErrorKind.FATAL)])
        with self.assertRaises(ModelProviderError):
            self.executor(logger, delays).complete(
                fatal_provider,
                [],
                [],
                purpose="main",
                turn=2,
                state=RecoveryState(),
            )
        self.assertEqual(fatal_provider.calls, 1)

    def test_failure_events_do_not_record_exception_body(self) -> None:
        logger = RecordingEventLogger()
        provider = SequenceProvider(
            [
                ModelProviderError(
                    ModelErrorKind.FATAL,
                    message="PRIVATE_EXCEPTION_BODY",
                )
            ]
        )

        with self.assertRaises(ModelProviderError):
            self.executor(logger, []).complete(
                provider,
                [],
                [],
                purpose="main",
                turn=1,
                state=RecoveryState(),
            )

        serialized = json.dumps(logger.events)
        self.assertNotIn("PRIVATE_EXCEPTION_BODY", serialized)
        failed = next(
            event
            for event in logger.events
            if event["event_type"] == "model_request_failed"
        )
        self.assertEqual(failed["data"]["error_kind"], "fatal")


if __name__ == "__main__":
    unittest.main()
