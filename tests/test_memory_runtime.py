import copy
import json
import tempfile
import unittest
from pathlib import Path

from tiny_harness.agent.loop import run_agent
from tiny_harness.agent.messages import ModelResponse
from tiny_harness.agent.session import AgentSession
from tiny_harness.memory import MemoryRuntime, create_memory_runtime


class FakeProvider:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
            }
        )
        if not self.responses:
            raise AssertionError("FakeProvider has no response left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingEventLogger:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event_type, data=None) -> None:
        self.events.append(
            {"event_type": event_type.value, "data": dict(data or {})}
        )


class MemoryRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_memory(
        self,
        *,
        filename: str = "tabs.md",
        name: str = "tabs",
        description: str = "User prefers tabs for indentation",
        body: str = "RUNTIME_MEMORY_BODY_SENTINEL",
    ) -> Path:
        path = self.workspace / ".tinyharness" / "memory" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            "type: user\n"
            "---\n\n"
            f"{body}\n",
            encoding="utf-8",
        )
        return path

    def test_public_runtime_facade_preserves_disabled_behavior(self) -> None:
        def complete_for(purpose, messages, tools):
            del purpose, messages, tools
            raise AssertionError("disabled Memory must not call the model")

        runtime = create_memory_runtime(
            self.workspace,
            enabled=False,
            extraction_enabled=True,
            complete_for=complete_for,
            event_logger=RecordingEventLogger(),
        )
        messages = [
            {"role": "system", "name": "tinyharness_memory_catalog", "content": "old"},
            {"role": "user", "name": "tinyharness_relevant_memory", "content": "old"},
            {"role": "user", "content": "task"},
        ]

        runtime.initialize(messages, "task")

        self.assertIsInstance(runtime, MemoryRuntime)
        self.assertEqual(messages, [{"role": "user", "content": "task"}])
        self.assertEqual(runtime.run_metadata(), {})
        self.assertIsNone(runtime.final_answer_hook)

    def test_side_query_loads_body_before_main_and_extracts_after_stop(self) -> None:
        self.write_memory()
        provider = FakeProvider(
            [
                ModelResponse(
                    '{"selected_memories":["tabs.md"]}',
                    None,
                    [],
                    "stop",
                ),
                ModelResponse("done", None, [], "stop"),
                ModelResponse('{"memories":[]}', None, [], "stop"),
            ]
        )
        logger = RecordingEventLogger()
        messages = [{"role": "user", "content": "use my tab preference"}]

        answer = run_agent(
            provider,
            self.workspace,
            messages,
            memory_enabled=True,
            max_context_chars=100_000,
            event_logger=logger,
        )

        self.assertEqual(answer, "done")
        self.assertEqual(len(provider.calls), 3)
        selection_request, main_request, extraction_request = provider.calls
        self.assertEqual(selection_request["tools"], [])
        self.assertNotIn(
            "RUNTIME_MEMORY_BODY_SENTINEL",
            json.dumps(selection_request, ensure_ascii=False),
        )
        self.assertIn(
            "RUNTIME_MEMORY_BODY_SENTINEL",
            json.dumps(main_request, ensure_ascii=False),
        )
        self.assertEqual(extraction_request["tools"], [])
        self.assertIn(
            "Extract only durable cross-session Memory",
            extraction_request["messages"][0]["content"],
        )

        self.assertEqual(logger.events[0]["event_type"], "run_started")
        requested = [
            event["data"]["purpose"]
            for event in logger.events
            if event["event_type"] == "model_requested"
        ]
        self.assertEqual(
            requested,
            ["memory_selection", "main", "memory_extraction"],
        )
        self.assertNotIn(
            "RUNTIME_MEMORY_BODY_SENTINEL",
            json.dumps(logger.events, ensure_ascii=False),
        )

    def test_session_extracts_then_selects_memory_on_next_submit(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse("noted", None, [], "stop"),
                ModelResponse(
                    json.dumps(
                        {
                            "memories": [
                                {
                                    "name": "user-tabs",
                                    "type": "user",
                                    "description": "User prefers tabs",
                                    "body": "Always use tabs for indentation.",
                                }
                            ]
                        }
                    ),
                    None,
                    [],
                    "stop",
                ),
                ModelResponse(
                    '{"selected_memories":["user-tabs.md"]}',
                    None,
                    [],
                    "stop",
                ),
                ModelResponse("created", None, [], "stop"),
                ModelResponse('{"memories":[]}', None, [], "stop"),
            ]
        )
        session = AgentSession(
            provider,
            self.workspace,
            "system",
            memory_enabled=True,
        )

        self.assertEqual(
            session.submit("I prefer tabs. Remember that."),
            "noted",
        )
        memory_path = (
            self.workspace
            / ".tinyharness"
            / "memory"
            / "user-tabs.md"
        )
        self.assertTrue(memory_path.exists())
        self.assertEqual(session.submit("Create the next file"), "created")

        second_main_request = provider.calls[3]
        self.assertIn(
            "Always use tabs for indentation.",
            json.dumps(second_main_request, ensure_ascii=False),
        )
        self.assertFalse(
            any(
                message.get("name")
                in {
                    "tinyharness_memory_catalog",
                    "tinyharness_relevant_memory",
                }
                for message in session.messages
            )
        )

    def test_invalid_selector_falls_back_without_blocking_main_run(self) -> None:
        self.write_memory()
        provider = FakeProvider(
            [
                ModelResponse("not-json", None, [], "stop"),
                ModelResponse("done", None, [], "stop"),
                ModelResponse('{"memories":[]}', None, [], "stop"),
            ]
        )
        logger = RecordingEventLogger()

        answer = run_agent(
            provider,
            self.workspace,
            [{"role": "user", "content": "please use tabs"}],
            memory_enabled=True,
            event_logger=logger,
        )

        self.assertEqual(answer, "done")
        self.assertIn(
            "RUNTIME_MEMORY_BODY_SENTINEL",
            json.dumps(provider.calls[1], ensure_ascii=False),
        )
        selected = next(
            event for event in logger.events if event["event_type"] == "memory_selected"
        )
        self.assertEqual(selected["data"]["method"], "keyword")
        self.assertTrue(selected["data"]["selection_fallback"])
        self.assertEqual(
            selected["data"]["selection_failure_type"],
            "MemorySelectionError",
        )

    def test_extraction_failure_is_fail_open_and_metadata_only(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse("answer", None, [], "stop"),
                ModelResponse("PRIVATE_INVALID_OUTPUT", None, [], "stop"),
            ]
        )
        logger = RecordingEventLogger()

        answer = run_agent(
            provider,
            self.workspace,
            [{"role": "user", "content": "task"}],
            memory_enabled=True,
            event_logger=logger,
        )

        self.assertEqual(answer, "answer")
        self.assertFalse((self.workspace / ".tinyharness").exists())
        failed = next(
            event
            for event in logger.events
            if event["event_type"] == "memory_extraction_failed"
        )
        self.assertEqual(failed["data"]["error_type"], "MemoryExtractionError")
        self.assertNotIn("PRIVATE_INVALID_OUTPUT", json.dumps(logger.events))

    def test_memory_is_opt_in(self) -> None:
        self.write_memory()
        provider = FakeProvider([ModelResponse("done", None, [], "stop")])

        answer = run_agent(
            provider,
            self.workspace,
            [{"role": "user", "content": "use tabs"}],
        )

        self.assertEqual(answer, "done")
        self.assertEqual(len(provider.calls), 1)
        self.assertNotIn(
            "RUNTIME_MEMORY_BODY_SENTINEL",
            json.dumps(provider.calls[0], ensure_ascii=False),
        )

    def test_tenth_written_memory_triggers_consolidation_after_extraction(self) -> None:
        for index in range(9):
            self.write_memory(
                filename=f"fact-{index}.md",
                name=f"fact-{index}",
                description=f"Durable fact {index}",
                body=f"BODY-{index}",
            )
        provider = FakeProvider(
            [
                ModelResponse('{"selected_memories":[]}', None, [], "stop"),
                ModelResponse("main answer", None, [], "stop"),
                ModelResponse(
                    json.dumps(
                        {
                            "memories": [
                                {
                                    "name": "fact-9",
                                    "type": "project",
                                    "description": "Durable fact 9",
                                    "body": "BODY-9",
                                }
                            ]
                        }
                    ),
                    None,
                    [],
                    "stop",
                ),
                ModelResponse(
                    json.dumps(
                        {
                            "memories": [
                                {
                                    "name": "combined-facts",
                                    "type": "project",
                                    "description": "Consolidated durable facts",
                                    "body": "BODY-0 through BODY-9",
                                }
                            ]
                        }
                    ),
                    None,
                    [],
                    "stop",
                ),
            ]
        )
        logger = RecordingEventLogger()

        answer = run_agent(
            provider,
            self.workspace,
            [{"role": "user", "content": "Remember one more durable fact"}],
            memory_enabled=True,
            max_context_chars=100_000,
            event_logger=logger,
        )

        self.assertEqual(answer, "main answer")
        requested = [
            event["data"]["purpose"]
            for event in logger.events
            if event["event_type"] == "model_requested"
        ]
        self.assertEqual(
            requested,
            [
                "memory_selection",
                "main",
                "memory_extraction",
                "memory_consolidation",
            ],
        )
        consolidation_call = provider.calls[3]
        self.assertEqual(consolidation_call["tools"], [])
        consolidation_text = json.dumps(
            consolidation_call["messages"],
            ensure_ascii=False,
        )
        self.assertIn("modified_at", consolidation_text)
        self.assertNotIn("modified_ns", consolidation_text)
        active = list(
            (
                self.workspace / ".tinyharness" / "memory"
            ).glob("*.md")
        )
        self.assertEqual(
            {path.name for path in active},
            {"MEMORY.md", "combined-facts.md"},
        )
        archived = list(
            (
                self.workspace / ".tinyharness" / "memory" / "archive"
            ).glob("*/*.md")
        )
        self.assertEqual(len(archived), 10)
        completed = next(
            event
            for event in logger.events
            if event["event_type"] == "memory_consolidation_completed"
        )
        self.assertEqual(completed["data"]["before_count"], 10)
        self.assertEqual(completed["data"]["after_count"], 1)
        self.assertNotIn("BODY-0", json.dumps(logger.events, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
