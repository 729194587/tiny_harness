import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tiny_harness.agent.context import create_run_context
from tiny_harness.agent.loop import agent_loop as core_agent_loop
from tiny_harness.agent.loop import run_agent as agent_loop
from tiny_harness.agent.messages import ModelResponse, ToolCall
from tiny_harness.models.base import ModelErrorKind, ModelProviderError
from tiny_harness.runtime.context import ContextProtocolError
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.runtime.recovery import RecoveryPolicy
from tiny_harness.runtime.skills import discover_skills
from tiny_harness.tools.definition import ToolDefinition
from tiny_harness.tools.registry import ToolRegistry


class FakeProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
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


class AgentLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_loop_executes_a_discovered_tool_without_knowing_its_name(self) -> None:
        observed = []
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="phase_one_probe",
                description="Record one probe value.",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                execute=lambda call, arguments: (
                    observed.append((call.id, arguments["value"])) or "recorded"
                ),
            )
        )
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("probe-1", "phase_one_probe", '{"value":"ok"}')],
                    "tool_calls",
                ),
                ModelResponse("finished", None, [], "stop"),
            ]
        )

        class AllowPolicy:
            def decide(self, tool_name, arguments):
                del tool_name, arguments
                return PermissionDecision.ALLOW

        with patch(
            "tiny_harness.agent.context.discover_tools",
            return_value=registry,
        ):
            answer = agent_loop(
                provider,
                self.workspace,
                [{"role": "user", "content": "run probe"}],
                permission_policy=AllowPolicy(),
                allow_subagent=False,
            )

        self.assertEqual(answer, "finished")
        self.assertEqual(observed, [("probe-1", "ok")])
        self.assertEqual(
            [tool["function"]["name"] for tool in provider.calls[0]["tools"]],
            ["phase_one_probe"],
        )

    def test_three_argument_core_loop_runs_with_prebuilt_context(self) -> None:
        provider = FakeProvider(
            [ModelResponse("done", None, [], "stop")]
        )
        messages = [{"role": "user", "content": "task"}]
        context = create_run_context(
            provider,
            self.workspace,
            allow_subagent=False,
            skill_catalog=discover_skills(self.workspace, sources=()),
        )

        answer = core_agent_loop(messages, context, "task")

        self.assertEqual(answer, "done")
        self.assertEqual(messages[-1]["content"], "done")

    def test_active_request_mismatch_fails_before_provider_call(self) -> None:
        provider = FakeProvider(
            [ModelResponse("must not run", None, [], "stop")]
        )
        messages = [{"role": "user", "content": "actual task"}]
        context = create_run_context(
            provider,
            self.workspace,
            max_context_tokens=25_000,
            allow_subagent=False,
            skill_catalog=discover_skills(self.workspace, sources=()),
        )

        with self.assertRaises(ContextProtocolError):
            core_agent_loop(messages, context, "different task")

        self.assertEqual(provider.calls, [])
        self.assertEqual(messages, [{"role": "user", "content": "actual task"}])

    def test_rejects_skill_catalog_from_a_different_workspace(self) -> None:
        other_workspace = self.workspace / "other"
        other_workspace.mkdir()
        catalog = discover_skills(other_workspace, sources=())

        with self.assertRaisesRegex(
            ValueError,
            "Skill catalog workspace does not match run workspace",
        ):
            create_run_context(
                FakeProvider([]),
                self.workspace,
                skill_catalog=catalog,
            )

    def test_returns_final_text_and_appends_assistant_message(self) -> None:
        provider = FakeProvider(
            [ModelResponse("finished", None, [], "stop")]
        )
        messages = [{"role": "user", "content": "do the task"}]

        answer = agent_loop(provider, self.workspace, messages)

        self.assertEqual(answer, "finished")
        self.assertEqual(
            messages[-1],
            {"role": "assistant", "content": "finished"},
        )
        self.assertEqual(len(provider.calls), 1)

    def test_executes_multiple_tool_calls_before_next_model_call(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    "I will create both files.",
                    [
                        ToolCall(
                            "call-1",
                            "write_file",
                            '{"path":"a.txt","content":"A"}',
                        ),
                        ToolCall(
                            "call-2",
                            "write_file",
                            '{"path":"b.txt","content":"B"}',
                        ),
                    ],
                    "tool_calls",
                ),
                ModelResponse("created", None, [], "stop"),
            ]
        )
        messages = [{"role": "user", "content": "create two files"}]

        answer = agent_loop(provider, self.workspace, messages)

        self.assertEqual(answer, "created")
        self.assertEqual((self.workspace / "a.txt").read_text(encoding="utf-8"), "A")
        self.assertEqual((self.workspace / "b.txt").read_text(encoding="utf-8"), "B")
        self.assertEqual(len(provider.calls), 2)

        second_request = provider.calls[1]["messages"]
        self.assertEqual(second_request[-3]["role"], "assistant")
        self.assertEqual(
            second_request[-3]["reasoning_content"],
            "I will create both files.",
        )
        self.assertEqual(
            [call["id"] for call in second_request[-3]["tool_calls"]],
            ["call-1", "call-2"],
        )
        self.assertEqual(
            [message["tool_call_id"] for message in second_request[-2:]],
            ["call-1", "call-2"],
        )

    def test_tool_error_is_fed_back_to_model(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("call-1", "read_file", '{"path":"missing.txt"}')],
                    "tool_calls",
                ),
                ModelResponse("handled the error", None, [], "stop"),
            ]
        )
        messages = [{"role": "user", "content": "read the file"}]

        answer = agent_loop(provider, self.workspace, messages)

        self.assertEqual(answer, "handled the error")
        tool_message = provider.calls[1]["messages"][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "call-1")
        self.assertTrue(tool_message["content"].startswith("Error: FileNotFoundError:"))

    def test_multiple_tools_receive_independent_permission_decisions(self) -> None:
        command = (
            f'"{sys.executable}" -c '
            '"from pathlib import Path; Path(\'blocked.txt\').write_text(\'bad\')"'
        )
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall(
                            "write-call",
                            "write_file",
                            '{"path":"allowed.txt","content":"ok"}',
                        ),
                        ToolCall(
                            "bash-call",
                            "bash",
                            json.dumps({"command": command}),
                        ),
                    ],
                    "tool_calls",
                ),
                ModelResponse("continued after denial", None, [], "stop"),
            ]
        )
        messages = [{"role": "user", "content": "run both"}]

        answer = agent_loop(
            provider,
            self.workspace,
            messages,
            permission_prompt=lambda *_: False,
        )

        self.assertEqual(answer, "continued after denial")
        self.assertEqual(
            (self.workspace / "allowed.txt").read_text(encoding="utf-8"),
            "ok",
        )
        self.assertFalse((self.workspace / "blocked.txt").exists())
        tool_messages = provider.calls[1]["messages"][-2:]
        self.assertEqual(
            [message["tool_call_id"] for message in tool_messages],
            ["write-call", "bash-call"],
        )
        self.assertIn(
            "current shell command is not allowed",
            tool_messages[1]["content"],
        )

    def test_repeated_bash_denials_prompt_replanning_without_forcing_stop(self):
        denied_calls = [
            ToolCall(
                f"bash-{index}",
                "bash",
                json.dumps({"command": f"python verify_{index}.py"}),
            )
            for index in (1, 2)
        ]
        provider = FakeProvider(
            [
                ModelResponse(None, None, denied_calls, "tool_calls"),
                ModelResponse("finished after replanning", None, [], "stop"),
            ]
        )

        answer = agent_loop(provider, self.workspace, [], max_turns=2)

        self.assertEqual(answer, "finished after replanning")
        self.assertEqual(len(provider.calls), 2)
        tool_results = [
            message
            for message in provider.calls[1]["messages"]
            if message.get("role") == "tool"
        ]
        self.assertNotIn("Previous shell actions", tool_results[-2]["content"])
        self.assertIn("Previous shell actions", tool_results[-1]["content"])
        self.assertIn("choose another available tool", tool_results[-1]["content"])

    def test_passes_all_registered_tool_schemas(self) -> None:
        provider = FakeProvider([ModelResponse("done", None, [], "stop")])

        agent_loop(provider, self.workspace, [])

        names = [
            schema["function"]["name"]
            for schema in provider.calls[0]["tools"]
        ]
        self.assertEqual(
            names,
            [
                "read_file",
                "write_file",
                "edit_file",
                "list_files",
                "search_code",
                "glob",
                "grep",
                "bash",
                "load_skill",
                "task",
                "todo_write",
            ],
        )

    def test_executes_todo_and_file_tools_in_model_order(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall(
                            "todo-1",
                            "todo_write",
                            json.dumps(
                                {
                                    "todos": [
                                        {
                                            "content": "Create the file",
                                            "status": "in_progress",
                                        }
                                    ]
                                }
                            ),
                        ),
                        ToolCall(
                            "write-1",
                            "write_file",
                            '{"path":"planned.txt","content":"done"}',
                        ),
                    ],
                    "tool_calls",
                ),
                ModelResponse("finished", None, [], "stop"),
            ]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            answer = agent_loop(provider, self.workspace, [])

        self.assertEqual(answer, "finished")
        self.assertEqual(
            (self.workspace / "planned.txt").read_text(encoding="utf-8"),
            "done",
        )
        tool_messages = provider.calls[1]["messages"][-2:]
        self.assertEqual(
            [message["tool_call_id"] for message in tool_messages],
            ["todo-1", "write-1"],
        )
        self.assertIn("[>] Create the file", tool_messages[0]["content"])

    def test_todo_state_is_isolated_between_agent_runs(self) -> None:
        first_provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [
                        ToolCall(
                            "todo-1",
                            "todo_write",
                            '{"todos":[{"content":"First run only","status":"pending"}]}',
                        )
                    ],
                    "tool_calls",
                ),
                ModelResponse("first done", None, [], "stop"),
            ]
        )
        second_provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall(f"list-{index}", "list_files", "{}")],
                    "tool_calls",
                )
                for index in range(1, 4)
            ]
            + [ModelResponse("second done", None, [], "stop")]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            agent_loop(first_provider, self.workspace, [])
        agent_loop(second_provider, self.workspace, [])

        self.assertNotIn("First run only", json.dumps(second_provider.calls[-1]["messages"]))

    def test_rejects_non_positive_max_turns(self) -> None:
        provider = FakeProvider([])

        with self.assertRaisesRegex(ValueError, "at least 1"):
            agent_loop(provider, self.workspace, [], max_turns=0)

        self.assertEqual(provider.calls, [])

    def test_last_turn_finalizes_without_tools_and_sees_prior_tool_results(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("call-1", "list_files", "{}")],
                    "tool_calls",
                ),
                ModelResponse(
                    None,
                    None,
                    [ToolCall("call-2", "list_files", "{}")],
                    "tool_calls",
                ),
                ModelResponse("best available answer", None, [], "stop"),
            ]
        )

        answer = agent_loop(provider, self.workspace, [], max_turns=3)

        self.assertEqual(answer, "best available answer")
        self.assertEqual(len(provider.calls), 3)
        self.assertTrue(provider.calls[0]["tools"])
        self.assertTrue(provider.calls[1]["tools"])
        self.assertEqual(provider.calls[2]["tools"], [])
        final_messages = provider.calls[2]["messages"]
        self.assertEqual(
            [
                message["tool_call_id"]
                for message in final_messages
                if message.get("role") == "tool"
            ],
            ["call-1", "call-2"],
        )
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "execution turn budget is exhausted"
                in message.get("content", "")
                for message in final_messages
            )
        )

    def test_early_final_answer_skips_finalization_turn(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse(
                    None,
                    None,
                    [ToolCall("call-1", "list_files", "{}")],
                    "tool_calls",
                ),
                ModelResponse("done early", None, [], "stop"),
            ]
        )

        answer = agent_loop(provider, self.workspace, [], max_turns=3)

        self.assertEqual(answer, "done early")
        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(provider.calls[1]["tools"])

    def test_finalization_tool_call_is_not_executed_or_committed(self) -> None:
        provider = FakeProvider(
            [
                ModelResponse(
                    "best available answer",
                    None,
                    [ToolCall("forbidden-1", "list_files", "{}")],
                    "tool_calls",
                )
            ]
        )
        messages = [{"role": "user", "content": "inspect"}]

        answer = agent_loop(
            provider,
            self.workspace,
            messages,
            max_turns=1,
            allow_subagent=False,
        )

        self.assertEqual(answer, "best available answer")
        self.assertFalse(any(item.get("role") == "tool" for item in messages))
        self.assertFalse(any(item.get("tool_calls") for item in messages))
        self.assertEqual(messages[-1]["content"], "best available answer")

    def test_twenty_turn_budget_has_nineteen_tool_turns_then_finalization(self) -> None:
        responses = [
            ModelResponse(
                None,
                None,
                [ToolCall(f"call-{turn}", "list_files", "{}")],
                "tool_calls",
            )
            for turn in range(1, 20)
        ]
        responses.append(ModelResponse("done", None, [], "stop"))
        provider = FakeProvider(responses)

        answer = agent_loop(provider, self.workspace, [], max_turns=20)

        self.assertEqual(answer, "done")
        self.assertEqual(len(provider.calls), 20)
        self.assertTrue(all(call["tools"] for call in provider.calls[:19]))
        self.assertEqual(provider.calls[19]["tools"], [])
        for previous, current in zip(provider.calls[:18], provider.calls[1:19]):
            self.assertEqual(current["messages"][:len(previous["messages"])], previous["messages"])
            self.assertEqual(current["tools"], previous["tools"])
        self.assertNotIn("TinyHarness runtime state:", str(provider.calls))

    def test_provider_retry_stays_within_one_logical_turn(self) -> None:
        provider = FakeProvider(
            [
                ModelProviderError(ModelErrorKind.SERVER_UNAVAILABLE),
                ModelResponse("done", None, [], "stop"),
            ]
        )

        answer = agent_loop(
            provider,
            self.workspace,
            [],
            max_turns=2,
            recovery_policy=RecoveryPolicy(
                max_retries=1,
                base_delay_seconds=0,
                max_delay_seconds=0,
                jitter_ratio=0,
            ),
        )

        self.assertEqual(answer, "done")
        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(all(call["tools"] for call in provider.calls))

    def test_does_not_treat_length_stop_as_final_answer(self) -> None:
        provider = FakeProvider(
            [ModelResponse("partial", None, [], "length")]
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Model response is not executable: length",
        ):
            agent_loop(provider, self.workspace, [])

    def test_failed_finish_reason_never_commits_or_dispatches_tool_calls(self) -> None:
        for finish_reason in ("length", "content_filter"):
            with self.subTest(finish_reason=finish_reason):
                path = f"{finish_reason}.txt"
                provider = FakeProvider(
                    [
                        ModelResponse(
                            None,
                            None,
                            [
                                ToolCall(
                                    "write-1",
                                    "write_file",
                                    json.dumps(
                                        {"path": path, "content": "SIDE_EFFECT"}
                                    ),
                                )
                            ],
                            finish_reason,
                        )
                    ]
                )
                messages = [{"role": "user", "content": "task"}]

                with self.assertRaisesRegex(
                    RuntimeError,
                    f"Model response is not executable: {finish_reason}",
                ):
                    agent_loop(
                        provider,
                        self.workspace,
                        messages,
                        max_turns=1,
                        skill_catalog=discover_skills(self.workspace, sources=()),
                    )

                self.assertEqual(messages, [{"role": "user", "content": "task"}])
                self.assertFalse((self.workspace / path).exists())

    def test_inconsistent_success_finish_reason_fails_before_commit(self) -> None:
        cases = [
            ModelResponse(
                None,
                None,
                [
                    ToolCall(
                        "write-1",
                        "write_file",
                        '{"path":"bad.txt","content":"SIDE_EFFECT"}',
                    )
                ],
                "stop",
            ),
            ModelResponse(None, None, [], "tool_calls"),
        ]
        for response in cases:
            with self.subTest(finish_reason=response.finish_reason):
                messages = [{"role": "user", "content": "task"}]
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Model response is not executable",
                ):
                    agent_loop(
                        FakeProvider([response]),
                        self.workspace,
                        messages,
                        skill_catalog=discover_skills(self.workspace, sources=()),
                    )
                self.assertEqual(messages, [{"role": "user", "content": "task"}])
                self.assertFalse((self.workspace / "bad.txt").exists())


if __name__ == "__main__":
    unittest.main()
