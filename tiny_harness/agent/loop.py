"""The minimal TinyHarness agent loop."""

from pathlib import Path
from typing import Any

from tiny_harness.models.base import ModelProvider
from tiny_harness.runtime.context import prepare_context
from tiny_harness.runtime.events import (
    NULL_EVENT_LOGGER,
    EventLogError,
    EventLogger,
    EventType,
    ScopedEventLogger,
)
from tiny_harness.runtime.hooks import ToolHooks
from tiny_harness.runtime.permissions import (
    DEFAULT_PERMISSION_POLICY,
    PermissionPolicy,
    PermissionPrompt,
)
from tiny_harness.runtime.todos import TodoManager
from tiny_harness.tools.registry import dispatch, tool_schemas
from tiny_harness.tools.task import SubagentRunner

TODO_REMINDER_ROUNDS = 3
DEFAULT_SUBAGENT_MAX_TURNS = 10


def agent_loop(
    provider: ModelProvider,
    workspace: Path,
    messages: list[dict[str, Any]],
    *,
    max_turns: int = 20,
    permission_policy: PermissionPolicy = DEFAULT_PERMISSION_POLICY,
    permission_prompt: PermissionPrompt | None = None,
    event_logger: EventLogger = NULL_EVENT_LOGGER,
    max_context_chars: int | None = None,
    tool_hooks: ToolHooks | None = None,
    subagent_max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS,
    allow_subagent: bool = True,
) -> str:
    """Call the model and tools until a final text response is returned."""

    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    if max_context_chars is not None and max_context_chars < 1:
        raise ValueError("max_context_chars must be at least 1")
    if subagent_max_turns < 1:
        raise ValueError("subagent_max_turns must be at least 1")

    tools = tool_schemas(include_task=allow_subagent)
    todo_manager = TodoManager()
    rounds_since_todo = 0
    current_turn = 0
    run_data = {"max_turns": max_turns}
    if allow_subagent:
        run_data["subagent_max_turns"] = subagent_max_turns
    if max_context_chars is not None:
        run_data["max_context_chars"] = max_context_chars
    event_logger.emit(EventType.RUN_STARTED, run_data)

    subagent_runner: SubagentRunner | None = None
    if allow_subagent:

        def run_subagent(prompt: str, parent_tool_call_id: str) -> str:
            print("\n[Subagent started]")
            child_messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a coding subagent working in {workspace}. "
                        "Complete only the delegated task and return a concise "
                        "final answer. Use todo_write for multi-step work."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            child_logger = ScopedEventLogger(
                event_logger,
                {
                    "agent_scope": "subagent",
                    "parent_tool_call_id": parent_tool_call_id,
                },
            )
            try:
                answer = agent_loop(
                    provider,
                    workspace,
                    child_messages,
                    max_turns=subagent_max_turns,
                    permission_policy=permission_policy,
                    permission_prompt=permission_prompt,
                    event_logger=child_logger,
                    max_context_chars=max_context_chars,
                    tool_hooks=tool_hooks,
                    subagent_max_turns=subagent_max_turns,
                    allow_subagent=False,
                )
            except Exception:
                print("[Subagent failed]")
                raise
            print("[Subagent done]")
            return answer or "(no summary)"

        subagent_runner = run_subagent

    try:
        for current_turn in range(1, max_turns + 1):
            request_messages = messages
            if max_context_chars is not None:
                prepared = prepare_context(messages, tools, max_context_chars)
                request_messages = prepared.messages
                if prepared.dropped_blocks:
                    event_logger.emit(
                        EventType.CONTEXT_TRIMMED,
                        {
                            "turn": current_turn,
                            "before_chars": prepared.before_chars,
                            "after_chars": prepared.after_chars,
                            "dropped_blocks": prepared.dropped_blocks,
                            "dropped_messages": prepared.dropped_messages,
                        },
                    )
            event_logger.emit(
                EventType.MODEL_REQUESTED,
                {"turn": current_turn},
            )
            response = provider.complete(request_messages, tools)
            event_logger.emit(
                EventType.MODEL_RESPONDED,
                {
                    "turn": current_turn,
                    "finish_reason": response.finish_reason,
                    "tool_call_count": len(response.tool_calls),
                    "content_length": len(response.content or ""),
                },
            )

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": response.content,
            }
            if response.reasoning_content is not None:
                assistant_message["reasoning_content"] = response.reasoning_content
            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments_json,
                        },
                    }
                    for call in response.tool_calls
                ]
            messages.append(assistant_message)

            if not response.tool_calls:
                if response.finish_reason != "stop":
                    raise RuntimeError(
                        "Model stopped without a final answer: "
                        f"{response.finish_reason}"
                    )
                answer = response.content or ""
                event_logger.emit(
                    EventType.RUN_FINISHED,
                    {
                        "turns": current_turn,
                        "answer_length": len(answer),
                    },
                )
                return answer

            todo_revision = todo_manager.revision
            for call in response.tool_calls:
                result = dispatch(
                    workspace,
                    call,
                    permission_policy=permission_policy,
                    permission_prompt=permission_prompt,
                    event_logger=event_logger,
                    tool_hooks=tool_hooks,
                    todo_manager=todo_manager,
                    subagent_runner=subagent_runner,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": result.content,
                    }
                )

            if todo_manager.revision != todo_revision:
                rounds_since_todo = 0
            else:
                rounds_since_todo += 1

            if rounds_since_todo >= TODO_REMINDER_ROUNDS:
                reminder = (
                    "<todo-reminder>\n"
                    "Update your todo list.\n\n"
                    "Current todos:\n"
                    f"{todo_manager.render()}\n"
                    "</todo-reminder>"
                )
                messages[-1]["content"] += f"\n\n{reminder}"
                event_logger.emit(
                    EventType.TODO_REMINDER,
                    {
                        "turn": current_turn,
                        "rounds_since_todo": rounds_since_todo,
                        "todo_count": len(todo_manager.items),
                    },
                )
                rounds_since_todo = 0

        raise RuntimeError(f"Maximum model turns reached: {max_turns}")
    except EventLogError:
        raise
    except Exception as error:
        event_logger.emit(
            EventType.RUN_FAILED,
            {
                "turn": current_turn,
                "error_type": type(error).__name__,
            },
        )
        raise
