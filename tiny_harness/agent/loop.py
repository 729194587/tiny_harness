"""The minimal TinyHarness agent loop."""

import copy
from pathlib import Path
from typing import Any

from tiny_harness.agent.messages import ModelResponse
from tiny_harness.models.base import (
    ModelErrorKind,
    ModelProvider,
    ModelProviderError,
)
from tiny_harness.runtime.context import (
    CompactionRequest,
    ContextCompactor,
    context_char_count,
)
from tiny_harness.runtime.events import (
    NULL_EVENT_LOGGER,
    EventLogError,
    EventLogger,
    EventType,
    ScopedEventLogger,
)
from tiny_harness.runtime.goal import (
    DEFAULT_MAX_GOAL_RETRIES,
    GoalController,
    GoalEvaluator,
    GoalNotAchievedError,
    PromptGoalEvaluator,
)
from tiny_harness.runtime.hooks import ToolHooks
from tiny_harness.runtime.permissions import (
    DEFAULT_PERMISSION_POLICY,
    PermissionPolicy,
    PermissionPrompt,
)
from tiny_harness.runtime.recovery import (
    RecoveryExecutor,
    RecoveryPolicy,
    RecoveryState,
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
    recovery_policy: RecoveryPolicy = RecoveryPolicy(),
    goal_condition: str | None = None,
    max_goal_retries: int = DEFAULT_MAX_GOAL_RETRIES,
    goal_evaluator: GoalEvaluator | None = None,
) -> str:
    """Call the model and tools until a final text response is returned."""

    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    if max_context_chars is not None and max_context_chars < 1:
        raise ValueError("max_context_chars must be at least 1")
    if subagent_max_turns < 1:
        raise ValueError("subagent_max_turns must be at least 1")
    if max_goal_retries < 0:
        raise ValueError("max_goal_retries must be at least 0")
    if goal_evaluator is not None and goal_condition is None:
        raise ValueError("goal_evaluator requires goal_condition")

    tools = tool_schemas(
        include_task=allow_subagent,
        include_compact=max_context_chars is not None,
    )
    todo_manager = TodoManager()
    current_turn = 0
    recovery_executor = RecoveryExecutor(
        recovery_policy,
        event_logger=event_logger,
    )

    def complete_summary(
        summary_messages: list[dict[str, Any]],
        summary_tools: list[dict[str, Any]],
    ) -> ModelResponse:
        return recovery_executor.complete(
            provider,
            summary_messages,
            summary_tools,
            purpose="summary",
            turn=current_turn,
            state=RecoveryState(),
        )

    def complete_goal_evaluation(
        evaluation_messages: list[dict[str, Any]],
        evaluation_tools: list[dict[str, Any]],
    ) -> ModelResponse:
        return recovery_executor.complete(
            provider,
            evaluation_messages,
            evaluation_tools,
            purpose="goal_evaluation",
            turn=current_turn,
            state=RecoveryState(),
        )

    goal_controller: GoalController | None = None
    if goal_condition is not None:
        evaluator = goal_evaluator or PromptGoalEvaluator(
            complete_goal_evaluation,
            max_context_chars=max_context_chars,
        )
        goal_controller = GoalController(
            goal_condition,
            evaluator,
            max_retries=max_goal_retries,
        )
        goal_controller.upsert_marker(messages)

    compaction_request = (
        CompactionRequest() if max_context_chars is not None else None
    )
    compactor = (
        ContextCompactor(
            workspace,
            provider,
            tools,
            max_context_chars,
            event_logger=event_logger,
            summary_complete=complete_summary,
        )
        if max_context_chars is not None
        else None
    )
    rounds_since_todo = 0
    run_data = {
        "max_turns": max_turns,
        "max_model_retries": recovery_policy.max_retries,
    }
    if allow_subagent:
        run_data["subagent_max_turns"] = subagent_max_turns
    if max_context_chars is not None:
        run_data["max_context_chars"] = max_context_chars
    if goal_controller is not None:
        run_data["goal_enabled"] = True
        run_data["max_goal_retries"] = max_goal_retries
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
                    recovery_policy=recovery_policy,
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
            if compactor is not None:
                prepared = compactor.prepare(messages, todo_manager.render())
                messages[:] = prepared.messages
                request_messages = copy.deepcopy(messages)
            recovery_state = RecoveryState()
            while True:
                try:
                    response = recovery_executor.complete(
                        provider,
                        request_messages,
                        tools,
                        purpose="main",
                        turn=current_turn,
                        state=recovery_state,
                        context_recovery_available=(
                            compactor is not None
                            and not recovery_state.reactive_compact_used
                        ),
                    )
                    break
                except ModelProviderError as error:
                    if (
                        error.kind is not ModelErrorKind.CONTEXT_LENGTH
                        or compactor is None
                        or recovery_state.reactive_compact_used
                    ):
                        raise

                    # Mark the single reactive opportunity before doing any
                    # recovery work. A failed summary must never recurse.
                    recovery_state.reactive_compact_used = True
                    failed_request_chars = context_char_count(
                        request_messages,
                        tools,
                    )
                    prepared = compactor.reactive_compact(
                        messages,
                        todo_manager.render(),
                        failed_request_chars=failed_request_chars,
                    )
                    messages[:] = prepared.messages
                    request_messages = copy.deepcopy(messages)
                    event_logger.emit(
                        EventType.MODEL_RETRY_SCHEDULED,
                        {
                            "purpose": "main",
                            "turn": current_turn,
                            "attempt": recovery_state.attempt,
                            "delay_ms": 0,
                            "error_kind": error.kind.value,
                            "recovery": "reactive_compact",
                        },
                    )

            if response.finish_reason == "stop":
                if response.tool_calls:
                    raise RuntimeError(
                        "Model response is not executable: stop with tool calls"
                    )
            elif response.finish_reason == "tool_calls":
                if not response.tool_calls:
                    raise RuntimeError(
                        "Model response is not executable: tool_calls without calls"
                    )
            else:
                raise RuntimeError(
                    "Model response is not executable: "
                    f"{response.finish_reason}"
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
            if not response.tool_calls:
                answer = response.content or ""
                if goal_controller is not None:
                    evaluation_number = goal_controller.state.evaluations + 1
                    event_logger.emit(
                        EventType.GOAL_EVALUATION_REQUESTED,
                        {
                            "turn": current_turn,
                            "evaluation": evaluation_number,
                        },
                    )
                    decision = goal_controller.evaluate(messages, answer)
                    if (
                        decision.action == "block"
                        and current_turn < max_turns
                    ):
                        goal_controller.record_continuation()
                    event_logger.emit(
                        EventType.GOAL_EVALUATED,
                        {
                            "turn": current_turn,
                            "evaluation": goal_controller.state.evaluations,
                            "outcome": decision.action,
                            "reason_length": len(decision.reason),
                            "retries_used": goal_controller.state.retries_used,
                        },
                    )
                    if decision.action == "block":
                        if current_turn < max_turns:
                            goal_controller.upsert_marker(messages)
                        continue
                    if decision.action == "impossible":
                        raise GoalNotAchievedError(
                            "Goal evaluator determined completion is impossible: "
                            f"{decision.reason}"
                        )
                    if decision.action == "limit":
                        raise GoalNotAchievedError(
                            "Goal remains unverified after maximum automatic "
                            f"continuations: {decision.reason}"
                        )
                    if decision.action != "achieved":
                        raise GoalNotAchievedError(
                            f"Unknown goal decision: {decision.action}"
                        )
                messages.append(assistant_message)
                finish_data = {
                    "turns": current_turn,
                    "answer_length": len(answer),
                }
                if goal_controller is not None:
                    finish_data["goal_evaluations"] = (
                        goal_controller.state.evaluations
                    )
                event_logger.emit(
                    EventType.RUN_FINISHED,
                    finish_data,
                )
                return answer

            messages.append(assistant_message)

            todo_revision = todo_manager.revision
            compact_revision = (
                compaction_request.revision
                if compaction_request is not None
                else 0
            )
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
                    compaction_request=compaction_request,
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

            if (
                compactor is not None
                and compaction_request is not None
                and compaction_request.revision != compact_revision
            ):
                prepared = compactor.compact_history(
                    messages,
                    todo_manager.render(),
                    reason="manual",
                )
                messages[:] = prepared.messages

        if goal_controller is not None:
            raise GoalNotAchievedError(
                "Maximum model turns reached before goal verification: "
                f"{max_turns}"
            )
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
