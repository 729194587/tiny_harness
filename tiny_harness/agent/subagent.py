"""Fresh-context execution for the synchronous task subagent."""

from collections.abc import Callable
from pathlib import Path

from tiny_harness.context.token_meter import DEFAULT_TOKEN_METER, TokenMeter
from tiny_harness.models.base import ModelProvider
from tiny_harness.agent.environment import EnvironmentAdapter
from tiny_harness.runtime.events import EventLogger, ScopedEventLogger, EventType, EventLogError
from tiny_harness.runtime.hooks import ToolHooks
from tiny_harness.runtime.permissions import PermissionPolicy, PermissionPrompt
from tiny_harness.runtime.recovery import RecoveryPolicy
from tiny_harness.runtime.skills import SkillCatalog
from tiny_harness.runtime.shell_runner import DEFAULT_SHELL_RUNNER, ShellRunner
from tiny_harness.runtime.test_runner import TestRunner

AgentEntrypoint = Callable[..., str]


class SubagentExecutor:
    """Run one delegated prompt through an isolated child Agent Loop."""

    def __init__(
        self,
        run_agent: AgentEntrypoint,
        provider: ModelProvider,
        workspace: Path,
        *,
        max_turns: int,
        permission_policy: PermissionPolicy,
        permission_prompt: PermissionPrompt | None,
        event_logger: EventLogger,
        max_context_tokens: int | None,
        token_meter: TokenMeter = DEFAULT_TOKEN_METER,
        tool_hooks: ToolHooks | None,
        recovery_policy: RecoveryPolicy,
        skill_catalog: SkillCatalog,
        shell_runner: ShellRunner = DEFAULT_SHELL_RUNNER,
        test_runner: TestRunner | None = None,
        memory_enabled: bool = False,
        environment_adapter: EnvironmentAdapter | None = None,
        progress_enabled: bool = False,
    ) -> None:
        self.run_agent = run_agent
        self.provider = provider
        self.workspace = workspace
        self.max_turns = max_turns
        self.permission_policy = permission_policy
        self.permission_prompt = permission_prompt
        self.event_logger = event_logger
        self.max_context_tokens = max_context_tokens
        self.token_meter = token_meter
        self.tool_hooks = tool_hooks
        self.recovery_policy = recovery_policy
        self.skill_catalog = skill_catalog
        self.test_runner = test_runner
        self.shell_runner = shell_runner
        self.memory_enabled = memory_enabled
        self.environment_adapter = environment_adapter
        self.progress_enabled = progress_enabled

    def __call__(self, prompt: str, parent_tool_call_id: str) -> str:
        child_messages = [
            {
                "role": "system",
                "content": (
                    "You are a coding subagent sharing the same workspace as "
                    "the parent agent. "
                    "Complete only the delegated task and return a concise "
                    "final answer. Use todo_write for multi-step work."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        child_logger = ScopedEventLogger(
            self.event_logger,
            {
                "agent_scope": "subagent",
                "parent_tool_call_id": parent_tool_call_id,
            },
        )
        child_logger.emit(EventType.SUBAGENT_STARTED)
        try:
            answer = self.run_agent(
                self.provider,
                self.workspace,
                child_messages,
                max_turns=self.max_turns,
                permission_policy=self.permission_policy,
                permission_prompt=self.permission_prompt,
                event_logger=child_logger,
                max_context_tokens=self.max_context_tokens,
                token_meter=self.token_meter,
                tool_hooks=self.tool_hooks,
                subagent_max_turns=self.max_turns,
                allow_subagent=False,
                recovery_policy=self.recovery_policy,
                skill_catalog=self.skill_catalog,
                test_runner=self.test_runner,
                shell_runner=self.shell_runner,
                memory_enabled=self.memory_enabled,
                memory_extraction_enabled=False,
                is_main_agent=False,
                **({"progress_enabled": True} if self.progress_enabled else {}),
                **({"environment_adapter": self.environment_adapter}
                   if self.environment_adapter is not None else {}),
            )
        except EventLogError:
            raise
        except Exception as error:
            child_logger.emit(EventType.SUBAGENT_FAILED, {"error_type": type(error).__name__})
            raise
        child_logger.emit(EventType.SUBAGENT_FINISHED)
        return answer or "(no summary)"
