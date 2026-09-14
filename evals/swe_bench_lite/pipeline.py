"""Serial SWE-bench Lite Dev smoke rollout orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tiny_harness.__main__ import DEFAULT_MAX_CONTEXT_TOKENS
from tiny_harness.agent.loop import run_agent
from tiny_harness.agent.environment import EnvironmentAdapter
from tiny_harness.environments import CodingEnvironmentAdapter
from tiny_harness.models.base import ModelProvider
from tiny_harness.runtime.events import JsonlEventLogger
from tiny_harness.runtime.context import CompactionConfig
from tiny_harness.runtime.permissions import PermissionDecision
from tiny_harness.runtime.tool_trace import ToolTraceConfig

from .calibration import CalibrationResult, calibrate_task
from .data import SweEvaluationBundle, SweTask, load_agent_tasks, load_evaluation_bundles
from .docker_workspace import DockerTaskEnvironment
from .evaluator import new_run_id
from .observability import WorkspaceMutationLogger, source_metadata

DEFAULT_SELECTED_TASKS = Path(__file__).with_name("selected_tasks.jsonl")
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
AGENT_SYSTEM_PROMPT = (
    "You are a coding agent working in the current repository workspace. "
    "Use relative paths with all tools. "
    "Fix the reported issue in the current repository. Inspect the code and tests "
    "as needed. Make the smallest correct change. Run relevant tests when useful."
)


class ContainerPermissionPolicy:
    """Allow discovered tools inside the task's disposable container boundary."""

    def decide(self, tool_name: str, arguments: Mapping[str, Any]) -> PermissionDecision:
        del tool_name, arguments
        return PermissionDecision.ALLOW


@dataclass(frozen=True)
class RolloutResult:
    instance_id: str
    final_answer: str
    model_patch: str


def _experiment_config(
    max_turns: int,
    subagent_max_turns: int,
    max_context_tokens: int | None,
    environment_adapter: EnvironmentAdapter | None,
    working_context_trigger_tokens: int,
    working_context_target_tokens: int,
    keep_recent_tool_batches: int,
) -> dict[str, Any]:
    """Use identical configuration fields in task and run metadata."""
    adapter_type = type(environment_adapter)
    return {
        "max_turns": max_turns,
        "subagent_max_turns": subagent_max_turns,
        "max_context_tokens": max_context_tokens,
        "working_context_trigger_tokens": working_context_trigger_tokens,
        "working_context_target_tokens": working_context_target_tokens,
        "keep_recent_tool_batches": keep_recent_tool_batches,
        "memory_enabled": False,
        "workspace_mutation_observation_enabled": True,
        "environment_adapter": (
            f"{adapter_type.__module__}.{adapter_type.__qualname__}"
            if environment_adapter is not None else None
        ),
        "coding_environment_enabled": isinstance(environment_adapter, CodingEnvironmentAdapter),
    }


def rollout_task(
    task: SweTask,
    provider: ModelProvider,
    output_dir: Path,
    *,
    model_name_or_path: str,
    max_turns: int = 20,
    subagent_max_turns: int = 10,
    max_context_tokens: int | None = DEFAULT_MAX_CONTEXT_TOKENS,
    working_context_trigger_tokens: int = CompactionConfig.working_context_trigger_tokens,
    working_context_target_tokens: int = CompactionConfig.working_context_target_tokens,
    keep_recent_tool_batches: int = CompactionConfig.keep_recent_tool_batches,
    environment_factory: Callable[..., DockerTaskEnvironment] = DockerTaskEnvironment,
    agent_entrypoint: Callable[..., str] = run_agent,
    environment_adapter: EnvironmentAdapter | None = None,
) -> RolloutResult:
    """Run an agent using only SweTask; evaluator bundles cannot enter this API."""

    provenance = source_metadata()
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    final_path = output_dir / "final_answer.txt"
    patch_path = output_dir / "model.patch"
    final_path.touch(exist_ok=True)
    patch_path.touch(exist_ok=True)
    events_path.touch(exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    experiment_config = _experiment_config(
        max_turns, subagent_max_turns, max_context_tokens,
        environment_adapter,
        working_context_trigger_tokens, working_context_target_tokens, keep_recent_tool_batches,
    )
    experiment_config.update(provenance)
    (output_dir / "metadata.json").write_text(json.dumps({
        **experiment_config, "instance_id": task.instance_id, "status": "RUNNING",
        "model_name_or_path": model_name_or_path, "started_at": started_at,
    }, indent=2) + "\n", encoding="utf-8")
    try:
        with environment_factory(task, network_mode="none") as environment:
            if environment.workspace is None or environment.shell_runner is None:
                raise RuntimeError("Docker task environment did not start")
            answer = agent_entrypoint(
                provider,
                environment.workspace,
                [
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": task.problem_statement},
                ],
                max_turns=max_turns,
                subagent_max_turns=subagent_max_turns,
                max_context_tokens=max_context_tokens,
                working_context_trigger_tokens=working_context_trigger_tokens,
                working_context_target_tokens=working_context_target_tokens,
                keep_recent_tool_batches=keep_recent_tool_batches,
                permission_policy=ContainerPermissionPolicy(),
                event_logger=WorkspaceMutationLogger(JsonlEventLogger(events_path), environment.workspace),
                tool_trace=ToolTraceConfig(enabled=True, result_preview_chars=200),
                shell_runner=environment.shell_runner,
                memory_enabled=False,
                **({"environment_adapter": environment_adapter}
                   if environment_adapter is not None else {}),
            )
            model_patch = environment.collect_patch()
        final_path.write_text(answer, encoding="utf-8")
        patch_path.write_text(model_patch, encoding="utf-8", newline="\n")
        metadata = {
            "instance_id": task.instance_id,
            "status": "COMPLETED",
            **experiment_config,
            "model_name_or_path": model_name_or_path,
            "max_context_tokens": max_context_tokens,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        result = RolloutResult(task.instance_id, answer, model_patch)
    except BaseException as error:
        metadata = {
            "instance_id": task.instance_id,
            "status": "FAILED",
            **experiment_config,
            "model_name_or_path": model_name_or_path,
            "max_context_tokens": max_context_tokens,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        raise
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return result


def write_predictions(
    path: Path,
    rollouts: Sequence[RolloutResult],
    model_name_or_path: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for rollout in rollouts:
            stream.write(
                json.dumps(
                    {
                        "instance_id": rollout.instance_id,
                        "model_name_or_path": model_name_or_path,
                        "model_patch": rollout.model_patch,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def select_instances(values: Sequence[Any], instance_id: str | None) -> list[Any]:
    if instance_id is None:
        return list(values)
    selected = [value for value in values if value.instance_id == instance_id]
    if not selected:
        raise ValueError(f"Unknown selected smoke instance: {instance_id}")
    return selected


def run_selected_smoke(
    provider: ModelProvider,
    *,
    model_name_or_path: str,
    selected_path: Path = DEFAULT_SELECTED_TASKS,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    run_id: str | None = None,
    instance_id: str | None = None,
    max_turns: int = 20,
    subagent_max_turns: int = 10,
    max_context_tokens: int | None = DEFAULT_MAX_CONTEXT_TOKENS,
    working_context_trigger_tokens: int = CompactionConfig.working_context_trigger_tokens,
    working_context_target_tokens: int = CompactionConfig.working_context_target_tokens,
    keep_recent_tool_batches: int = CompactionConfig.keep_recent_tool_batches,
    calibrator: Callable[..., CalibrationResult] = calibrate_task,
    rollout: Callable[..., RolloutResult] = rollout_task,
    environment_adapter: EnvironmentAdapter | None = None,
) -> Path:
    """Calibrate then serially roll out the selected four-task smoke set."""

    active_run_id = run_id or new_run_id("smoke")
    experiment_config = _experiment_config(
        max_turns, subagent_max_turns, max_context_tokens,
        environment_adapter,
        working_context_trigger_tokens, working_context_target_tokens, keep_recent_tool_batches,
    )
    experiment_config.update(source_metadata())
    experiment_config["model_name_or_path"] = model_name_or_path
    run_dir = (results_root / active_run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metadata.json").write_text(json.dumps({
        **experiment_config, "run_id": active_run_id, "status": "RUNNING",
    }, indent=2) + "\n", encoding="utf-8")
    tasks = select_instances(load_agent_tasks(selected_path), instance_id)
    bundles = {
        item.instance_id: item
        for item in select_instances(load_evaluation_bundles(selected_path), instance_id)
    }
    rollouts: list[RolloutResult] = []
    task_status: list[dict[str, Any]] = []
    for task in tasks:
        task_dir = run_dir / "tasks" / task.instance_id
        task_dir.mkdir(parents=True, exist_ok=True)
        calibration = calibrator(
            task,
            bundles[task.instance_id],
            task_dir / "calibration",
        )
        if not calibration.calibrated:
            for name in ("final_answer.txt", "model.patch", "events.jsonl"):
                (task_dir / name).touch(exist_ok=True)
            metadata = {
                "instance_id": task.instance_id,
                "status": "SKIPPED_CALIBRATION_FAILED",
                **experiment_config,
                "calibration": asdict(calibration),
            }
            (task_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            task_status.append(metadata)
            continue
        try:
            completed = rollout(
                task,
                provider,
                task_dir,
                model_name_or_path=model_name_or_path,
                max_turns=max_turns,
                subagent_max_turns=subagent_max_turns,
                max_context_tokens=max_context_tokens,
                working_context_trigger_tokens=working_context_trigger_tokens,
                working_context_target_tokens=working_context_target_tokens,
                keep_recent_tool_batches=keep_recent_tool_batches,
                **({"environment_adapter": environment_adapter}
                   if environment_adapter is not None else {}),
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            task_status.append(
                {
                    "instance_id": task.instance_id,
                    "status": "FAILED",
                    **experiment_config,
                    "error_type": type(error).__name__,
                }
            )
            continue
        rollouts.append(completed)
        completed_status = {
            "instance_id": task.instance_id,
            "status": "COMPLETED",
            **experiment_config,
            "calibration": asdict(calibration),
        }
        task_metadata_path = task_dir / "metadata.json"
        task_metadata = json.loads(task_metadata_path.read_text(encoding="utf-8"))
        task_metadata["calibration"] = asdict(calibration)
        for name, value in experiment_config.items():
            task_metadata.setdefault(name, value)
        task_metadata_path.write_text(
            json.dumps(task_metadata, indent=2) + "\n", encoding="utf-8"
        )
        task_status.append(completed_status)
    predictions_path = run_dir / "predictions.jsonl"
    write_predictions(predictions_path, rollouts, model_name_or_path)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": active_run_id,
                **experiment_config,
                "benchmark": "SWE-bench Lite Dev selected smoke set",
                "is_full_swe_bench_lite_score": False,
                "model_name_or_path": model_name_or_path,
                "max_context_tokens": max_context_tokens,
                "selected_tasks_path": str(selected_path.resolve()),
                "tasks": task_status,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir
