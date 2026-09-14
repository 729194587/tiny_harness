"""Single-run summaries using the existing report and official evaluator."""

import hashlib
import json
from pathlib import Path

from .evaluator import run_official_evaluation
from .pipeline import DEFAULT_SELECTED_TASKS
from .report import analyze_run

ERROR = "Evaluation Error"


def _json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _path(value, base):
    path = Path(value.replace("\\", "/"))
    return (path if path.is_absolute() else base / path).resolve()


def locate_run(run_dir, dataset=None):
    predictions = run_dir / "predictions.jsonl"
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1 or not isinstance(rows[0].get("instance_id"), str) or not rows[0]["instance_id"]:
        raise ValueError("finalize requires exactly one prediction instance")
    instance = rows[0]["instance_id"]
    metadata = _json(run_dir / "metadata.json")
    tasks = metadata.get("tasks", [])
    if tasks and {task.get("instance_id") for task in tasks} != {instance}:
        raise ValueError("finalize requires a single-task run matching predictions")
    for path in (run_dir / "tasks").glob("*/metadata.json"):
        if _json(path).get("instance_id") not in (None, instance):
            raise ValueError("Task artifacts do not match predictions")
    source = (dataset.resolve() if dataset is not None else
              _path(metadata["selected_tasks_path"], run_dir)
              if metadata.get("selected_tasks_path") else DEFAULT_SELECTED_TASKS.resolve())
    return predictions, source, instance


def _verdict(output_dir, instance):
    conclusions = []
    for path in sorted(output_dir.rglob("*.json")):
        data = _json(path)
        if instance in (data.get("error_ids") or []):
            return ERROR, str(path)
        for key, status in (("resolved_ids", "Resolved"), ("unresolved_ids", "Unresolved")):
            if instance in (data.get(key) or []):
                conclusions.append((status, str(path)))
        item = data.get(instance)
        if path.name == "report.json" and isinstance(item, dict) and type(item.get("resolved")) is bool:
            conclusions.append(("Resolved" if item["resolved"] else "Unresolved", str(path)))
    if conclusions and len({status for status, _ in conclusions}) == 1:
        return conclusions[0]
    return ERROR, None


def _existing(run_dir, predictions, instance, fingerprint):
    for root in (run_dir / "official_evaluation", run_dir.parent / "official_evaluation"):
        for path in sorted(root.glob("*/metadata.json"), reverse=True):
            data = _json(path)
            if data.get("returncode") != 0 or not isinstance(data.get("predictions_path"), str):
                continue
            if _path(data["predictions_path"], path.parent) != predictions:
                continue
            recorded = data.get("finalizer_predictions_sha256")
            if recorded is not None and recorded != fingerprint:
                continue
            # Old evaluations have no digest; don't reuse after predictions changed.
            if recorded is None and predictions.stat().st_mtime_ns > path.stat().st_mtime_ns:
                continue
            status, report = _verdict(path.parent, instance)
            if status != ERROR:
                return path.parent, status, report
    return None


SUMMARY_FIELDS = (
    "instance_id", "official_status", "turns", "tool_calls", "first_workspace_mutation_turn",
    "prompt_tokens", "completion_tokens", "total_cache_hit_tokens", "total_cache_miss_tokens",
    "overall_cache_hit_rate", "checkpoint_count", "summary_cache_hit_rate",
)


def render_summary(summary):
    def display(value):
        return "未知" if value is None else str(value)
    lines = [f"- {key}: {display(summary[key])}" for key in SUMMARY_FIELDS]
    for item in summary["checkpoints"]:
        lines.append(f"- checkpoint turn {display(item['turn'])}: "
                     f"{display(item['before_tokens'])} → {display(item['after_tokens'])}")
    if summary["evaluation_error"]:
        lines.append(f"- 评测错误: {summary['evaluation_error']}")
    return "\n".join(lines) + "\n"


def finalize_run(run_dir: Path, *, dataset: Path | None = None):
    run_dir = run_dir.resolve()
    predictions, source, instance = locate_run(run_dir, dataset)
    metrics = analyze_run(run_dir)
    fingerprint = hashlib.sha256(predictions.read_bytes()).hexdigest()
    existing = _existing(run_dir, predictions, instance, fingerprint)
    output_dir = report = error = None
    status = ERROR
    if existing:
        output_dir, status, report = existing
    else:
        try:
            if not any(json.loads(line).get("instance_id") == instance
                       for line in source.read_text(encoding="utf-8").splitlines() if line.strip()):
                raise ValueError("Evaluation dataset does not contain the prediction instance")
            result = run_official_evaluation(predictions, source, run_dir, instance_ids=(instance,))
            output_dir = result.output_dir
            if result.returncode != 0:
                raise RuntimeError(f"Evaluator exited with code {result.returncode}")
            metadata_path = output_dir / "metadata.json"
            metadata = _json(metadata_path)
            metadata["finalizer_predictions_sha256"] = fingerprint
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            status, report = _verdict(output_dir, instance)
            if status == ERROR:
                error = "Official evaluation has no unambiguous successful grader conclusion"
        except Exception as exc:
            status = ERROR
            error = f"{type(exc).__name__}: {exc}"
    summary = {
        **metrics, "instance_id": instance, "official_status": status,
        "evaluation_reused": existing is not None,
        "evaluation_output_dir": str(output_dir) if output_dir else None,
        "evaluation_report": report, "evaluation_error": error,
        "predictions_path": str(predictions), "predictions_sha256": fingerprint,
        "dataset_path": str(source),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(render_summary(summary), encoding="utf-8")
    return summary
