"""Pinned Verified selection and model-free environment qualification.

This entry point deliberately has no rollout/provider construction path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

from .calibration import calibrate_task, official_log_grader
from .data import _records, load_agent_tasks, load_evaluation_bundles
from .docker_workspace import DockerTaskEnvironment
from .evaluator import run_official_evaluation

DATASET = "SWE-bench/SWE-bench_Verified"
REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"
SOURCE_FILE = "data/test-00000-of-00001.parquet"
SOURCE_SHA256 = "030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25"
SEED = 20261014
ROOT = Path(__file__).parent
DEFAULT_SELECTION = ROOT / "verified" / "selected_tasks.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def select_rows(rows, excluded, *, count=30, seed=SEED):
    ids = [row["instance_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source instance IDs")
    eligible = [row for row in rows if row["instance_id"] not in excluded]
    if not 0 < count <= len(eligible):
        raise ValueError("Invalid sample size")
    # Hash ranking is independent of source row order and Python RNG versions.
    return sorted(eligible, key=lambda row: (
        hashlib.sha256(f"{seed}\n{row['instance_id']}".encode()).hexdigest(),
        row["instance_id"],
    ))[:count]


def prepare(source: Path, output: Path, exclusions: list[Path]) -> None:
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("Source does not match the pinned official dataset")
    import pyarrow.parquet as parquet
    rows = parquet.read_table(source).to_pylist()
    excluded = {row["instance_id"] for path in exclusions for row in _records(path)}
    selected = select_rows(rows, excluded)
    output.mkdir(parents=True, exist_ok=False)
    selection = output / "selected_tasks.jsonl"
    # Official 5.0.2 loads local JSONL using the platform's default encoding.
    # ASCII JSON escapes preserve Unicode text on Windows without changing it.
    selection.write_text("".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in selected), encoding="utf-8", newline="\n")
    # Validate both sides of the leak-resistant boundary before publishing.
    load_agent_tasks(selection)
    load_evaluation_bundles(selection)
    write_json(output / "manifest.json", {
        "dataset": DATASET, "revision": REVISION, "split": "test",
        "source_file": SOURCE_FILE, "source_sha256": sha256(source),
        "source_count": len(rows), "seed": SEED,
        "sampling_rule": "ascending SHA-256(UTF-8(str(seed) + LF + instance_id)), ID tie-break; first 30 after exclusions",
        "exclusion_hash_encoding": "file bytes with CRLF normalized to LF",
        "exclusions": [{"file": path.name, "sha256": text_sha256(path)} for path in exclusions],
        "excluded_instance_ids": sorted(excluded),
        "excluded_source_count": sum(row["instance_id"] in excluded for row in rows),
        "selection_sha256": sha256(selection),
        "instance_ids": [row["instance_id"] for row in selected],
        "smoke_instance_ids": [row["instance_id"] for row in selected[:3]],
        "benchmark_claim": "30-task preparation subset, not a full Verified benchmark score",
    })


def grade_gold(task, bundle, selection, directory):
    """Exercise the official CLI too; exit zero alone is not a verdict."""
    predictions = directory / "gold_predictions.jsonl"
    predictions.write_text(json.dumps({"instance_id": task.instance_id,
        "model_name_or_path": "reference-gold", "model_patch": bundle.patch}) + "\n", encoding="utf-8")
    evaluation = run_official_evaluation(predictions, selection, directory,
        instance_ids=(task.instance_id,), timeout_seconds=1800)
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in evaluation.output_dir.rglob("report.json")]
    verdicts = [r[task.instance_id].get("resolved") for r in reports if task.instance_id in r]
    return "PASSED" if verdicts and all(v is True for v in verdicts) else "FAILED"


def qualify(selection: Path, output: Path, *, smoke: bool = True):
    """Keep every selected row, including unattempted and failed samples."""
    manifest = json.loads(selection.with_name("manifest.json").read_text(encoding="utf-8"))
    if sha256(selection) != manifest["selection_sha256"]:
        raise ValueError("Selection checksum mismatch")
    tasks = load_agent_tasks(selection)
    bundles = {b.instance_id: b for b in load_evaluation_bundles(selection)}
    if [t.instance_id for t in tasks] != manifest["instance_ids"]:
        raise ValueError("Selection IDs differ from manifest")
    if len(tasks) != 30 or len(bundles) != 30 or manifest["smoke_instance_ids"] != manifest["instance_ids"][:3]:
        raise ValueError("Expected 30 distinct tasks and the first 3 as smoke tasks")
    output.mkdir(parents=True, exist_ok=False)
    results = [{"instance_id": t.instance_id, "status": "NOT_RUN", "official_cli": "NOT_RUN"} for t in tasks]
    summary = {"selection_sha256": sha256(selection), "model_api_calls": 0,
               "swebench_version": version("swebench"),
               "mode": "smoke" if smoke else "all", "results": results}
    write_json(output / "summary.json", summary)
    attempted = set(manifest["smoke_instance_ids"] if smoke else manifest["instance_ids"])
    for task, result in zip(tasks, results):
        if task.instance_id not in attempted:
            continue
        directory = output / task.instance_id
        directory.mkdir()
        result["status"] = "ENVIRONMENT_FAILED"
        try:
            subprocess.run(["docker", "info"], capture_output=True, text=True,
                           timeout=30, check=True)
            # Separate availability from baseline/gold reproduction failures.
            with DockerTaskEnvironment(task, command_timeout_seconds=1200) as env:
                env.exec("python --version && git rev-parse HEAD", check=True)
            inspected = subprocess.run(
                ["docker", "image", "inspect", task.image, "--format", "{{json .RepoDigests}}"],
                capture_output=True, text=True, timeout=30, check=True,
            )
            result["image_digests"] = json.loads(inspected.stdout)
            result["environment"] = "READY"
            result["status"] = "CALIBRATION_FAILED"
            reports = []
            def grader(task, bundle, log_path, label):
                report = dict(official_log_grader(task, bundle, log_path, label))
                write_json(directory / f"{label}-report.json", report)
                reports.append(report)
                return report

            calibration = calibrate_task(task, bundles[task.instance_id], directory / "calibration", grader=grader)
            result["calibration"] = asdict(calibration)
            result["status"] = "RUNNABLE" if calibration.calibrated else "CALIBRATION_FAILED"
            if any(report.get("infra_failure") for report in reports):
                result["status"] = "ENVIRONMENT_FAILED"
            if smoke and calibration.calibrated:
                result["official_cli"] = "FAILED"
                result["official_cli"] = grade_gold(task, bundles[task.instance_id], selection, directory)
        except Exception as error:
            result["error_type"] = type(error).__name__
            # Keep diagnostics locally; no credentials or model environment read.
            (directory / "error.txt").write_text(str(error), encoding="utf-8")
            if isinstance(error, subprocess.CalledProcessError):
                (directory / "process.log").write_text(
                    (error.stdout or "") + (error.stderr or ""), encoding="utf-8")
        write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("source", type=Path)
    prepare_parser.add_argument("output", type=Path)
    prepare_parser.add_argument("--exclude", type=Path, action="append", default=[])
    calibrate_parser = sub.add_parser("calibrate")
    calibrate_parser.add_argument("output", type=Path)
    calibrate_parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    calibrate_parser.add_argument("--all", action="store_true", help="Explicitly qualify all 30; default is first 3")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.source, args.output, [ROOT / "dev.jsonl", ROOT / "selected_tasks.jsonl", *args.exclude])
        return 0
    summary = qualify(args.selection, args.output, smoke=not args.all)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(any(r["status"] not in {"NOT_RUN", "RUNNABLE"} or r["official_cli"] == "FAILED" for r in summary["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
