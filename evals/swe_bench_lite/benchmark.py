"""Single-configuration Test60 controller. No execution occurs on import.

Run via ``python -m evals.swe_bench_lite.benchmark --help``. Prepare/report are
offline; execution requires explicit flags. Frozen v1 data are never rewritten.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
from uuid import uuid4

from .data import load_agent_tasks, load_evaluation_bundles
from .freeze import AGENT_FIELDS, json_bytes

ROOT = Path(__file__).resolve().parents[2]
FROZEN = Path(__file__).with_name("frozen") / "test60-v1"
FROZEN_SUMS = "ed1383ec5ee43849c3de4b332266319d6e0102bf378ca373a0fd00fada5c9a87"
PROTOCOL = Path(__file__).with_name("test60-protocol.json")
TRANSIENT = {"connection", "rate_limit", "server_unavailable"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write_new(path, value, *, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()
                     if jsonl else json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def safe_file(root, relative):
    part = PurePosixPath(relative)
    if (part.is_absolute() or ".." in part.parts or "\\" in relative
            or ":" in relative or not part.parts):
        raise ValueError("Unsafe artifact path")
    path = root / relative
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError("Artifact symlink is not allowed")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact escapes its root")
    return path


def verify_bundle(bundle=FROZEN):
    bundle = Path(bundle).resolve()
    sums = safe_file(bundle, "SHA256SUMS").read_bytes()
    if sha(sums) != FROZEN_SUMS:
        raise ValueError("Frozen checksum inventory changed")
    names = set()
    for line in sums.decode().splitlines():
        expected, name = line.split("  ", 1)
        if name in names or sha(safe_file(bundle, name).read_bytes()) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
        names.add(name)
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual != names | {"SHA256SUMS"}:
        raise ValueError("Unexpected frozen artifacts")
    manifest = read(bundle / "manifest.json")
    tasks = load_agent_tasks(bundle / "agent/agent_tasks.jsonl")
    private = load_evaluation_bundles(bundle / "evaluator/selection.jsonl")
    rows = [json.loads(line) for line in (bundle / "agent/agent_tasks.jsonl").read_bytes().splitlines()]
    expected = manifest["selected_instance_ids"]
    if (manifest["split"] != "test" or len(expected) != 60 or len(set(expected)) != 60
            or [t.instance_id for t in tasks] != expected
            or [t.instance_id for t in private] != expected
            or any(set(row) != set(AGENT_FIELDS) for row in rows)):
        raise ValueError("Frozen input identities do not agree")
    return tasks, private, manifest


def source_files():
    files = [ROOT / "pyproject.toml"]
    files += list((ROOT / "tiny_harness").rglob("*.py"))
    files += list((ROOT / "tiny_harness/skills").rglob("SKILL.md"))
    files += list(Path(__file__).parent.glob("*.py"))
    files += [PROTOCOL]
    result = {}
    for path in sorted(files):
        if path.is_symlink():
            raise ValueError("Source symlink requires inspection")
        result[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    return result


def package_fingerprint(name):
    dist = importlib.metadata.distribution(name)
    files = {}
    for item in dist.files or ():
        if str(item).startswith(name + "/") and str(item).endswith(".py"):
            files[str(item)] = sha(dist.locate_file(item).read_bytes())
    return {"version": dist.version, "python_files_sha256": sha(json_bytes(files))}


class NoModel:
    def complete(self, *args, **kwargs):
        raise AssertionError("Offline preparation must not call a model")


def capability_fingerprint(workspace, config):
    from tiny_harness.agent.context import create_run_context
    from .pipeline import ContainerPermissionPolicy
    context = create_run_context(NoModel(), workspace, permission_policy=ContainerPermissionPolicy(),
                                 **config["budgets"])
    skills = {name: {"origin": item.source.origin.value,
                     "sha256": sha(item.path.read_bytes())}
              for name, item in context.skill_catalog.manifests.items()}
    return {"tools": context.tools, "skills": skills}


def current_fingerprint(config):
    from .pipeline import AGENT_SYSTEM_PROMPT
    from .observability import source_metadata
    with tempfile.TemporaryDirectory() as temporary:
        capabilities = capability_fingerprint(Path(temporary), config)
    return {
        "source": {name: sha(content) for name, content in source_files().items()},
        "packages": sorted([d.metadata["Name"], d.version] for d in importlib.metadata.distributions()),
        "grader": package_fingerprint("swebench"),
        "provider_adapter": package_fingerprint("openai"),
        "python": sys.version, "platform": platform.platform(),
        "prompt_sha256": sha(AGENT_SYSTEM_PROMPT.encode()),
        "capabilities": capabilities, "git": source_metadata(),
    }


def validate_config(config):
    if set(config) != {"model", "model_version", "base_url", "budgets"}:
        raise ValueError("Use the single-configuration model/budget schema")
    if not all(isinstance(config[k], str) and config[k].strip() for k in ("model", "model_version", "base_url")):
        raise ValueError("Explicit model/version/endpoint required")
    url = urlsplit(config["base_url"])
    if url.scheme not in {"https", "http"} or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("Endpoint must not contain credentials or query data")
    b = config["budgets"]
    if set(b) != {"max_turns", "subagent_max_turns", "max_context_tokens"}:
        raise ValueError("Unsupported budget keys")
    if any(type(v) is not int or v < 1 for v in b.values()):
        raise ValueError("Budgets must be positive integers")


def prepare(campaign, config, bundle=FROZEN):
    validate_config(config)
    verify_bundle(bundle)
    campaign = Path(campaign).resolve()
    if campaign.is_relative_to(Path(bundle).resolve()) or Path(bundle).resolve().is_relative_to(campaign):
        raise ValueError("Campaign and frozen data must be separate")
    fingerprint = current_fingerprint(config)
    campaign.mkdir(parents=True, exist_ok=False)
    for name, content in source_files().items():
        p = campaign / "source" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    write_new(campaign / "campaign.json", {
        "format_version": 1, "bundle": str(Path(bundle).resolve()),
        "frozen_sums_sha256": FROZEN_SUMS, "config": config,
        "protocol": read(PROTOCOL), "fingerprint": fingerprint,
        "sampling": "unchanged ChatCompletionsProvider defaults; no profile or tool override",
        "model_version_limit": "Operator-declared version; mutable provider aliases cannot be verified locally",
    })
    write_new(campaign / "campaign.sha256.json", {"sha256": sha((campaign / "campaign.json").read_bytes())})
    return campaign


def load_campaign(campaign, *, execution=False):
    campaign = Path(campaign).resolve()
    meta = read(safe_file(campaign, "campaign.json"))
    if sha((campaign / "campaign.json").read_bytes()) != read(campaign / "campaign.sha256.json")["sha256"]:
        raise ValueError("Campaign configuration changed")
    tasks, private, manifest = verify_bundle(Path(meta["bundle"]))
    validate_config(meta["config"])
    if meta["protocol"] != read(PROTOCOL) or meta["frozen_sums_sha256"] != FROZEN_SUMS:
        raise ValueError("Execution protocol identity changed")
    for name, expected in meta["fingerprint"]["source"].items():
        if sha(safe_file(campaign / "source", name).read_bytes()) != expected:
            raise ValueError("Recorded source snapshot changed")
    if execution:
        now = current_fingerprint(meta["config"])
        # A dirty Git flag is evidence, not source identity. Contents are authoritative.
        if {k: v for k, v in now.items() if k != "git"} != {
            k: v for k, v in meta["fingerprint"].items() if k != "git"
        }:
            raise ValueError("Actual source/dependencies/prompt/tools/skills changed")
    return meta, {t.instance_id: t for t in tasks}, {t.instance_id: t for t in private}, manifest


def docker_json(*args):
    completed = subprocess.run(["docker", *args], check=True, capture_output=True,
                               text=True, timeout=30)
    return json.loads(completed.stdout)


def inspect_image(reference):
    item = docker_json("image", "inspect", reference)[0]
    image_id = item["Id"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("Invalid Docker image ID")
    return {"id": image_id, "repo_digests": sorted(item.get("RepoDigests") or []),
            "architecture": item["Architecture"], "os": item.get("Os"),
            "layers": item["RootFS"]["Layers"]}


@contextlib.contextmanager
def task_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "controller.lock"
    lock.mkdir()  # surviving lock after SIGKILL must be manually inspected
    write_new(lock / "owner.json", {"pid": os.getpid(), "host": platform.node()})
    try:
        yield
    finally:
        # Keep the receipt; only move our temporary coordination lock.
        destination = directory / "lock-history" / uuid4().hex
        destination.parent.mkdir(exist_ok=True)
        lock.rename(destination)


def attempts(directory):
    return sorted((directory / "attempts").glob("[0-9][0-9][0-9][0-9]"))


def attempt_result(attempt):
    result = read(attempt / "result.json") if (attempt / "result.json").exists() else {"status": "pending_or_not_run"}
    review = attempt / "interruption-review.json"
    if review.exists():
        value = read(review)
        if (result["status"] != "pending_or_not_run" or not value.get("operator")
                or not value.get("reason") or not value.get("confirmed_stopped")):
            raise ValueError("Invalid interruption review")
        result = {**result, "status": "infrastructure_failure", "review": value}
    return result


def infrastructure_attempt(campaign, attempt, meta):
    result = attempt_result(attempt)
    if result["status"] == "infrastructure_failure":
        return True
    path = attempt / "evaluation.json"
    if result["status"] == "completed" and path.exists():
        record = read(path)
        validate_evaluation(campaign, attempt, meta, record)
        return record["status"] == "infrastructure_failure"
    return False


def approve_retry(campaign, instance, operator, reason):
    meta, tasks, _, _ = load_campaign(campaign)
    if instance not in tasks or not operator.strip() or not reason.strip():
        raise ValueError("Known task, operator and infrastructure evidence required")
    directory = Path(campaign) / "tasks" / instance
    with task_lock(directory):
        prior = attempts(directory)
        if not prior or not infrastructure_attempt(Path(campaign), prior[-1], meta):
            raise ValueError("Only an established infrastructure failure allows a new attempt")
        write_new(directory / "approvals" / f"{len(prior)+1:04}.json", {
            "previous_attempt": prior[-1].name, "operator": operator, "reason": reason,
            "policy": "Next attempt replaces this infrastructure failure only; never select by score",
        })


def recover(campaign, instance, operator, reason, *, confirm_stopped=False):
    _, tasks, _, _ = load_campaign(campaign)
    if instance not in tasks or not operator.strip() or not reason.strip() or not confirm_stopped:
        raise ValueError("Manual process-stop confirmation and evidence required")
    directory = Path(campaign) / "tasks" / instance
    lock = directory / "controller.lock"
    if lock.exists():
        target = directory / "lock-history" / ("recovered-" + uuid4().hex)
        target.parent.mkdir(exist_ok=True)
        lock.rename(target)
    with task_lock(directory):
        prior = attempts(directory)
        if not prior or attempt_result(prior[-1])["status"] != "pending_or_not_run":
            raise ValueError("No unfinished attempt to review")
        write_new(prior[-1] / "interruption-review.json", {
            "operator": operator, "reason": reason, "confirmed_stopped": True,
        })
    # Recovery is classification only. A separate approval is needed for a new rollout.


def run_attempt(campaign, instance, provider_factory, *, rollout=None, calibrator=None,
                image_inspector=inspect_image):
    from .pipeline import rollout_task
    from .calibration import calibrate_task
    from .docker_workspace import DockerTaskEnvironment
    from tiny_harness.models.base import ModelProviderError
    import tiny_harness.agent.context as context_module
    import tiny_harness.agent.loop as loop_module

    campaign = Path(campaign).resolve()
    meta, tasks, private, _ = load_campaign(campaign, execution=True)
    if instance not in tasks:
        raise ValueError("Task is not in the frozen selection")
    directory = campaign / "tasks" / instance
    with task_lock(directory):
        prior = attempts(directory)
        number = len(prior) + 1
        if prior:
            if attempt_result(prior[-1])["status"] == "pending_or_not_run":
                raise ValueError("Unfinished attempt; inspect, do not replay")
            approval = read(directory / "approvals" / f"{number:04}.json")
            if (approval["previous_attempt"] != prior[-1].name
                    or not infrastructure_attempt(campaign, prior[-1], meta)):
                raise ValueError("New attempt lacks infrastructure approval")
        task = tasks[instance]
        image = image_inspector(task.image)  # inspect only; missing images fail before receipt/API
        pin = directory / "image.json"
        if pin.exists() and read(pin) != image:
            raise ValueError("Pinned image changed")
        if not pin.exists():
            write_new(pin, image)
        task = replace(task, image=image["id"])
        attempt = directory / "attempts" / f"{number:04}"
        attempt.mkdir(parents=True, exist_ok=False)
        identity = {"instance_id": instance, "attempt": number,
                    "campaign_sha256": sha((campaign / "campaign.json").read_bytes()),
                    "image": image, "started_at": datetime.now(timezone.utc).isoformat()}
        write_new(attempt / "receipt.json", identity)
        started = time.monotonic()
        phase = "calibration"

        def environment(value, **kwargs):
            return DockerTaskEnvironment(value, require_local_image=True, **kwargs)

        def checked_agent(provider, workspace, messages, **kwargs):
            # Same assembly, tools, messages and loop as run_agent; assert identity only.
            context = context_module.create_run_context(provider, workspace, **kwargs)
            actual = {"tools": context.tools, "skills": {
                name: {"origin": item.source.origin.value, "sha256": sha(item.path.read_bytes())}
                for name, item in context.skill_catalog.manifests.items()}}
            if actual != meta["fingerprint"]["capabilities"]:
                raise ValueError("Actual task tool/skill set differs from frozen configuration")
            return loop_module.agent_loop(messages, context, messages[-1]["content"])

        try:
            calibrated = (calibrator or calibrate_task)(
                task, private[instance], attempt / "calibration", environment_factory=environment,
                official_script=True)
            write_new(attempt / "calibration.json", asdict(calibrated))
            if not calibrated.calibrated:
                result = {"status": "infrastructure_failure", "subtype": "calibration_failed"}
            else:
                phase = "rollout"
                provider = provider_factory(meta["config"])
                completed = (rollout or rollout_task)(
                    task, provider, attempt / "rollout", model_name_or_path=meta["config"]["model"],
                    **meta["config"]["budgets"], environment_factory=environment,
                    agent_entrypoint=checked_agent)
                # Persist each task before another task can start. Never aggregate-only writes.
                prediction = {"instance_id": instance, "model_name_or_path": meta["config"]["model"],
                              "model_patch": completed.model_patch}
                write_new(attempt / "predictions.jsonl", prediction, jsonl=True)
                result = {"status": "completed", "prediction_sha256": sha((attempt / "predictions.jsonl").read_bytes())}
        except BaseException as error:
            interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
            service = isinstance(error, ModelProviderError) and error.kind.value in TRANSIENT
            infrastructure = service or phase == "calibration" or isinstance(error, (OSError, subprocess.SubprocessError))
            result = {"status": "pending_or_not_run" if interrupted else (
                "infrastructure_failure" if infrastructure else "harness_or_protocol_failure"),
                "subtype": "interrupted" if interrupted else type(error).__name__,
                "phase": phase, "service_failure": service}
            result["duration_seconds"] = time.monotonic() - started
            write_new(attempt / "result.json", result)
            if interrupted:
                raise
            return result
        result["duration_seconds"] = time.monotonic() - started
        write_new(attempt / "result.json", result)
        return result


def evaluation_binding(campaign, attempt, meta):
    identity = read(attempt / "receipt.json")
    if identity["campaign_sha256"] != sha((campaign / "campaign.json").read_bytes()):
        raise ValueError("Attempt campaign identity changed")
    binding = {"prediction_sha256": sha((attempt / "predictions.jsonl").read_bytes()),
            "campaign_sha256": identity["campaign_sha256"], "image": identity["image"],
            "frozen_sums_sha256": FROZEN_SUMS, "grader": meta["fingerprint"]["grader"], "split": "test"}
    if (attempt / "grading/dataset.jsonl").exists():
        binding["grading_dataset_sha256"] = sha((attempt / "grading/dataset.jsonl").read_bytes())
    return binding


def validate_evaluation(campaign, attempt, meta, record):
    if (record["binding"] != evaluation_binding(campaign, attempt, meta)
            or record["binding"]["prediction_sha256"] != read(attempt / "result.json")["prediction_sha256"]):
        raise ValueError("Grading identity mismatch")
    for name, expected in record["reports"].items():
        if sha(safe_file(attempt, name).read_bytes()) != expected:
            raise ValueError("Official grading artifacts changed")


def grade_attempt(campaign, instance, number, *, evaluator=None, image_inspector=inspect_image):
    from .evaluator import run_official_evaluation
    campaign = Path(campaign).resolve()
    meta, tasks, _, _ = load_campaign(campaign, execution=True)
    if instance not in tasks or number < 1:
        raise ValueError("Unknown task/attempt")
    directory = campaign / "tasks" / instance
    with task_lock(directory):
        attempt = directory / "attempts" / f"{number:04}"
        result = read(attempt / "result.json")
        if result["status"] != "completed":
            raise ValueError("Only completed predictions can be graded")
        binding = evaluation_binding(campaign, attempt, meta)
        if binding["prediction_sha256"] != result["prediction_sha256"]:
            raise ValueError("Saved prediction changed")
        saved = attempt / "evaluation.json"
        if saved.exists():
            record = read(saved)
            validate_evaluation(campaign, attempt, meta, record)
            return record
        if image_inspector(binding["image"]["id"]) != binding["image"]:
            raise ValueError("Grading image identity changed")
        grade_dir = attempt / "grading"
        grade_dir.mkdir(exist_ok=False)  # interruption never causes implicit replay
        write_new(grade_dir / "receipt.json", binding)
        rows = [json.loads(l) for l in (Path(meta["bundle"]) / "evaluator/selection.jsonl").read_bytes().splitlines()]
        row = next(row for row in rows if row["instance_id"] == instance)
        row["image"] = binding["image"]["id"]
        write_new(grade_dir / "dataset.jsonl", row, jsonl=True)
        binding["grading_dataset_sha256"] = sha((grade_dir / "dataset.jsonl").read_bytes())
        try:
            official = (evaluator or run_official_evaluation)(
                attempt / "predictions.jsonl", grade_dir / "dataset.jsonl", grade_dir,
                instance_ids=(instance,), split="test", timeout_seconds=2400,
                local_images_only=True)
            if official.returncode != 0:
                raise ValueError("Official evaluator failed")
            from .benchmark_report import official_status
            status, raw = official_status(official.output_dir, instance)
            reports = {p.relative_to(attempt).as_posix(): sha(p.read_bytes())
                       for p in grade_dir.rglob("*") if p.is_file()}
            record = {"binding": binding, "status": status, "official": raw, "reports": reports}
        except Exception as error:
            record = {"binding": binding, "status": "official_evaluation_error",
                      "error_type": type(error).__name__, "reports": {
                          p.relative_to(attempt).as_posix(): sha(p.read_bytes())
                          for p in grade_dir.rglob("*") if p.is_file()}}
        write_new(saved, record)
        return record


def capacity(campaign, host_disk, instances, additional_bytes=0, minimum_free_bytes=20 * 1024**3):
    _, tasks, _, _ = load_campaign(campaign)
    if not instances or any(i not in tasks for i in instances) or additional_bytes < 0 or minimum_free_bytes < 0:
        raise ValueError("Explicit frozen task batch and nonnegative disk budgets required")
    disk = shutil.disk_usage(host_disk)
    images = {}
    for instance in instances:
        try:
            images[instance] = inspect_image(tasks[instance].image)
        except subprocess.CalledProcessError:
            images[instance] = None
    return {"host_disk": str(Path(host_disk).resolve()), "free_bytes": disk.free,
            "additional_bytes_estimate": additional_bytes, "minimum_free_bytes": minimum_free_bytes,
            "capacity_ok": disk.free >= additional_bytes + minimum_free_bytes,
            "all_images_local": all(images.values()), "images": images,
            "limitation": "Use the actual Windows VHDX backing drive (e.g. /mnt/d), not WSL virtual capacity. Missing image/layer costs are unknown; zero estimate only checks current reserve. No images are downloaded"}


def cleanup_plan(campaign, references):
    # Deliberately no execute/delete mode. Human-reviewed exact commands only.
    meta, tasks, _, _ = load_campaign(campaign)
    protected = set()
    for instance in ("sqlfluff__sqlfluff-1733", "sqlfluff__sqlfluff-1517", "pylint-dev__astroid-1196"):
        dev = next(t for t in load_agent_tasks(Path(__file__).with_name("dev.jsonl")) if t.instance_id == instance)
        protected.add(inspect_image(dev.image)["id"])
    for pin in (Path(campaign) / "tasks").glob("*/image.json"):
        history = attempts(pin.parent)
        releasable = False
        if history and (history[-1] / "evaluation.json").exists():
            record = read(history[-1] / "evaluation.json")
            validate_evaluation(Path(campaign).resolve(), history[-1], meta, record)
            releasable = record["status"] in {"resolved", "unresolved"}
        if not releasable:
            protected.add(read(pin)["id"])
    container_ids = subprocess.run(["docker", "ps", "-aq"], check=True, capture_output=True, text=True, timeout=30).stdout.split()
    if container_ids:
        protected.update(c["Image"] for c in docker_json("container", "inspect", *container_ids))
    images = [inspect_image(reference) for reference in references]
    return {"manual_only": True, "candidates": [{"image": image,
             "protected": image["id"] in protected,
             "command": None if image["id"] in protected else ["docker", "image", "rm", image["id"]]}
            for image in images], "estimated_reclaim_bytes": None,
            "limitation": "Shared compressed blobs/snapshot chains require reference accounting. Never sum virtual sizes; Windows VHDX may not shrink"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("campaign", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--model-version", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--subagent-max-turns", type=int, default=10)
    p.add_argument("--max-context-tokens", type=int, default=125000)
    for command in ("report", "run", "grade", "approve-retry", "recover", "capacity", "cleanup-plan"):
        p = sub.add_parser(command)
        p.add_argument("campaign", type=Path)
        if command in {"run", "grade", "approve-retry", "recover"}:
            p.add_argument("--instance-id", required=True)
        if command in {"run", "grade"}:
            p.add_argument("--execute", action="store_true")
        if command == "grade":
            p.add_argument("--attempt", required=True, type=int)
        if command in {"approve-retry", "recover"}:
            p.add_argument("--operator", required=True)
            p.add_argument("--reason", required=True)
        if command == "recover":
            p.add_argument("--confirm-process-stopped", action="store_true")
        if command == "capacity":
            p.add_argument("--host-disk", required=True, type=Path)
            p.add_argument("--instance-id", action="append", required=True)
            p.add_argument("--additional-bytes", type=int, required=True)
            p.add_argument("--minimum-free-bytes", type=int, default=20 * 1024**3)
        if command == "cleanup-plan":
            p.add_argument("--image", action="append", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.campaign, {"model": args.model, "model_version": args.model_version,
                "base_url": args.base_url, "budgets": {k: getattr(args, k) for k in
                ("max_turns", "subagent_max_turns", "max_context_tokens")}})
        result = {"campaign": str(args.campaign), "status": "prepared"}
    elif args.command == "report":
        from .benchmark_report import report
        result = report(args.campaign)
    elif args.command in {"run", "grade"}:
        if not args.execute:
            parser.error("Explicit --execute required; run may incur API fees and grade starts Docker")
        if args.command == "grade":
            result = grade_attempt(args.campaign, args.instance_id, args.attempt)
        else:
            def provider(config):
                from tiny_harness.models.chat_completions import ChatCompletionsProvider
                return ChatCompletionsProvider(api_key=os.environ["TINYHARNESS_API_KEY"],
                    model=config["model"], base_url=config["base_url"])
            result = run_attempt(args.campaign, args.instance_id, provider)
    elif args.command == "approve-retry":
        approve_retry(args.campaign, args.instance_id, args.operator, args.reason)
        result = {"status": "approved"}
    elif args.command == "recover":
        recover(args.campaign, args.instance_id, args.operator, args.reason,
                confirm_stopped=args.confirm_process_stopped)
        result = {"status": "reviewed; no rollout started"}
    elif args.command == "capacity":
        result = capacity(args.campaign, args.host_disk, args.instance_id, args.additional_bytes, args.minimum_free_bytes)
    else:
        result = cleanup_plan(args.campaign, args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
