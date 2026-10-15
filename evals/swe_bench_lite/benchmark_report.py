"""Read-only fixed-denominator reporting for the single-configuration Test60."""
from collections import Counter
import json
from pathlib import Path

from .benchmark import (attempts, load_campaign, read, sha,
                        TRANSIENT, attempt_result, infrastructure_attempt, validate_evaluation)


def official_status(directory, instance):
    verdicts = set()
    raw = []
    infra = error = empty = False
    for path in Path(directory).rglob("*.json"):
        try:
            value = read(path)
        except (ValueError, OSError):
            continue
        if not isinstance(value, dict):
            continue
        relevant = {key: value[key] for key in (
            "resolved_ids", "unresolved_ids", "error_ids", "infra_failure_ids", "empty_patch_ids"
        ) if key in value}
        item = value.get(instance)
        if isinstance(item, dict) and type(item.get("resolved")) is bool:
            relevant[instance] = item
            verdicts.add("resolved" if item["resolved"] else "unresolved")
            infra |= item.get("infra_failure") is True
        if relevant:
            raw.append({"file": path.relative_to(directory).as_posix(), "report": relevant})
        if instance in value.get("resolved_ids", []):
            verdicts.add("resolved")
        if instance in value.get("unresolved_ids", []):
            verdicts.add("unresolved")
        infra |= instance in value.get("infra_failure_ids", [])
        error |= instance in value.get("error_ids", [])
        empty |= instance in value.get("empty_patch_ids", [])
    if "resolved" in verdicts and (len(verdicts) != 1 or error or infra or empty):
        return "official_evaluation_error", raw
    if infra:
        return "infrastructure_failure", raw
    if error:
        return "official_evaluation_error", raw
    if empty and not verdicts:
        return "unresolved", raw
    return (next(iter(verdicts)) if len(verdicts) == 1 else "official_evaluation_error"), raw


def attempt_metrics(attempt):
    from .report import analyze_run
    events_path = attempt / "rollout/events.jsonl"
    if not events_path.exists():
        return {"physical_requests": 0, "service_failed_requests": 0,
                "service_failure_rate": None, "usage_complete": False,
                "first_mutation_state": "unknown", "tool_calls": None}
    try:
        metrics = analyze_run(attempt / "rollout")
        events = [json.loads(line) for line in events_path.read_bytes().splitlines() if line.strip()]
    except (ValueError, KeyError, TypeError):
        return {"metrics_error": True, "usage_complete": False, "first_mutation_state": "unknown"}
    requests = sum(e["event_type"] == "model_requested" for e in events)
    service = sum(e["event_type"] == "model_request_failed" and e["data"].get("error_kind") in TRANSIENT for e in events)
    complete = metrics["mutation_observability"] == "complete" or (
        not any(e["event_type"] == "tool_started" for e in events)
        and any(e["event_type"] == "run_finished" for e in events))
    state = "unknown" if not complete else ("observed" if metrics["first_workspace_mutation_turn"] is not None else "never_mutated")
    metrics.update(physical_requests=requests, service_failed_requests=service,
                   service_failure_rate=service / requests if requests else None,
                   first_mutation_state=state,
                   usage_complete=requests == metrics["model_responses"] and all(
                       metrics[k] is not None for k in ("prompt_tokens", "completion_tokens")))
    return metrics


def report(campaign):
    campaign = Path(campaign).resolve()
    meta, tasks, _, _ = load_campaign(campaign)
    rows, ledger = [], []
    for instance, task in tasks.items():
        directory = campaign / "tasks" / instance
        history = attempts(directory)
        row = {"instance_id": instance, "repo": task.repo, "status": "pending_or_not_run", "attempt": None}
        for index, attempt in enumerate(history):
            receipt = read(attempt / "receipt.json")
            if receipt["instance_id"] != instance or receipt["attempt"] != index + 1:
                raise ValueError("Attempt receipt identity mismatch")
            if index:
                approval = read(directory / "approvals" / f"{index+1:04}.json")
                if (approval["previous_attempt"] != history[index-1].name
                        or not infrastructure_attempt(campaign, history[index-1], meta)):
                    raise ValueError("Attempt chain is not an approved infrastructure retry")
            if receipt["campaign_sha256"] != sha((campaign / "campaign.json").read_bytes()):
                raise ValueError("Attempt campaign identity mismatch")
            result = attempt_result(attempt)
            status = result["status"]
            metrics = attempt_metrics(attempt)
            item = {"instance_id": instance, "attempt": index + 1, "rollout_status": status,
                    "duration_seconds": result.get("duration_seconds"), "metrics": metrics}
            if status == "completed":
                status = "pending_or_not_run"
                evaluation = attempt / "evaluation.json"
                if evaluation.exists():
                    value = read(evaluation)
                    validate_evaluation(campaign, attempt, meta, value)
                    status = value["status"]
                    item["official"] = value.get("official")
            item["status"] = status
            if status not in {"resolved", "unresolved", "infrastructure_failure", "official_evaluation_error",
                              "harness_or_protocol_failure", "pending_or_not_run"}:
                raise ValueError("Unknown attempt status")
            ledger.append(item)
            row.update(status=status, attempt=index + 1, metrics=metrics,
                       duration_seconds=result.get("duration_seconds"))
        rows.append(row)
    counts = Counter(r["status"] for r in rows)
    repos = {}
    for row in rows:
        group = repos.setdefault(row["repo"], {"tasks": 0, "resolved": 0, "statuses": Counter()})
        group["tasks"] += 1
        group["resolved"] += row["status"] == "resolved"
        group["statuses"][row["status"]] += 1
    for value in repos.values():
        value["resolved_rate"] = value["resolved"] / value["tasks"]
    requests = sum(r["metrics"].get("physical_requests", 0) for r in ledger)
    service = sum(r["metrics"].get("service_failed_requests", 0) for r in ledger)
    totals = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "tool_calls"):
        values = [r["metrics"].get(field) for r in ledger]
        totals[field] = sum(values) if values and all(v is not None for v in values) else None
        totals[field + "_observed"] = sum(v for v in values if v is not None) if any(v is not None for v in values) else None
    if any(not r["metrics"].get("usage_complete") for r in ledger):
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            totals[field] = None
    durations = [r["duration_seconds"] for r in ledger]
    totals["duration_seconds"] = sum(durations) if durations and all(d is not None for d in durations) else None
    return {"denominator": 60, "resolved": counts["resolved"], "resolved_rate": counts["resolved"] / 60,
            "provisional": counts["pending_or_not_run"] > 0, "statuses": dict(counts),
            "per_repository": repos, "tasks": rows, "attempts": ledger, "totals_all_attempts": totals,
            "physical_requests": requests, "service_failed_requests": service,
            "service_failure_rate": service / requests if requests else None,
            "service_failure_rate_basis": "Observed model-request events only; incomplete logs may omit requests",
            "service_failure_attempts": sum(r["rollout_status"] == "infrastructure_failure" and
                attempt_result(campaign / "tasks" / r["instance_id"] / "attempts" / f"{r['attempt']:04}").get("service_failure", False)
                for r in ledger),
            "convention": "Latest approved infrastructure attempt is authoritative; never choose best verdict. Costs include ALL attempts; missing usage remains unknown"}
