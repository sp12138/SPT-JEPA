"""Inventory every dispatch entry without treating process exit as scientific success."""
import argparse
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(campaign):
    plan = read(campaign / "plan.json")
    rows = []
    for job in plan["jobs"]:
        root = Path(job["output"])
        status_path = root / "status.json"
        status = read(status_path)
        if not status:
            status_path = root / "results/status.json"
            status = read(status_path)
        status = status or {}
        exit_path = campaign / ("exit-" + job["id"] + ".json")
        exit_record = read(exit_path) or {}
        early = read(root / "early_stop.json")
        config = read(root / "config.json") or {}
        child_results = status.get("results", [])
        failures = [
            {key: result.get(key) for key in ["id", "model", "step", "error", "returncode", "output"]}
            for result in child_results
            if result.get("error") or result.get("returncode", 0)
        ] if isinstance(child_results, list) else []
        checkpoints = sorted((root / "checkpoints").glob("*.dcp"))
        saved_steps = [int(p.stem) for p in checkpoints if p.stem.isdigit()]
        step = status.get("step")
        if step is None and saved_steps:
            step = max(saved_steps)
        rc = exit_record.get("returncode")
        held = job.get("hold", False)
        smoke = "smoke" in job["id"]
        max_steps = config.get("max_steps")
        below_budget = step is not None and max_steps is not None and step < max_steps
        if early and (rc not in [None, 0] or below_budget or max_steps is None):
            outcome = "early_stopped"
        elif rc not in [None, 0]:
            outcome = "failed_exit"
        elif failures:
            outcome = "failed_children"
        elif status.get("state") == "finished" or rc == 0:
            outcome = "smoke_completed" if smoke else "budget_limited" if below_budget else "completed"
        elif held:
            outcome = "held_historical" if status or (root / "execution.json").exists() else "held_unlaunched"
        else:
            outcome = "unaccounted"
        rows.append(dict(
            id=job["id"], host=job["host"], gpu=job["gpu"], output=str(root),
            held=held, outcome=outcome, recorded_state=status.get("state"),
            exit_code=rc, checkpoint_or_recorded_step=step, requested_max_steps=max_steps,
            child_failures=failures, early_stop=early,
            native_checkpoints=[str(p) for p in checkpoints],
            status_path=str(status_path) if status_path.exists() else None,
            status_sha256=digest(status_path) if status_path.exists() else None,
            exit_path=str(exit_path) if exit_path.exists() else None,
            skipped=read(root / "skipped.json"),
        ))
    return plan, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    args = parser.parse_args()
    plan, rows = inventory(args.campaign)
    verification = read(args.verification)
    if verification is None:
        raise ValueError("Provide the fresh endpoint verifier result")
    registries = {}
    for name in [
        "late_native_quality_registry.json", "native_context_registry.json",
        "native_feedback_registry.json", "color_geometry_registry.json",
        "long_budget_diagnostics_registry.json",
    ]:
        jobs = read(args.campaign / name) or []
        pending = []
        for job in jobs:
            status = read(Path(job["output"]) / "status.json") or {}
            results = status.get("results", [])
            if status.get("state") != "finished" or any(
                x.get("error") or x.get("returncode", 0) for x in results
            ):
                pending.append(job["id"])
        registries[name] = dict(jobs=len(jobs), pending=pending)
    files = Path(plan["root"]) / "code/experiments/insights12h/results"
    manifest = read(files / "artifact-manifest.json")
    bad_hashes = [
        entry["path"] for entry in manifest["files"]
        if not (files / entry["path"]).exists()
        or digest(files / entry["path"]) != entry["sha256"]
    ]
    result = dict(
        checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        campaign=str(args.campaign), dispatch_entries=len(rows),
        outcomes=dict(collections.Counter(row["outcome"] for row in rows)),
        recorded_exit_codes=dict(collections.Counter(
            "missing" if row["exit_code"] is None else str(row["exit_code"]) for row in rows
        )),
        failed_child_tasks=sum(len(row["child_failures"]) for row in rows),
        unaccounted=[row["id"] for row in rows if row["outcome"] == "unaccounted"],
        endpoint_verification={key: verification[key] for key in [
            "checked_at", "complete", "final_models_checked", "expected_final_models", "pending", "errors"
        ]},
        supplementary_registries=registries,
        packaged_file_hashes=dict(checked=len(manifest["files"]), mismatches=bad_hashes),
        jobs=rows,
        scope="Dispatch inventory only. Early stops, held entries, failed attempts and scientific completeness are distinct. Historical Running records do not establish current liveness.",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "task-inventory.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    columns = ["id", "host", "gpu", "held", "outcome", "recorded_state", "exit_code",
               "checkpoint_or_recorded_step", "requested_max_steps", "child_failure_count", "output"]
    with (args.output / "task-inventory.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: len(row["child_failures"]) if key == "child_failure_count" else row.get(key)
                             for key in columns})
    print(json.dumps({key: result[key] for key in [
        "dispatch_entries", "outcomes", "recorded_exit_codes", "failed_child_tasks", "unaccounted",
        "endpoint_verification", "supplementary_registries", "packaged_file_hashes"
    ]}, indent=2))
    if result["unaccounted"] or bad_hashes or not verification["complete"] or any(
        value["pending"] for value in registries.values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
