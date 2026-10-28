"""Durable scientific gate markers and recovery status; it never runs training."""
import argparse
import fcntl
import json
import math
import time
from pathlib import Path


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def update(plan_path):
    plan = read(plan_path)
    root = Path(plan["root"])
    gates = root / "gates"
    for mode in ["S0", "S1", "S2"]:
        name = "full-" + mode + "-smoke"
        job = next((j for j in plan["jobs"] if j["id"] == name), None)
        exit_record = read(root / "exits" / (name + ".json"))
        if not job or not exit_record or exit_record["returncode"] != 0:
            continue
        marker = gates / (name + "-PASS.json")
        if marker.exists():
            continue
        path = Path(job["output"]) / "train_steps.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        rows = [row for row in rows if "loss" in row and "step" in row]
        if not rows or int(rows[-1]["step"]) + 1 != 200:
            continue
        if any(not math.isfinite(row["loss"]) for row in rows):
            continue
        if any(row.get("perf", {}).get("global_batch") != 128 for row in rows):
            continue
        learning_rates = [row["learning_rate_after_step"] for row in rows]
        if not (0 < learning_rates[0] < max(learning_rates) and abs(max(learning_rates) - 1e-4) < 1e-10):
            continue
        if mode == "S2" and not any("grad_norm_encoder_post_clip" in row.get("metrics", {}) for row in rows):
            continue
        atomic(marker, dict(job=name, source=job["source"], final_step=200, finite_losses=True,
                            global_batch=128, learning_rates=learning_rates, checked_at=time.time(),
                            scope="Execution, budget and learning-rate gate; not a scientific success claim."))
    # These markers permit independent baselines once the corresponding AS-JEPA
    # full-data family has an actual noncollapsed three-epoch configuration.
    for mode in ["L2", "S2"]:
        matches = []
        for job in plan["jobs"]:
            if job["id"].startswith("collapse-full-" + mode + "-beta") and "-stage3-" in job["id"]:
                result = read(Path(job["output"]) / "results/metrics.json")
                exit_record = read(root / "exits" / (job["id"] + ".json"))
                if result and exit_record and exit_record["returncode"] == 0 and not result["collapsed"]:
                    if result["cases"] == 256 and result["sampler_steps"] == 50:
                        matches.append(job["id"])
        if matches:
            atomic(root / "selection" / (mode + "-any-PASS.json"),
                   dict(family=mode, matching_audits=matches, checked_at=time.time(),
                        scope="Three-epoch noisy-posterior gate only; final model selection remains pending."))
    rows = []
    for job in plan["jobs"]:
        exit_record = read(root / "exits" / (job["id"] + ".json"))
        dispatch = read(Path(job["output"]) / "dispatch.json")
        if exit_record:
            state = "exit0" if exit_record["returncode"] == 0 else "failed"
        elif dispatch:
            state = "started_state_requires_worker_liveness"
        elif job.get("hold"):
            state = "held"
        elif not all(Path(x).exists() for x in job.get("requires", [])):
            state = "waiting_for_artifact_gate"
        else:
            state = "queued_or_waiting_for_dependency"
        rows.append(dict(id=job["id"], state=state, host=job["host"], gpu=job["gpu"]))
    workers = {path.stem: read(path) for path in (root / "workers").glob("*.json")}
    summary = dict(updated_at=time.time(), campaign=str(root), complete=False,
                   authorization="Finish original experiments and ten candidates; dev2/dev13/dev15; no time limit",
                   jobs=rows, workers=workers,
                   scope="Recovery inventory, not a completed-research certificate. Exit0 is not itself scientific success.")
    atomic(root / "campaign-status.json", summary)
    atomic(root.parents[1] / "reports/progress_latest.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    lock = None
    if args.watch:
        lock_path = Path(args.plan).parent / "workers/campaign-state.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        update(Path(args.plan))
        if not args.watch:
            break
        time.sleep(15)


if __name__ == "__main__":
    main()
