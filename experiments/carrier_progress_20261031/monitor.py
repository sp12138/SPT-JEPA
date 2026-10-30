"""Read-only 30-minute snapshots. No model/API calls, fixes or restarts."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def snapshot(roots, host, gpu=True):
    now, campaigns = time.time(), []
    for raw in roots:
        root = Path(raw)
        plan = read(root / "plan.json")
        if plan is None:
            campaigns.append(dict(root=str(root), state="awaiting_plan", complete=False))
            continue
        jobs = [j for j in plan["jobs"] if j["host"] == host]
        exits = {j["id"]: read(root / "exits" / (j["id"] + ".json")) for j in jobs}
        done = {k: v for k, v in exits.items() if v is not None}
        failed = {k: v for k, v in done.items() if v.get("returncode") != 0}
        worker = read(root / "workers" / (host + ".json"))
        progress, nonfinite = [], []
        for job in jobs:
            status = read(Path(job["output"]) / "status.json")
            if status and job["id"] not in done:
                progress.append(dict(id=job["id"], **{k: status.get(k) for k in
                    ["state", "step", "loss", "fm", "rate", "updated_at", "seconds_per_step"]}))
            if status and any(isinstance(status.get(k), (int, float)) and
                    not math.isfinite(status[k]) for k in ["loss", "fm", "rate"]):
                nonfinite.append(job["id"])
        terminal = len(done) == len(jobs)
        campaigns.append(dict(root=str(root), planned=len(jobs), completed=len(done),
            complete=terminal, failed=failed, nonfinite=nonfinite,
            running=(worker or {}).get("running", {}), progress=progress,
            held=(root / "HOLD_DISPATCH").exists(), worker_pid=(worker or {}).get("pid"),
            worker_stale=bool(jobs and not terminal and
                (worker is None or now - worker.get("updated_at", 0) > 180))))
    gpu_record = None
    if gpu:
        try:
            gpu_record = subprocess.check_output(["nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
                text=True, timeout=20).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            gpu_record = {"error": str(exc)}
    return dict(created_at=now, host=host, monitor_pid=os.getpid(), campaigns=campaigns,
        all_complete=all(c["complete"] for c in campaigns), gpus=gpu_record,
        needs_attention=any(c.get("failed") or c.get("worker_stale") or c.get("nonfinite") for c in campaigns),
        scope="Read-only collection; does not reason about results, fix or restart jobs.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--host", choices=["dev2", "dev13", "dev15"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=int, default=1800)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()
    if args.interval < 60:
        parser.error("Monitoring interval must be at least 60 seconds")
    out = Path(args.output)
    while True:
        result = snapshot(args.roots, args.host, not args.no_gpu)
        write(out / "latest.json", result)
        write(out / (str(int(result["created_at"])) + ".json"), result)
        print(json.dumps(dict(host=args.host, all_complete=result["all_complete"],
            needs_attention=result["needs_attention"], created_at=result["created_at"])), flush=True)
        if args.once or result["all_complete"] or (out / "STOP_MONITOR").exists():
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
