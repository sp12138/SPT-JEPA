"""Compact read-only status for matched robustness methods; no frequent automatic polling."""
import argparse
import datetime
import json
import math
import statistics
import time
from pathlib import Path


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True)
    a = p.parse_args()
    root = Path(a.campaign)
    plan = read(root / "plan.json")
    now = time.time()
    report = dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  training={}, completed_planning={}, endpoints={}, failures=[], warnings=[], workers={}, finished_jobs=0, total_jobs=len(plan["jobs"]))
    for host in ["dev2", "dev13", "dev15"]:
        worker = read(root / "workers" / (host + ".json"))
        report["workers"][host] = None if worker is None else dict(age_seconds=round(now - worker["updated_at"], 1), running_jobs=len(worker["running"]), pid=worker["pid"])
        if worker is None or now - worker["updated_at"] > 300:
            report["warnings"].append("Dispatcher heartbeat stale: " + host)
    planning = {}
    for job in plan["jobs"]:
        folder = Path(job["output"])
        exit_record = read(root / "exits" / (job["id"] + ".json"))
        if exit_record:
            report["finished_jobs"] += 1
            if exit_record["returncode"]:
                report["failures"].append(dict(id=job["id"], exit=exit_record))
        if job["phase"] == "training":
            value = read(folder / "status.json")
            row = dict(state="not-started", step=0) if value is None else {k: value.get(k) for k in ["state", "step", "loss", "fm", "rate", "feature_std", "active_dims", "seconds_per_step", "peak_gib"]}
            row["status_age_seconds"] = None if value is None else round(now - value["updated_at"], 1)
            row["exit"] = None if exit_record is None else exit_record["returncode"]
            report["training"][job["id"]] = row
            if value and any(not math.isfinite(float(value[k])) for k in ["loss", "fm", "rate"]):
                report["failures"].append(dict(id=job["id"], reason="Nonfinite numerical status"))
            if value and value["state"] == "running" and now - value["updated_at"] > 900:
                report["warnings"].append("Training heartbeat stale: " + job["id"])
        elif job["phase"] == "planning":
            value = read(folder / "metrics.json")
            if value and exit_record and not exit_record["returncode"]:
                planning.setdefault(job["method"], {}).setdefault(str(job["training_seed"]), {})[str(job["evaluation_seed"])] = value["metrics"]["success_rate"]
        elif job["phase"] == "endpoint":
            value = read(folder / "results/status.json")
            if value:
                report["endpoints"][job["id"]] = {k: value.get(k) for k in ["state", "stage", "completed", "total", "error"]}
    report["completed_planning"] = planning
    report["all_jobs_finished_successfully"] = report["finished_jobs"] == len(plan["jobs"]) and not report["failures"]
    output = root / "reports/progress-latest.json"
    tmp = output.with_suffix(".watch-tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(output)
    print(json.dumps(report))
    if report["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
