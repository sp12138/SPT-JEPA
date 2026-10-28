"""Read-only campaign telemetry; never launches or stops experiments."""
import argparse, datetime, json, math, time
from pathlib import Path


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def snapshot(root):
    plan = read(root / "plan.json")
    now = time.time()
    result = {"at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "training": {}, "planning": {}, "endpoints": {}, "failures": [], "worker_ages_seconds": {}}
    for host in ["dev2", "dev13", "dev15"]:
        value = read(root / "workers" / (host + ".json"))
        result["worker_ages_seconds"][host] = None if not value else round(now - value["updated_at"], 1)
    finished = 0
    for job in plan["jobs"]:
        folder = Path(job["output"])
        exit_record = read(root / "exits" / (job["id"] + ".json"))
        if exit_record:
            finished += 1
            if exit_record["returncode"]:
                result["failures"].append({"id": job["id"], "exit": exit_record})
        if job["phase"] == "training":
            status = read(folder / "status.json")
            row = {"state": "not-started", "step": 0} if not status else {k: status.get(k) for k in ["state", "step", "loss", "fm", "rate", "feature_std", "active_dims", "seconds_per_step"]}
            row["status_age_seconds"] = None if not status else round(now - status["updated_at"], 1)
            row["exit"] = None if not exit_record else exit_record["returncode"]
            result["training"][str(job["training_seed"])] = row
            if status and any(not math.isfinite(float(status[k])) for k in ["loss", "fm", "rate"]):
                result["failures"].append({"id": job["id"], "reason": "Nonfinite numerical status"})
        elif job["phase"].startswith("planning"):
            value = read(folder / "metrics.json")
            result["planning"][job["id"]] = None if not value else value["metrics"]["success_rate"]
        else:
            value = read(folder / "results/status.json")
            result["endpoints"][job["id"]] = None if not value else {k: value.get(k) for k in ["state", "stage", "completed", "total", "error"]}
    result["finished_jobs"] = finished
    result["total_jobs"] = len(plan["jobs"])
    result["all_jobs_finished_successfully"] = finished == len(plan["jobs"]) and not result["failures"]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True)
    p.add_argument("--once", action="store_true")
    p.add_argument("--interval", type=int, default=30)
    a = p.parse_args()
    root = Path(a.campaign)
    while True:
        result = snapshot(root)
        target = root / "reports/progress-latest.json"
        tmp = target.with_suffix(".watch-tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(target)
        print(json.dumps(result), flush=True)
        if result["failures"]:
            raise SystemExit(2)
        if a.once or result["all_jobs_finished_successfully"]:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
