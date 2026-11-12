"""Compact per-hour results and the latest complete equal-time comparison."""
import fcntl
import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.robocasa_compare.acquire import save
from experiments.robocasa_compare.evaluate import case_name


def collect(root, run, label):
    plan = json.loads((root/"plan.json").read_text())
    ready = json.loads((run/label/"ready.json").read_text())
    rows = [json.loads((root/"evaluation"/run.name/label/case_name(**case)/"summary.json").read_text())
            for case in plan["cases"]]
    assert len(rows) == 8 and all(r["passed"] and r["model_unchanged"] for r in rows)
    summary = dict(passed=True, run=run.name, label=label, cases=8, step=ready["step"],
        active_seconds=ready["active_seconds"],
        mean_return=float(np.mean([r["score"] for r in rows])),
        final_distance_mm=float(np.mean([r["final_goal_distance_m"]*1000 for r in rows])),
        final_successes=sum(r["final_task_complete"] for r in rows),
        any_successes=sum(r["any_task_complete"] for r in rows),
        case_sha256=[r["case_sha256"] for r in rows],
        per_case=[dict(kitchen=r["kitchen"], seed=r["seed"], score=r["score"],
            final_distance_mm=r["final_goal_distance_m"]*1000,
            final_success=r["final_task_complete"], trace_sha256=r["trace_sha256"]) for r in rows],
        collected_unix=time.time())
    save(root/"results"/run.name/(label+".json"), summary)
    aggregate(root)


def aggregate(root, filename="latest.json"):
    """Read completed endpoints; scope changes never require trainer restarts."""
    plan = json.loads((root/"plan.json").read_text())
    names = tuple(plan.get("active_runs", ("robojepa-seed1", "spt-seed1", "spt64-seed1")))
    assert names and len(set(names)) == len(names)
    assert Path(filename).name == filename and filename.endswith(".json")
    (root/"results").mkdir(exist_ok=True)
    # Native workers may finish concurrently. Serialize shared aggregation.
    with (root/"results/.collect.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        found = [{p.stem for p in (root/"results"/name).glob("hour-*.json")} for name in names]
        common = set.intersection(*found)
        latest = max(common, key=lambda x:int(x.split("-")[1])) if common else None
        comparison = None
        if latest:
            comparison = [json.loads((root/"results"/name/(latest+".json")).read_text()) for name in names]
            assert len({tuple(r["case_sha256"]) for r in comparison}) == 1
        result = dict(updated_unix=time.time(), latest_common_hour=latest, active_runs=list(names),
            primary_comparison="Active different methods, allseed1; no seed robustness claim", rows=comparison,
            latest_per_run={name:max(labels, key=lambda x:int(x.split("-")[1])) if labels else None
                            for name,labels in zip(names,found)})
        save(root/"results"/filename, result)
        return result


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", default="active-latest.json")
    a=p.parse_args();result=aggregate(a.root,a.output)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}))
