"""Compact per-hour results and the latest complete equal-time comparison."""
import fcntl
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
    # Four native workers may finish concurrently. Serialize shared aggregation.
    with (root/"results/.collect.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        names = ("robojepa-seed1", "spt-seed1", "spt64-seed1", "spt-grid16-seed1")
        found = [{p.stem for p in (root/"results"/name).glob("hour-*.json")} for name in names]
        common = set.intersection(*found)
        latest = max(common, key=lambda x:int(x.split("-")[1])) if common else None
        comparison = None
        if latest:
            comparison = [json.loads((root/"results"/name/(latest+".json")).read_text()) for name in names]
            assert len({tuple(r["case_sha256"]) for r in comparison}) == 1
        save(root/"results/latest.json", dict(updated_unix=time.time(), latest_common_hour=latest,
            primary_comparison="Four different methods, allseed1; no seed robustness claim", rows=comparison,
            latest_per_run={name:max(labels, key=lambda x:int(x.split("-")[1])) if labels else None
                            for name,labels in zip(names,found)}))
