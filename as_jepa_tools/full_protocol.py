"""Apply the prespecified full-cohort decisions; only emit durable queue gates.

Training/checkpoint execution remains with the frozen Lightning/Native entries.
This process never changes an existing run, config, checkpoint or source snapshot.
"""
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
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def choose(rows, planning=True, tolerance=.10, margin=10.):
    if not rows or any(not all(math.isfinite(row[x]) for x in ["score", "rate"]) for row in rows):
        raise ValueError("Complete finite selection evidence required")
    best = min(x["score"] for x in rows)
    tied = [x for x in rows if x["score"] <= best * (1 + tolerance)]
    if planning:
        if any(x.get("planning") is None for x in tied):
            return dict(state="waiting_for_tied_planning", tied=[x["id"] for x in tied])
        scores = [x["planning"] for x in tied]
        if max(scores) - min(scores) > margin:
            tied = [x for x in tied if x["planning"] >= max(scores) - margin]
    winner = min(tied, key=lambda x: (x["rate"], x["score"], x["beta"]))
    return dict(state="selected", winner=winner, tied=[x["id"] for x in tied],
                best_generation=best, generation_tolerance=tolerance, planning_margin=margin)


def phase_decision(trials, planning):
    if not trials or any(x.get("stage3") is None for x in trials):
        return dict(state="waiting_for_stage3")
    survivors = [x for x in trials if not x["stage3"]["collapsed"]]
    if not survivors:
        return dict(state="all_collapsed")
    if any(x.get("final") is None or x.get("score8") is None or x.get("score10") is None for x in survivors):
        return dict(state="waiting_for_full_budget")
    rows = []
    for x in survivors:
        if x["final"]["collapsed"]:
            continue
        score = .5 * (x["score10"]["h7_agent_ratio"] + x["score10"]["h7_block_ratio"])
        score8 = .5 * (x["score8"]["h7_agent_ratio"] + x["score8"]["h7_block_ratio"])
        rows.append(dict(id=x["id"], beta=x["beta"], score=score, score8=score8,
                         rate=x["final"]["rate_nats_per_frame"], planning=x.get("planning"),
                         active_dims=x["final"]["active_dims"], state_dim=x["final"]["state_dim"],
                         continuation_fraction=(score8-score)/max(abs(score8),1e-12)))
    if not rows:
        return dict(state="all_collapsed_at_final_budget")
    result = choose(rows, planning)
    result["rows"] = rows
    if result["state"] == "selected":
        winner = result["winner"]
        low, high = min(x["beta"] for x in trials), max(x["beta"] for x in trials)
        extension = winner["beta"]/4 if winner["beta"] == low else winner["beta"]*4 if winner["beta"] == high else None
        if extension is not None:
            result.update(state="edge_extension_required", requested_beta=extension)
    return result


def completed_json(root, job, relative):
    exit_record = read(root / "exits" / (job + ".json"))
    if not exit_record or exit_record["returncode"] != 0:
        return None
    return read(root / job / "results" / relative)


def update(registry):
    registry = read(registry)
    root = Path(registry["root"])
    selection = root / "selection"
    summary = dict(updated_at=time.time(), complete=False, families={})
    for family in ["L2", "S2"]:
        if (selection / (family + "-final-selection.json")).exists():
            summary["families"][family] = read(selection / (family + "-final-selection.json"))
            continue
        for phase in ["default", "ramp"]:
            if phase == "ramp" and not (selection / (family + "-default-all-collapsed.json")).exists():
                continue
            active = [x for x in registry["trials"] if x["family"] == family and x["phase"] == phase
                      and (not x.get("activation") or Path(x["activation"]).exists())]
            evidence = []
            for trial in active:
                prefix = trial["full_job"]
                row = dict(trial)
                row["stage3"] = completed_json(root, trial["stage3_audit"], "metrics.json")
                row["final"] = completed_json(root, trial["final_audit"], "metrics.json")
                row["score8"] = completed_json(root, trial["score8"], "metrics.json")
                row["score10"] = completed_json(root, trial["score10"], "metrics.json")
                if trial.get("planning"):
                    result = completed_json(root, trial["planning"], "metrics.json")
                    row["planning"] = result["metrics"]["success_rate"] if result else None
                evidence.append(row)
            passed_stage3=[x["id"] for x in evidence if x["stage3"] and not x["stage3"]["collapsed"]]
            if passed_stage3:
                atomic(selection/(family+"-any-PASS.json"),dict(family=family,phase=phase,matching_audits=passed_stage3,checked_at=time.time(),scope="Three-epoch gate only; final selection pending."))
            decision = phase_decision(evidence, planning=family=="L2")
            decision.update(family=family, phase=phase, checked_at=time.time(),
                            trial_ids=[x["id"] for x in active],
                            rule="Generation <=1.10*best; L2tiedP0within10points; lowestheldoutrate; edgefactor4extension;20epochsif8to10improvement>.05.",
                            scope="Automatic application of the committed original protocol; no new method tuning or human gallery approval.")
            summary["families"][family] = decision
            atomic(selection / (family + "-" + phase + "-progress.json"), decision)
            if decision["state"] == "all_collapsed":
                atomic(selection / (family + "-" + phase + "-all-collapsed.json"), decision)
                if phase == "ramp":
                    atomic(selection / (family + "-failed-after-declared-fallback.json"), decision)
                continue
            if decision["state"] == "all_collapsed_at_final_budget":
                atomic(selection / (family + "-" + phase + "-all-failed-at-final-budget.json"), decision)
                break
            if decision["state"] == "edge_extension_required":
                beta = decision["requested_beta"]
                matches = [x for x in registry["trials"] if x["family"]==family and x["phase"]==phase and math.isclose(x["beta"],beta,rel_tol=1e-12)]
                if matches:
                    atomic(Path(matches[0]["activation"]), decision)
                else:
                    atomic(selection / (family + "-needs-additional-edge-config.json"), decision)
                break
            if decision["state"] == "selected":
                trial_id = decision["winner"]["id"]
                atomic(selection / (trial_id + "-SELECTED.json"), decision)
                if decision["winner"]["active_dims"] == decision["winner"]["state_dim"]:
                    atomic(selection / (trial_id + "-CAPACITY128.json"), decision)
                if decision["winner"]["continuation_fraction"] > .05:
                    atomic(selection / (trial_id + "-EXTEND20.json"), decision)
                atomic(selection / (family + "-final-selection.json"), decision)
            break
    atomic(selection / "full-protocol-status.json", summary)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--registry",required=True)
    parser.add_argument("--watch",action="store_true")
    args=parser.parse_args()
    lock_path=Path(args.registry).parent/"workers/full-protocol.lock"
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            update(Path(args.registry))
            if not args.watch:break
            time.sleep(30)


if __name__=="__main__":
    main()
