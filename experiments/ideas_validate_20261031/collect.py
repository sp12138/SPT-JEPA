"""CPU-only collection of all frozen 24 short gates; never loads a model.

Default requires every planned endpoint to have completed successfully. Sources
remain in EFS. Small per-case outcomes and diagnostic summaries are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = [("carrier", "identity"), ("carrier", "unconstrained"), ("carrier", "bounded"),
        ("pathquery", "symbol"), ("pathquery", "completion"), ("pathquery", "transform")]
COMPARISONS = {"carrier": [("unconstrained", "identity"), ("bounded", "identity"), ("bounded", "unconstrained")],
               "pathquery": [("completion", "symbol"), ("transform", "symbol"), ("transform", "completion")]}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def paired_outcomes(candidate, reference, bootstrap=10000, seed=1042):
    a, b = np.asarray(candidate, dtype=float), np.asarray(reference, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("Paired outcomes require equal nonempty vectors")
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("Strict success must be binary")
    delta = a-b
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(len(delta), size=(bootstrap, len(delta)))].mean(1)*100
    return dict(n_cases=len(a), difference_pp=float(delta.mean()*100),
                gained=int(((a == 1) & (b == 0)).sum()), lost=int(((a == 0) & (b == 1)).sum()),
                both_success=int(((a == 1) & (b == 1)).sum()), both_failure=int(((a == 0) & (b == 0)).sum()),
                case_bootstrap_ci95_pp=np.quantile(draws, [.025, .975]).tolist(),
                per_case_difference=delta.astype(int).tolist())


def reduce_state_audit(state):
    """Preserve case errors; omit duplicated untrained-head per-case CE only."""
    state = dict(state)
    queries = state.get("path_queries")
    if queries:
        queries = dict(queries)
        modes = {queries["trained_mode"]}
        if "symbol" in modes:
            modes.add("symbol_future_only")
        queries["conditions"] = {
            condition: {head: {key: value for key, value in scores.items()
                               if key != "per_window_ce" or head in modes}
                        for head, scores in heads.items()}
            for condition, heads in queries["conditions"].items()}
        state["path_queries"] = queries
    return state


def probe_label_summaries(probes):
    """Dimension-explicit values; never assume all coordinate targets are 3-D."""
    result = {}
    # R² arrays are retained as the authoritative values. The known benchmark
    # metadata supplies task labels but not dimensions, so do not invent slices.
    for mode, scores in probes["scores"].items():
        result[mode] = dict(mean_r2=scores.get("mean_r2"),
                            r2_against_train_mean=scores.get("r2_against_train_mean"),
                            mse=scores.get("mse"),
                            informative_target_dimensions=scores.get("informative_target_dimensions"))
    return result


def validate_initial_identity(initial, records):
    if initial.get("passed") is not True:
        raise AssertionError("Initial-tensor review has not passed")
    runs = initial.get("runs", [])
    if len(runs) != len(records):
        raise AssertionError("Initial-tensor review must retain all 24 runs")
    if {row["id"] for row in runs} != {row["id"] for row in records}:
        raise AssertionError("Initial-tensor review IDs do not match the protocol")
    if not all(row.get("all_original_tensors_equal") is True for row in runs):
        raise AssertionError("An original native tensor differs at initialization")
    for row in runs:
        if "common_tensors" in row and row["common_tensors"] != 550:
            raise AssertionError("Unexpected original tensor count")
    pair_check = initial.get("paired_family_tensors_equal")
    if pair_check is not True and not isinstance(pair_check, list):
        raise AssertionError("Within-family initial tensor review must explicitly pass")
    # The root's review may provide detailed family entries instead of one bool.
    if isinstance(pair_check, list):
        if not pair_check:
            raise AssertionError("Missing family identity entries")
        for row in pair_check:
            if isinstance(row, bool) and not row:
                raise AssertionError("Failed family identity")
            if isinstance(row, dict) and any(row.get(key) is False for key in ["equal", "passed", "all_equal"]):
                raise AssertionError("Failed family identity")


def collect(root, bootstrap=10000):
    root = Path(root)
    plan, protocol = read(root/"plan.json"), read(root/"protocol.json")
    records = protocol["runs"]
    expected = {(task, seed, family, arm) for task in ["tworoom", "cube"]
                for seed in [1, 2] for family, arm in ARMS}
    actual = {(r["task"], r["seed"], r["family"], r["arm"]) for r in records}
    if len(records) != 24 or actual != expected:
        raise AssertionError("Protocol must contain the exact 24-arm/seed/task grid")
    if len(plan["jobs"]) != 120 or protocol["stop_step"] != 600:
        raise AssertionError("Unexpected endpoint/update budget")
    expected_jobs = {prefix + rec["id"] for rec in records
                     for prefix in ["", "health-", "info-", "transition-", "planning-"]}
    if {job["id"] for job in plan["jobs"]} != expected_jobs:
        raise AssertionError("Endpoint IDs do not match the exact declared grid")
    exits = []
    for job in plan["jobs"]:
        record = read(root/"exits"/(job["id"]+".json"))
        if record["returncode"] != 0:
            raise AssertionError("Endpoint failed: " + job["id"])
        if job["source"]["source_commit"] != protocol["source_commit"]:
            raise AssertionError("Mixed frozen source revisions")
        exits.append(dict(id=job["id"], phase=job.get("phase"), returncode=record["returncode"],
                          source_commit=job["source"]["source_commit"]))
    initial_path = root/"initial-tensor-identity.json"
    initial = read(initial_path)
    validate_initial_identity(initial, records)
    preflight_path = Path(protocol["preflight"])/"review.json"
    preflight = read(preflight_path)
    if preflight.get("passed") is not True:
        raise AssertionError("Preflight review must explicitly pass")
    rows, task_references = [], {}
    for rec in sorted(records, key=lambda r: (r["task"], ARMS.index((r["family"], r["arm"])), r["seed"])):
        name = rec["id"]
        paths = dict(config=root/name/"config.json", status=root/name/"status.json",
                     size=root/name/"model_size.json", health=root/("health-"+name)/"health.json",
                     info=root/("info-"+name)/"information-audit.json",
                     state=root/("transition-"+name)/"state-audit.json",
                     windows=root/("transition-"+name)/"windows.json",
                     planning=root/("planning-"+name)/"metrics.json")
        config, status, size, health, info, state, windows, planning = [read(paths[k]) for k in paths]
        if digest(paths["config"]) != rec["config_sha256"] or state["config_sha256"] != rec["config_sha256"]:
            raise AssertionError("Config hash mismatch: " + name)
        if status["state"] != "finished" or status["step"] != 600:
            raise AssertionError("Training did not finish at its declared short cap")
        if health["audit"]["step"] != 600 or info["step"] != 600 or state["step"] != 600:
            raise AssertionError("Mixed checkpoint steps")
        if not all(value for value in health["audit"].values() if isinstance(value, bool)):
            raise AssertionError("Health check failed")
        if size["total"] != size["trainable"] + size["frozen"]:
            raise AssertionError("Parameter partition mismatch")
        if size["total"] != health["parameters"] or size["total"] != planning["checkpoint_audit"]["parameters"]:
            raise AssertionError("Different models were evaluated")
        if not planning["checkpoint_audit"]["strict_equal"]:
            raise AssertionError("Planning export load did not match")
        successes = planning["metrics"]["episode_successes"]
        if len(successes) != 50 or not all(x in [0, 1, False, True] for x in successes):
            raise AssertionError("Expected 50 binary strict-case outcomes")
        rate = 100*sum(successes)/50
        if abs(rate-planning["metrics"]["success_rate"]) > 1e-8:
            raise AssertionError("Success rate/individual outcomes mismatch")
        if len(planning["pairs"]) != 50 or state["cases"] != 64 or windows["count"] != 64:
            raise AssertionError("Case count mismatch")
        if digest(paths["windows"]) != state["manifest_sha256"]:
            raise AssertionError("State audit manifest hash mismatch")
        if len(set(info["probes"]["train_episodes"]) & set(info["probes"]["validation_episodes"])):
            raise AssertionError("Probe training/validation leakage")
        reference = dict(planning_manifest_sha256=planning["manifest_sha256"],
                         planning_pairs=planning["pairs"], solver=planning["solver"],
                         plan=planning["plan"], protocol=planning["protocol"], split=planning["split"],
                         normalization=planning["normalization"],
                         state_manifest_sha256=state["manifest_sha256"], state_windows=windows,
                         health_windows=[{k: row[k] for k in ["episode", "start"]} for row in health["cases"]],
                         information_windows=[{k: row[k] for k in ["episodes", "starts"]} for row in info["batches"]],
                         probe_train_episodes=info["probes"]["train_episodes"],
                         probe_validation_episodes=info["probes"]["validation_episodes"],
                         probe_label_keys=info["probes"]["label_keys"])
        task = rec["task"]
        if task in task_references:
            if canonical(reference) != canonical(task_references[task]):
                raise AssertionError("Task cases/planner/probe windows differ across arms: " + name)
        else:
            task_references[task] = reference
        queries = state.get("path_queries")
        if rec["family"] == "pathquery" and (not queries or queries["trained_mode"] != rec["arm"]):
            raise AssertionError("Missing or mismatched path-query audit")
        if rec["family"] == "carrier" and "carrier_chart_certificate" not in state:
            raise AssertionError("Missing actual chart certificate")
        trace_path = root/name/"metrics.jsonl"
        trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        rows.append(dict(**rec, step=600, success_rate=rate,
            episode_successes=[bool(x) for x in successes],
            planning_manifest_sha256=planning["manifest_sha256"],
            state_manifest_sha256=state["manifest_sha256"], model_size=size,
            training=dict(status=status, trace=trace), health=health,
            information=dict(seconds=info.get("seconds"), scope=info.get("scope"),
                             batches=info["batches"], probes=info["probes"],
                             probe_summary=probe_label_summaries(info["probes"])),
            state=reduce_state_audit(state), planning=dict(seconds=planning["seconds"],
                checkpoint=planning["checkpoint"], checkpoint_audit=planning["checkpoint_audit"],
                normalization=planning["normalization"], topology_record_count=len(planning.get("topology", []))),
            source_artifacts={kind: dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)
                              for kind, path in paths.items()}))
    lookup = {(r["task"], r["seed"], r["family"], r["arm"]): r for r in rows}
    family_checks, paired, seed_averages = [], [], []
    for task in ["tworoom", "cube"]:
        query_rows = [r for r in rows if r["task"] == task and r["family"] == "pathquery"]
        query_signature = [canonical({key: r["state"]["path_queries"][key]
                            for key in ["query_step", "query_indices", "label_support"]}) for r in query_rows]
        if len(set(query_signature)) != 1:
            raise AssertionError("Query labels/prototypes sampling differ within a task")
        for family in COMPARISONS:
            for seed in [1, 2]:
                group = [r for r in rows if (r["task"], r["family"], r["seed"]) == (task, family, seed)]
                if len({r["model_size"]["total"] for r in group}) != 1:
                    raise AssertionError("Within-family total capacity differs")
                family_checks.append(dict(task=task, seed=seed, family=family,
                    equal_total_parameters=True, total_parameters=group[0]["model_size"]["total"],
                    trainable_parameters={r["arm"]:r["model_size"]["trainable"] for r in group},
                    calibration_sha256={r["arm"]:read(root/r["id"]/"config.json").get("carrier_calibration",{}).get("sha256") for r in group}))
                if family == "carrier" and len(set(family_checks[-1]["calibration_sha256"].values())) != 1:
                    raise AssertionError("Carrier calibration differs within a matched family")
                for candidate, reference_arm in COMPARISONS[family]:
                    a, b = lookup[task, seed, family, candidate], lookup[task, seed, family, reference_arm]
                    paired.append(dict(task=task, seed=seed, family=family, candidate_arm=candidate,
                        reference_arm=reference_arm, candidate=a["id"], reference=b["id"],
                        **paired_outcomes(a["episode_successes"], b["episode_successes"], bootstrap)))
            for candidate, reference_arm in COMPARISONS[family]:
                seed_rows = [r for r in paired if (r["task"],r["family"],r["candidate_arm"],r["reference_arm"]) == (task,family,candidate,reference_arm)]
                matrix = np.asarray([r["per_case_difference"] for r in seed_rows], dtype=float)
                # Cases are shared across training seeds. Resample the same case
                # indices for both; do not pretend there are 100 independent cases.
                rng = np.random.default_rng(1042)
                draws = matrix.mean(0)[rng.integers(50,size=(bootstrap,50))].mean(1)*100
                seed_averages.append(dict(task=task,family=family,candidate_arm=candidate,
                    reference_arm=reference_arm,seeds=[r["seed"] for r in seed_rows],
                    differences_by_seed_pp=[r["difference_pp"] for r in seed_rows],
                    mean_difference_pp=float(matrix.mean()*100),
                    case_bootstrap_ci95_pp=np.quantile(draws,[.025,.975]).tolist(),
                    scope="Conditional on these two trained seeds; shared-case bootstrap, not training-seed uncertainty."))
    return dict(root=str(root), protocol=protocol, source_commit=protocol["source_commit"],
        completed_endpoints=exits, completed_endpoint_count=len(exits), rows=rows,
        initial_tensor_identity=initial, preflight_review=preflight,
        task_references=task_references, within_family_checks=family_checks,
        paired=paired, paired_seed_averages=seed_averages,
        historical_context=dict(task_order=["tworoom","cube"],
            temporal_forecast600={"tworoom":[6,8],"cube":[14,10]},
            untrained={"tworoom":[4,4],"cube":[8,8]},
            seeds=[1,2], scope="Context only; not matched capacity/parameter-training status. Not included in paired comparisons or confidence intervals."),
        bootstrap=dict(repetitions=bootstrap,seed=1042,
            scope="Exploratory paired empirical-case bootstrap. Degenerate observed outcomes can produce zero-width intervals; these are not population guarantees or multiple-comparison-adjusted tests."),
        collector_sha256=digest(__file__),
        scope="All six arms, both tasks and both training seeds retained. Matched mechanism comparisons are within family only. Clean latent/observed/query/renderer/task measurements remain distinct; no automatic long-run continuation, broad-seed stability or novelty claim.")


def write_report(result, destination):
    lines = ["# Six-arm 600-update validation", "", f"Frozen source `{result['source_commit']}`; all 120 planned endpoints completed with exit 0. Twenty-four fresh runs, both seeds and both tasks retained.", "",
             "The direct comparisons are within the carrier family and within the path-query family. Historical TF and untrained results below are context only. This remains a short gate inside the original warmup, not a long-run or general-stability result.", "",
             "| Task | Family | Arm | Seed 1 strict success | Seed 2 strict success | Trainable / total parameters |", "|---|---|---|---:|---:|---:|"]
    lookup = {(r["task"],r["seed"],r["family"],r["arm"]):r for r in result["rows"]}
    for task in ["tworoom","cube"]:
        for family,arm in ARMS:
            a,b=lookup[task,1,family,arm],lookup[task,2,family,arm]
            sizes = [f"{x['model_size']['trainable']:,}/{x['model_size']['total']:,}" for x in [a,b]]
            lines.append(f"| {task} | {family} | {arm} | {a['success_rate']:g}% | {b['success_rate']:g}% | {sizes[0] if sizes[0]==sizes[1] else '; '.join(sizes)} |")
    lines += ["", "Historical context: TF600 TwoRoom 6/8%, Cube 14/10%; untrained 4/4% and 8/8%. These are not capacity-matched controls.", "",
              "| Task | Family | Candidate − reference | Seed 1 / seed 2 difference | Mean difference, paired case CI |", "|---|---|---|---:|---:|"]
    for row in result["paired_seed_averages"]:
        d=row["differences_by_seed_pp"];lo,hi=row["case_bootstrap_ci95_pp"]
        lines.append(f"| {row['task']} | {row['family']} | {row['candidate_arm']} − {row['reference_arm']} | {d[0]:+g}/{d[1]:+g} pp | {row['mean_difference_pp']:+g} [{lo:+g}, {hi:+g}] pp |")
    lines += ["", "Intervals resample the same 50 cases jointly across the two fixed training seeds. They do not estimate training-seed uncertainty, adjust for the exploratory comparison family, or establish general superiority.", "",
              "`results.json` retains every case outcome, health/intervention record, probe result, normalized forecast error, trained-query per-case score, chart certificate, capacity partition, source path and artifact hash. Weights and NPZ trajectory arrays remain on EFS.", "",
              "Interpretation requirements: compare clean causal rollout against initial carry; teacher-forced and shifted-observed carry receive additional real observations. Query-only zero states test prototype/mask shortcuts. An active chart normalization or an informative observed probe is not proof of better forecast transmission or planning.", "",
              "![Planning](planning-all-arms.png)", "", "![Forecast](forecast-transmission.png)", "",
              "![Queries](query-shortcuts.png)", "", "![Capacity and chart](capacity-and-chart.png)", "",
              "![Input use](input-use.png)", ""]
    Path(destination).write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--bootstrap",type=int,default=10000)
    args=parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result=collect(args.root,args.bootstrap)
    (out/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    write_report(result,out/"RESULTS.md")
    print(json.dumps(dict(rows=len(result["rows"]),endpoints=result["completed_endpoint_count"],
        scores=[dict(task=r["task"],family=r["family"],arm=r["arm"],seed=r["seed"],success_rate=r["success_rate"]) for r in result["rows"]]),indent=2))


if __name__=="__main__":
    main()
