"""CPU-only complete collection: 32 bounded evaluations and separate replay controls."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from experiments.ideas_validate_20261031.collect import digest, paired_outcomes

B = Path("/efs/prescottsun/SPT-JEPA")
OLD = B/"runs/action_projection_evaluation_20261031"
NEW = B/"runs/action_projection_recovery_20261031"
OUT = B/"reports/action-projection6000-20261031"
FAMILIES = ("native-cal", "TF64", "F64", "local1024")
SIGNATURE = ("manifest_sha256", "pairs", "solver", "plan", "normalization", "split", "protocol")


def read(path): return json.loads(Path(path).read_text())
@lru_cache(None)
def sha(path): return digest(path)


def joint(pairs):
    d = np.array([np.array(a, float)-np.array(b, float) for a,b in pairs])
    assert d.shape == (2,50)
    draws = np.random.default_rng(1042).integers(50, size=(10000,50))
    values = d[:,draws].mean((0,2))*100
    return dict(difference_pp=float(d.mean()*100), case_bootstrap_ci95_pp=np.quantile(values,[.025,.975]).tolist())


def endpoint(root, job, expected_exit):
    exit_record = read(root/"exits"/(job["id"]+".json"))
    assert exit_record["returncode"] == expected_exit
    spec = read(job["spec"]); assert sha(job["spec"]) == job["spec_sha256"]
    assert spec["source"] == job["source"] and spec["output"] == job["output"]
    for path,h in spec["immutable_inputs"].items(): assert sha(path) == h, path
    for manifest_key in ["manifest","runtime_manifest"]:
        manifest = read(spec["source"][manifest_key])
        for rel,h in manifest["files"].items(): assert sha(str(Path(manifest["source"])/rel)) == h
    path = Path(job["output"])/"metrics.json"; metric = read(path); ref = read(spec["reference_metrics"])
    for key in SIGNATURE: assert metric[key] == ref[key], (job["id"],key)
    assert metric["checkpoint"] == spec["checkpoint"] == ref["checkpoint"]
    assert metric["checkpoint_audit"] == ref["checkpoint_audit"] and metric["checkpoint_audit"]["strict_equal"]
    diag = metric["controller_diagnostic"]; assert diag["mode"] == job["mode"] and diag["source"] == job["source"]
    contract=diag["input_contract"]
    assert contract["spec_sha256"] == job["spec_sha256"] and contract["matched_cases_checkpoint_solver_plan_normalization"] and contract["checkpoint_files_unchanged"]
    assert contract["reference_metrics"] == spec["reference_metrics"] and contract["reference_sha256"] == sha(spec["reference_metrics"])
    for path,h in diag["helper_files"].items(): assert sha(path) == h
    actual, original = metric["metrics"]["episode_successes"], ref["metrics"]["episode_successes"]
    assert len(actual) == len(original) == len(metric["pairs"]) == 50 and np.isin(actual,[0,1]).all()
    np.testing.assert_allclose(metric["metrics"]["success_rate"],np.mean(actual)*100)
    stats=diag["stats"]
    if job["mode"] == "bounded":
        assert stats["score_calls"] > 0 and stats["execution_calls"] > 0 and stats["returned_command_violations"] == 0
    else:
        assert diag["original_replay_successes_exact"] == (actual == original)
        assert stats["score_calls"] == stats["execution_calls"] == 0
    return dict(id=job["id"], root=str(root), task=job["task"], family=job["family"], seed=job["training_seed"],
        case_set=job["case_set"], mode=job["mode"], exit_record=exit_record,
        success=float(np.mean(actual)*100), original_success=float(np.mean(original)*100), outcomes=actual, original_outcomes=original,
        signature={k:metric[k] for k in SIGNATURE}, source=job["source"], checkpoint=spec["checkpoint"],
        weights_sha256=spec["immutable_inputs"][str(Path(spec["checkpoint"])/"weights.pt")],
        config_sha256=spec["immutable_inputs"][str(Path(spec["checkpoint"])/"config.json")],
        reference=dict(path=spec["reference_metrics"],sha256=sha(spec["reference_metrics"])),
        artifact=dict(path=str(path),sha256=sha(str(path))), stats=stats, helper_files=diag["helper_files"],
        original_replay_successes_exact=diag["original_replay_successes_exact"],
        paired_vs_historical=paired_outcomes(actual,original))


def main():
    old,new = read(OLD/"plan.json"),read(NEW/"plan.json")
    assert len(old["jobs"])==36 and len(new["jobs"])==10
    launch,freeze=read(NEW/"launched-dev15.json"),read(NEW/"freeze-record.json")
    assert sha(str(NEW/"plan.json"))==launch["plan_sha256"]
    assert sha(str(NEW/"protocol.json"))==launch["protocol_sha256"]==freeze["protocol_sha256"]
    assert read(NEW/"source-record.json")["source_commit"]=="2e6e000f023a3cef09535d6be38746a12a5e2138"
    for root in [OLD,NEW]:
        assert (root/"HOLD_DISPATCH").exists() and read(root/"closed-owned-processes.json")["closed"]
    assert sha(OLD/"plan.json") == read(OLD/"closed-owned-processes.json")["old_plan_sha256"]
    old_bounded=[j for j in old["jobs"] if j["mode"]=="bounded" and (OLD/"exits"/(j["id"]+".json")).exists()]
    new_bounded=[j for j in new["jobs"] if j["mode"]=="bounded"]
    assert len(old_bounded)==24 and len(new_bounded)==8
    assert {j["id"] for j in old_bounded}.isdisjoint(j["id"] for j in new_bounded)
    rows=[endpoint(OLD,j,0) for j in old_bounded]+[endpoint(NEW,j,0) for j in new_bounded]
    controls=[endpoint(OLD,j,1) for j in old["jobs"] if j["mode"]=="original" and j["task"]=="cube"]
    controls += [endpoint(NEW,j,0) for j in new["jobs"] if j["mode"]=="original"]
    assert len(controls)==4
    expected={(t,f,s,c) for t in ["tworoom","cube"] for f in FAMILIES for s in [1,2] for c in ["primary","fresh"]}
    groups={(r["task"],r["family"],r["seed"],r["case_set"]):r for r in rows}
    assert set(groups)==expected and len(rows)==32
    between, joint_rows, first_banks = [],[],[]
    for task in ["tworoom","cube"]:
        for case in ["primary","fresh"]:
            subset=[r for r in rows if r["task"]==task and r["case_set"]==case]
            assert all(r["signature"]==subset[0]["signature"] for r in subset)
            first_banks.append(dict(task=task,case_set=case,
                action_inputs_identical=len({r["stats"]["first_score"]["input_sha256"] for r in subset})==1,
                scope="First score only; subsequent optimized banks and trajectories differ."))
            for family in FAMILIES:
                pairs=[(groups[(task,family,s,case)]["outcomes"],groups[(task,family,s,case)]["original_outcomes"]) for s in [1,2]]
                joint_rows.append(dict(task=task,family=family,case_set=case,comparison="bounded-minus-historical",**joint(pairs)))
                for reference in ["TF64","native-cal"]:
                    if family==reference:continue
                    pairs=[]
                    for seed in [1,2]:
                        a,b=groups[(task,family,seed,case)],groups[(task,reference,seed,case)]
                        pairs.append((a["outcomes"],b["outcomes"]))
                        between.append(dict(task=task,family=family,reference=reference,seed=seed,case_set=case,
                            **paired_outcomes(a["outcomes"],b["outcomes"])))
                    joint_rows.append(dict(task=task,family=family,case_set=case,comparison="bounded-minus-bounded-"+reference,**joint(pairs)))
    bare_path=B/"reports/bare-replay6000-20261031/results.json";bare=read(bare_path);assert bare["completed"]==5
    clipping=[]
    for task in ["tworoom","cube"]:
        for family in FAMILIES:
            stats=[r["stats"] for r in rows if r["task"]==task and r["family"]==family]
            clipping.append(dict(task=task,family=family,
                candidate_clipped_coordinate_pct=100*sum(s["score_clipped_coordinates"] for s in stats)/sum(s["score_coordinates"] for s in stats),
                execution_clipped_coordinate_pct=100*sum(s["execution_clipped_coordinates"] for s in stats)/sum(s["execution_finite_coordinates"] for s in stats),
                returned_command_violations=sum(s["returned_command_violations"] for s in stats)))
    result=dict(completed_bounded=32,old_bounded=24,recovery_bounded=8,rows=rows,original_controls=controls,
        between_model_per_seed=between,paired_two_fixed_seeds=joint_rows,first_bank_checks=first_banks,
        clipping_summaries=clipping,
        bare_repeats=dict(path=str(bare_path),sha256=sha(str(bare_path)),rows=bare["rows"]),
        protocol=read(NEW/"protocol.json"),closure=read(NEW/"closed-owned-processes.json"),
        artifacts=dict(collector_sha256=sha(__file__),old_plan_sha256=sha(str(OLD/"plan.json")),new_plan_sha256=sha(str(NEW/"plan.json"))),
        limitations="Historical reference is fixed for all comparisons, never chosen from old/wrapped/bare outcomes. Case bootstrap conditions on two fixed training seeds and observed simulator realizations; it excludes training-seed population and environment repeat variance. Native LeWM and the proposed models differ in capacity/objective/recipe. No matched-capacity or new learning-method claim.")
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    lines=["# Common bounded-controller evaluation at 6000", "", "All 32 bounded endpoints exit 0: 24 original-root + 8 recovery-root. The two failed Cube replay controls remain exit 1; two recovered TwoRoom replay controls exit 0.", "",
        "This common bounded controller does not yield a cross-task winner. TwoRoom TF improves to10% primary/7% fresh, mainly through seed1, while native-cal reaches76%/84% and F64 reaches80%/82%. In Cube, F64 falls on all four seed/case-set evaluations to8%/7% mean; local1024 reaches21%/22% against native-cal17%/19% and TF14%/20%, but these small Cube advantages do not establish a substantial gain and local1024 remains weaker on TwoRoom. The controller intervention does not resolve the contrasting task weaknesses.", "",
        "| Task | Model | Seed | Primary historical → bounded % | Fresh historical → bounded % |", "|---|---|---:|---|---|"]
    for task in ["tworoom","cube"]:
        for family in FAMILIES:
            for seed in [1,2]:
                cells=[]
                for case in ["primary","fresh"]:
                    r=groups[(task,family,seed,case)];cells.append(f"{r['original_success']:g} → {r['success']:g}")
                lines.append(f"| {task} | {family} | {seed} | {cells[0]} | {cells[1]} |")
    lines += ["", "| Task | Model | Cases | Contrast | Difference pp | Conditional 95% interval |", "|---|---|---|---|---:|---|"]
    for r in joint_rows:
        lo,hi=r["case_bootstrap_ci95_pp"]
        lines.append(f"| {r['task']} | {r['family']} | {r['case_set']} | {r['comparison']} | {r['difference_pp']:+.1f} | [{lo:.1f}, {hi:.1f}] |")
    lines += ["", result["limitations"], "", "Projection is applied uniformly before candidate model rollout and before returning raw actions. TwoRoom also changes recorded action history; Cube lacks equivalent native clipping, so this changes its actual controller. It does not change learned weights or sample-level losses. All within-task comparisons use the same bounded treatment and preserve CEM proposal/update rules and evaluation budget.", "",
        "All four task/case-set first candidate-bank input hashes agree across all models and seeds; later banks are not assumed identical. All32 evaluations report zero returned-command bound violations. Across the four model families,21.6–29.8% of scored TwoRoom action coordinates and2.4–3.3% of Cube coordinates were clipped. These are descriptive fractions along each treated run, not common-bank causal comparisons. Box feasibility alone does not establish joint training-distribution support or prediction accuracy.", "",
        "The four pass-through controls and five bare repeats are retained separately in results.json. Two bare runs also differ from their historical vectors; wrapper zero effect and a specific reset cause are not established. No historical result is replaced. Fresh cases come from the existing validation pool."]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(output=str(OUT),rows=[{k:r[k] for k in ["id","success","original_success"]} for r in rows]),indent=2))


if __name__ == "__main__":main()
