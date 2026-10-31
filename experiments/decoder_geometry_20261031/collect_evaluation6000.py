"""Complete-only CPU aggregation of the 48 original-command metric endpoints."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from experiments.ideas_validate_20261031.collect import digest, paired_outcomes

B = Path("/efs/prescottsun/SPT-JEPA")
ROOT = B/"runs/decoder_geometry_evaluation_20261031"
OUT = B/"reports/decoder-geometry-evaluation6000-20261031"
METRICS = ("identity", "decoder", "rotated")
SIGNATURE = ("manifest_sha256", "pairs", "solver", "plan", "protocol", "normalization", "split")


def read(path): return json.loads(Path(path).read_text())
@lru_cache(None)
def sha(path): return digest(path)


def joint(pairs):
    delta = np.array([np.asarray(a, float)-np.asarray(b, float) for a,b in pairs])
    assert delta.shape == (2, 50)
    draws = np.random.default_rng(1042).integers(50, size=(10000, 50))
    values = delta[:, draws].mean((0, 2))*100
    return dict(difference_pp=float(delta.mean()*100), case_bootstrap_ci95_pp=np.quantile(values, [.025,.975]).tolist(),
        scope="Cases jointly resampled across the two fixed seeds; not training-seed population uncertainty.")


def main():
    plan, protocol = read(ROOT/"plan.json"), read(ROOT/"protocol.json")
    assert len(plan["jobs"]) == 48 and (ROOT/"HOLD_DISPATCH").exists()
    closure = read(ROOT/"closed-owned-processes.json"); assert closure["closed"]
    source = read(ROOT/"source-record.json"); assert source["source_commit"] == "7db889f6f1f0a65ade3590079ade5e94dc2f5b04"
    manifest = read(source["manifest"])
    source_audit = read(ROOT/"source-preflight.json")
    assert source_audit["original_source"]["source_commit"] == "8e0f821bba84e7ed8c0f16500d7c880fde56d4cf"
    for path, expected in source_audit["byte_equal_files"].items(): assert sha(path) == expected
    for rel in ["as_jepa_tools/benchmark_evaluate.py", "experiments/decoder_geometry_20261031/evaluate.py",
                "experiments/decoder_geometry_20261031/objective.py", "experiments/carrier_progress_20261031/metric_core.py"]:
        assert sha(str(Path(source["source"])/rel)) == manifest["files"][rel]
    rows, groups = [], {}
    for job in plan["jobs"]:
        assert read(ROOT/"exits"/(job["id"]+".json"))["returncode"] == 0 and job["source"] == source
        folder = Path(job["output"]); path = folder/"metrics.json"; metric = read(path); diagnostic = metric["decoder_geometry"]
        assert diagnostic["source"] == source and diagnostic["selected"] == job["metric"]
        assert diagnostic["source_audit"] == source_audit
        assert diagnostic["source_record_sha256"] == sha(str(ROOT/"source-record.json"))
        assert diagnostic["wrapper_sha256"] == sha(str(Path(source["source"])/"experiments/decoder_geometry_20261031/evaluate.py"))
        assert diagnostic["metric_order"] == list(METRICS) and metric["checkpoint_audit"]["strict_equal"]
        assert all(diagnostic["compatibility"].values())
        reference = diagnostic["reference"]; assert sha(reference["path"]) == reference["sha256"]
        old = read(reference["path"])
        for key in SIGNATURE: assert metric[key] == old[key], key
        assert sha(metric["manifest"]) == metric["manifest_sha256"]
        calibration = Path(diagnostic["calibration"])
        assert sha(str(calibration/"calibration.json")) == diagnostic["calibration_sha256"]
        cal = read(calibration/"calibration.json")
        assert cal["passed"] and cal["source"]["source_commit"] == "70328dc101d68f5367705144a01a5d679d940b68"
        assert cal["checkpoint"] == metric["checkpoint"] and sha(str(Path(metric["checkpoint"])/"weights.pt")) == cal["weights_sha256"]
        assert sha(str(calibration/"matrices.npz")) == diagnostic["matrices_sha256"] == cal["artifacts"]["matrices.npz"]
        audit = diagnostic["candidate_audit"]
        assert audit["manual_identity_max_abs_error"] == 0 and audit["extra_model_rollouts"] == 0
        assert audit["metric_order"] == list(METRICS) and audit["calls"] > 0
        assert sha(str(folder/"metric-costs.npz")) == audit["costs_sha256"]
        assert sha(str(folder/"metric-candidate-banks.json")) == audit["candidate_banks_sha256"]
        banks = read(folder/"metric-candidate-banks.json"); assert len(banks) == audit["calls"]
        assert set(banks[0]["selected_indices"]) == set(METRICS)
        actual, previous = metric["metrics"]["episode_successes"], old["metrics"]["episode_successes"]
        assert len(actual) == len(previous) == len(metric["pairs"]) == 50
        assert np.isin(actual, [0,1]).all()
        np.testing.assert_allclose(metric["metrics"]["success_rate"], np.mean(actual)*100)
        changed = [i for i,(a,b) in enumerate(zip(actual,previous)) if a != b]
        assert changed == diagnostic["identity_replay"]["changed_case_indices"]
        assert (not changed) == diagnostic["identity_replay"]["success_exact"]
        row = dict(id=job["id"], task=job["task"], family=job["family"], seed=job["seed"], case_set=job["case_set"],
            metric=job["metric"], success=float(metric["metrics"]["success_rate"]), outcomes=actual,
            signature={key:metric[key] for key in SIGNATURE}, weights_sha256=cal["weights_sha256"],
            calibration_sha256=diagnostic["calibration_sha256"], matrices_sha256=diagnostic["matrices_sha256"],
            reference=reference, identity_replay=dict(applicable=job["metric"] == "identity", exact=not changed,
                changed_case_indices=changed, old_outcomes=previous), candidate_audit=audit, first_bank=banks[0],
            artifact=dict(path=str(path), sha256=sha(str(path))))
        rows.append(row)
        key = (job["task"], job["family"], job["seed"], job["case_set"])
        groups.setdefault(key, {})[job["metric"]] = row
    assert len(groups) == 16
    comparisons, common_banks = [], []
    for key, arms in groups.items():
        assert set(arms) == set(METRICS)
        first = arms["identity"]
        for row in arms.values():
            assert row["signature"] == first["signature"] and row["weights_sha256"] == first["weights_sha256"]
            assert row["matrices_sha256"] == first["matrices_sha256"]
        common_banks.append(dict(task=key[0], family=key[1], seed=key[2], case_set=key[3],
            first_bank_equal={field:len({arms[m]["first_bank"][field] for m in METRICS}) == 1
                for field in ["action_candidates_sha256", "goal_emb_sha256", "endpoint_difference_sha256"]}))
        for a,b in [("decoder","identity"),("decoder","rotated"),("rotated","identity")]:
            comparisons.append(dict(task=key[0], family=key[1], seed=key[2], case_set=key[3],
                comparison=a+"-minus-"+b, **paired_outcomes(arms[a]["outcomes"], arms[b]["outcomes"])))
    joint_rows = []
    for task in ["tworoom", "cube"]:
        for family in ["carrier-identity", "temporal-forecast"]:
            for case in ["primary", "fresh"]:
                for a,b in [("decoder","identity"),("decoder","rotated"),("rotated","identity")]:
                    pairs = [(groups[(task,family,s,case)][a]["outcomes"],groups[(task,family,s,case)][b]["outcomes"]) for s in [1,2]]
                    joint_rows.append(dict(task=task, family=family, case_set=case, comparison=a+"-minus-"+b, **joint(pairs)))
    identity = [r for r in rows if r["metric"] == "identity"]; assert len(identity) == 16
    replay_differences = [dict(id=r["id"], old_success=sum(r["identity_replay"]["old_outcomes"])*2,
        actual_success=r["success"], changed_cases=[dict(index=i, old=r["identity_replay"]["old_outcomes"][i],
        actual=r["outcomes"][i]) for i in r["identity_replay"]["changed_case_indices"]])
        for r in identity if not r["identity_replay"]["exact"]]
    result = dict(completed=48, rows=rows, per_seed_paired=comparisons, paired_two_fixed_seeds=joint_rows,
        identity_replays=16, identity_exact_count=sum(r["identity_replay"]["exact"] for r in identity),
        replay_differences=replay_differences, first_bank_checks=common_banks, source=source, source_audit=source_audit, closure=closure,
        artifacts=dict(plan_sha256=sha(str(ROOT/"plan.json")), protocol_sha256=sha(str(ROOT/"protocol.json")), collector_sha256=sha(__file__)),
        scope="Fixed model, original command-action controller diagnostic. Primary/fresh separate; no projected-action treatment, new training, tuning or algorithm novelty claim.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    lines = ["# Decoder geometry at6000: original-command evaluation", "",
        f"All48 endpoints exit0; {result['identity_exact_count']}/16 identity evaluations reproduce every old6000 case outcome. Same cases, solver, budget, normalization and strict protocol are verified. Model/metric/source hashes and recorded candidate-bank archives pass integrity checks.", "",
        "| Task | Model | Seed | Primary I/G/R | Fresh I/G/R |", "|---|---|---:|---|---|"]
    for task in ["tworoom", "cube"]:
        for family in ["carrier-identity", "temporal-forecast"]:
            for seed in [1,2]:
                values = [" / ".join(f"{groups[(task,family,seed,c)][m]['success']:g}" for m in METRICS) for c in ["primary","fresh"]]
                lines.append(f"| {task} | {family} | {seed} | {values[0]} | {values[1]} |")
    lines += ["", "I=identity, G=decoder Gram, R=same-spectrum random rotation; values are strict success percentages.", "",
        "| Task | Model | Cases | Contrast | Difference pp | Conditional95% interval |", "|---|---|---|---|---:|---|"]
    for row in joint_rows:
        if row["comparison"] == "rotated-minus-identity": continue
        lo,hi = row["case_bootstrap_ci95_pp"]
        lines.append(f"| {row['task']} | {row['family']} | {row['case_set']} | {row['comparison']} | {row['difference_pp']:+.1f} | [{lo:.1f}, {hi:.1f}] |")
    lines += ["", "Intervals jointly resample50 cases across the two fixed training seeds. They do not quantify training-seed population uncertainty; multiple exploratory contrasts are fully retained. Fresh pairs come from the existing validation pool, not an independent test dataset.", "",
        "The fixed decoder metric does not support promotion across both tasks. TwoRoom carrier changes from83% to77% on primary and77% to78% on fresh cases. Its advantage over rotation on fresh cases is evidence that the learned axes can matter, but it does not establish a gain over native Euclidean cost. TwoRoom temporal-forecast rises from3% to10% on both case sets, while rotation reaches9%/11%; that gain does not identify decoder-specific geometry as the cause. Cube effects change sign across seeds or case sets, and no family has a consistent positive effect.", "",
        "Identity replay differences (zero-based case indices):"]
    for row in replay_differences:
        changes = ", ".join(f"{c['index']}:{int(c['old'])}->{int(c['actual'])}" for c in row["changed_cases"])
        lines.append(f"- {row['id']}: old{row['old_success']:g}% -> actual{row['actual_success']:g}%; {changes}.")
    lines += ["", "All original model/runtime byte checks, checkpoint hashes and evaluation signatures agree; every instrumented native cost has zero arithmetic error. These facts do not establish the cause of the five changed historical case outcomes. The table and paired comparisons use actual contemporaneous identity evaluations, and exact historical trajectory reproducibility is not claimed.", "",
        f"All{len(common_banks)} first-bank action/goal/endpoint hash comparisons agree across I/G/R: {all(all(x['first_bank_equal'].values()) for x in common_banks)}.", "",
        "Every metric's three costs are recorded on identical banks within its run. First-bank hashes across treatments are reported separately; later CEM populations may diverge and are not treated as common banks. Raw trace/participation rank are coordinate-dependent decoder diagnostics, not task value or missing-information proofs.", "",
        "This intervention changes only the fixed planning metric. Original command actions remain unchanged, including any existing command/executed-action mismatch. The independent action-projection control is not combined here. All completed roots are held and owned idle dispatchers/monitors are closed; no extension is launched."]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(output=str(OUT), identity_exact=result["identity_exact_count"], rows=[{k:r[k] for k in ["id","success"]} for r in rows]), indent=2))


if __name__ == "__main__": main()
