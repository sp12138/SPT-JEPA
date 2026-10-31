"""CPU-only review of five predeclared original Cube evaluator repeats."""
import hashlib
import json
from pathlib import Path

B = Path("/efs/prescottsun/SPT-JEPA")
ROOT = B/"runs/bare_replay6000_20261031"
OUT = B/"reports/bare-replay6000-20261031"
SIGNATURE = ("manifest_sha256", "pairs", "solver", "plan", "protocol", "normalization", "split")


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan, protocol = read(ROOT/"plan.json"), read(ROOT/"protocol.json")
    freeze, launch = read(ROOT/"freeze-record.json"), read(ROOT/"launched-dev2.json")
    assert len(plan["jobs"]) == 5 and (ROOT/"HOLD_DISPATCH").exists()
    assert read(ROOT/"closed-owned-processes.json")["closed"]
    assert sha(ROOT/"protocol.json") == freeze["protocol_sha256"] == launch["protocol_sha256"]
    assert sha(ROOT/"plan.json") == launch["plan_sha256"]
    for path, expected in protocol["immutable_inputs"].items(): assert sha(path) == expected
    rows = []
    for job, spec in zip(plan["jobs"], protocol["jobs"]):
        assert job["id"] == spec["id"] and job["source"] == spec["bare_source"]
        assert read(ROOT/"exits"/(job["id"]+".json"))["returncode"] == 0
        current_path = Path(job["output"])/"metrics.json"
        metrics = {"old": read(spec["original_metrics"]), "wrapped": read(spec["wrapped_metrics"]),
                   "bare": read(current_path)}
        for label, metric in metrics.items():
            for key in SIGNATURE: assert metric[key] == spec["expected_signature"][key], (job["id"], label, key)
            assert metric["checkpoint"] == metrics["old"]["checkpoint"]
            assert len(metric["metrics"]["episode_successes"]) == 50 and metric["checkpoint_audit"]["strict_equal"]
        outcomes = {k:v["metrics"]["episode_successes"] for k,v in metrics.items()}
        comparisons = {}
        for a,b in [("bare","old"),("wrapped","old"),("bare","wrapped")]:
            comparisons[a+"-vs-"+b] = [dict(index=i, **{a:x,b:y}) for i,(x,y) in
                enumerate(zip(outcomes[a],outcomes[b])) if x != y]
        rows.append(dict(id=job["id"], success={k:sum(v)*2 for k,v in outcomes.items()}, outcomes=outcomes,
            differences=comparisons, source=job["source"], original_dispatch=spec["original_dispatch"],
            wrapped_dispatch=spec["wrapper_dispatch"], bare_dispatch=read(Path(job["output"])/"dispatch.json"),
            original_environment=spec["original_environment"], wrapped_environment=spec["wrapper_environment"],
            bare_environment=job["environment"], seeds={k:v.get("seeds") for k,v in metrics.items()},
            signature=spec["expected_signature"], metrics_paths=dict(old=spec["original_metrics"],wrapped=spec["wrapped_metrics"],bare=str(current_path)),
            metric_hashes=dict(old=sha(spec["original_metrics"]),wrapped=sha(spec["wrapped_metrics"]),bare=sha(current_path))))
    differing = sum(bool(r["differences"]["bare-vs-old"]) for r in rows)
    result = dict(completed=5, bare_vs_old_differing_runs=differing, rows=rows, freeze=freeze,
        source_scope="Original committed sources and exact bare commands; only output paths and physical device allocation changed.",
        interpretation="A bare-repeat difference demonstrates that a wrapper is not necessary for that historical discrepancy. It does not establish zero wrapper effect or identify a specific numerical/reset cause. These are targeted reproducibility checks, not five independent benchmark seeds.",
        closure=read(ROOT/"closed-owned-processes.json"), collector_sha256=sha(__file__))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    lines = ["# Five bare Cube replay checks", "", f"All 5 endpoints exit 0; {differing}/5 bare repeats differ from historical case vectors.", "",
        "| Check | Old / wrapped / bare success % | Bare vs old differing indices | Bare vs wrapped differing indices |", "|---|---|---|---|"]
    for r in rows:
        rates=" / ".join(str(r["success"][k]) for k in ["old","wrapped","bare"])
        indices=[", ".join(str(c["index"]) for c in r["differences"][k]) or "none" for k in ["bare-vs-old","bare-vs-wrapped"]]
        lines.append(f"| {r['id']} | {rates} | {indices[0]} | {indices[1]} |")
    lines += ["", result["interpretation"], "", "Cases, weights, CEM budget/seed, normalization, protocol and original model/evaluator source are verified. Actual device placement and EGL identifiers are retained alongside the original and wrapped dispatch records. No reset, simulator seed, action, planner or model change was introduced.", "", "All original, wrapped and bare case vectors remain available; none replaces historical results. The new root is held and its owned idle dispatcher closed."]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print((OUT/"RESULTS.md").read_text())


if __name__ == "__main__": main()
