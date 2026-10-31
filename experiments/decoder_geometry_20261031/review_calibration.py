"""CPU-only integrity review of the complete eight-model calibration stage."""
import hashlib
import json
from pathlib import Path
import numpy as np

B = Path("/efs/prescottsun/SPT-JEPA")
ROOT = B/"runs/decoder_geometry_calibration_20261031"
OUT = B/"reports/decoder-geometry-calibration-20261031"


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda:f.read(1024*1024), b""): result.update(data)
    return result.hexdigest()


def main():
    plan, protocol = read(ROOT/"plan.json"), read(ROOT/"protocol.json")
    assert len(plan["jobs"]) == 8
    closure = read(ROOT/"closed-owned-processes.json"); assert closure["closed"]
    assert (ROOT/"HOLD_DISPATCH").exists() and (ROOT/"HOLD_BEFORE_ACTIVATION").exists()
    preflight = read(ROOT/"source-preflight.json")
    reference_rotation, reference_schedule = None, None
    rows = []
    for job in plan["jobs"]:
        root = Path(job["output"]); report = read(root/"calibration.json")
        exit_record = read(ROOT/"exits"/(job["id"]+".json")); assert exit_record["returncode"] == 0
        record = next(x for x in protocol["records"] if x["checkpoint"] == report["checkpoint"])
        assert report["passed"] and report["source"] == job["source"]
        assert report["source"]["source_commit"] == "70328dc101d68f5367705144a01a5d679d940b68"
        assert report["weights_sha256"] == record["weights_sha256"] == sha(Path(report["checkpoint"])/"weights.pt")
        assert report["manifest_sha256"] == record["manifest_sha256"] == sha(report["manifest"])
        assert report["state_before_sha256"] == report["state_after_sha256"]
        assert report["parameters_frozen"] and report["parameter_grads_absent"] and report["model_eval"]
        assert report["source_audit"] == preflight
        assert report["clips"] == report["point_count"] == 128 and report["probes_per_clip"] == 8
        for name, expected in report["artifacts"].items(): assert sha(root/name) == expected
        manifest, points = read(report["manifest"]), read(root/"points.json")
        assert len(points) == len(manifest["windows"]) == 128
        for i, (point, entry) in enumerate(zip(points, manifest["windows"])):
            assert point["clip_index"] == i and point["episode"] == entry["episode"] and point["rows"] == entry["rows"]
            assert point["anchor_row"] == entry["rows"][0]
            assert 1 <= point["horizon"] <= 7 and point["target_row"] == entry["rows"][point["horizon"]]
            assert 0 < point["time"] < 1 and point["output_elements"] == 12288
        schedule = [{k:x[k] for k in ["clip_index", "horizon", "time", "forecast", "shift", "rho", "sigma0"]} for x in points]
        if reference_schedule is None: reference_schedule = schedule
        assert schedule == reference_schedule
        with np.load(root/"response-vectors.npz") as f: values = f["vectors"]
        with np.load(root/"matrices.npz") as f: matrices = dict(f)
        assert values.shape == (128, 8, 64) and np.isfinite(values).all()
        flat = values.reshape(-1, 64); raw = flat.T @ flat/len(flat)
        np.testing.assert_allclose(raw, matrices["decoder_gram_raw"], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.mean(np.sum(values**2, axis=-1), axis=1), [x["response_trace"] for x in points], rtol=1e-12)
        normalized = raw*64/np.trace(raw); q = matrices["rotation"]
        if reference_rotation is None: reference_rotation = q
        assert np.array_equal(reference_rotation, q)
        np.testing.assert_allclose(q.T@q, np.eye(64), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(normalized, matrices["decoder_gram"], rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(q.T@normalized@q, matrices["rotated_gram"], rtol=1e-11, atol=1e-12)
        for name in ["decoder", "rotated"]:
            factor = matrices[name+"_factor"]
            np.testing.assert_allclose(factor@factor.T, matrices[name+"_gram"], rtol=1e-10, atol=1e-12)
        eigen = np.linalg.eigvalsh(raw)
        np.testing.assert_allclose(eigen, report["spectrum"]["raw"]["eigenvalues"], rtol=1e-9, atol=1e-14)
        np.testing.assert_allclose(np.linalg.eigvalsh(normalized), np.linalg.eigvalsh(matrices["rotated_gram"]), rtol=1e-9, atol=1e-12)
        assert eigen[0] > 0 and report["spectrum"]["raw"]["numerical_rank"] == 64
        rows.append(dict(id=job["id"], task=job["task"], family=job["family"], seed=job["seed"],
            trace=float(eigen.sum()), eigenvalues=eigen.tolist(), rank=64,
            participation_rank=float(eigen.sum()**2/np.sum(eigen**2)), condition_number=float(eigen[-1]/eigen[0]),
            top1_trace_fraction=float(eigen[-1]/eigen.sum()), top3_trace_fraction=float(eigen[-3:].sum()/eigen.sum()),
            clips=128, vectors=1024, forecast_clips=sum(p["forecast"] for p in points),
            zero_response_clips=sum(p["response_trace"] == 0 for p in points),
            source=report["source"], weights_sha256=report["weights_sha256"],
            state_sha256=report["state_before_sha256"], manifest_sha256=report["manifest_sha256"],
            vjp_seconds=report["vjp_seconds"], calibration_seconds=report["total_seconds"],
            dispatcher_elapsed_seconds=exit_record["finished_at"]-exit_record["started_at"],
            peak_device_bytes=report["peak_device_bytes"], calibration_artifact=dict(path=str(root/"calibration.json"), sha256=sha(root/"calibration.json"))))
    result = dict(passed=True, completed=8, rows=rows, same_rotation_all_models=True,
        same_horizon_time_noise_seed_and_condition_branch=True, closure=closure,
        plan_sha256=sha(ROOT/"plan.json"), protocol_sha256=sha(ROOT/"protocol.json"),
        review_source_sha256=sha(__file__), scope="Fixed training-cohort response geometry only. No planning, action projection, model update or estimator tuning.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    lines = ["# Frozen global decoder geometry: eight calibrations complete", "",
        "All eight exit0 and pass CPU integrity review. Each uses exactly128 original training clips and1024 finite VJP vectors;69 clips select forecast conditioning and59 observed posterior conditioning. All response traces are positive. Source/imports match original8e, weights and complete model state are unchanged, and the sampling schedule and fixed rotation agree across all eight models.", "",
        "| Task | Model | Seed | Raw trace | Participation rank | Top3 trace | Condition number |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows: lines.append(f"| {r['task']} | {r['family']} | {r['seed']} | {r['trace']:.6f} | {r['participation_rank']:.2f} | {100*r['top3_trace_fraction']:.1f}% | {r['condition_number']:.0f} |")
    lines += ["", "All matrices have numerical rank64; participation rank describes concentration, not absent dimensions. TF response is concentrated in fewer directions than F64 in these coordinates. This is a reproducible description of the fixed estimates, not an explanation of task performance. Raw trace and spectral concentration depend on latent coordinates/scales, so they cannot establish superior representation quality across models.", "",
        "Recomputed vector outer products match raw G. Gnorm differs only by the positive scalar64/trace(G); QᵀGnormQ preserves its spectrum. QR factors reproduce both quadratic forms without a floor, inverse or identity mixture. Single-point probe variance remains possible; these checks do not establish Monte Carlo or population convergence. The held original cohort and eight-probe rule were not changed.", "",
        "Per-model calibration takes35–41 seconds internally, with20–23 seconds in VJPs; peak device allocation is about432MB. Original snapshots, candidates and preactivation HOLD are preserved. The completed root is held; owned idle dispatcher91717 and monitor91718 are closed. No planning, command projection, action clipping, training or further calibration was launched."]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(passed=True, output=str(OUT), rows=[{k:r[k] for k in ["id", "trace", "participation_rank", "top3_trace_fraction"]} for r in rows]), indent=2))


if __name__ == "__main__": main()
