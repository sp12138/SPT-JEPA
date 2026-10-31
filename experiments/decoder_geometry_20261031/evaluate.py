"""Original evaluator and command actions; only its fixed endpoint metric changes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

from experiments.carrier_progress_20261031.metric_core import MetricRecorder, digest
from experiments.decoder_geometry_20261031.objective import METRICS, install_objective, validate_matrices


def main():
    p = argparse.ArgumentParser()
    for name in ["manifest", "checkpoint", "data", "output", "calibration", "calibration-sha256",
                 "reference", "reference-sha256", "source-record"]: p.add_argument("--"+name, required=True)
    p.add_argument("--metric", choices=METRICS, required=True)
    a = p.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    if (out/"metrics.json").exists(): raise FileExistsError("Preserve previous evaluation")
    source = json.loads(Path(a.source_record).read_text()); source_manifest = json.loads(Path(source["manifest"]).read_text())
    for relative in ["experiments/decoder_geometry_20261031/evaluate.py", "experiments/decoder_geometry_20261031/objective.py",
                     "experiments/carrier_progress_20261031/metric_core.py"]:
        assert digest(Path(source["source"])/relative) == source_manifest["files"][relative]
    assert Path(__file__).resolve() == (Path(source["source"])/"experiments/decoder_geometry_20261031/evaluate.py").resolve()
    calibration = Path(a.calibration); report = json.loads((calibration/"calibration.json").read_text())
    assert digest(calibration/"calibration.json") == a.calibration_sha256
    assert report["source"]["source_commit"] == "70328dc101d68f5367705144a01a5d679d940b68"
    assert report["passed"] and report["clips"] == 128 and report["probes_per_clip"] == 8 and report["dimensions"] == 64
    assert report["state_before_sha256"] == report["state_after_sha256"]
    assert Path(a.checkpoint).resolve() == Path(report["checkpoint"]).resolve()
    assert digest(Path(a.checkpoint)/"weights.pt") == report["weights_sha256"]
    assert digest(Path(a.checkpoint)/"config.json") == report["config_sha256"]
    cfg = json.loads((Path(a.checkpoint).parent/"config.json").read_text())
    assert Path(cfg["data"]).resolve() == Path(a.data).resolve()
    assert digest(report["manifest"]) == report["manifest_sha256"]
    train_manifest = json.loads(Path(report["manifest"]).read_text())
    assert digest(Path(a.data)/"dataset_card.json") == train_manifest["dataset_card_sha256"]
    assert digest(calibration/"matrices.npz") == report["artifacts"]["matrices.npz"]
    with np.load(calibration/"matrices.npz") as f: matrices = dict(f)
    validate_matrices(matrices)
    assert digest(a.reference) == a.reference_sha256
    reference = json.loads(Path(a.reference).read_text()); manifest = json.loads(Path(a.manifest).read_text())
    assert reference["checkpoint"] == a.checkpoint and reference["checkpoint_audit"]["strict_equal"]
    assert reference["manifest_sha256"] == digest(a.manifest) and reference["pairs"] == manifest["pairs"]
    assert len(manifest["pairs"]) == len(reference["metrics"]["episode_successes"]) == 50
    recorder = MetricRecorder(out, metric_names=METRICS)
    from as_jepa_tools import benchmark_evaluate
    from experiments.decoder_geometry_20261031.calibrate import verify_sources
    source_audit = verify_sources(source)
    original_source = source_audit["original_source"]
    assert digest(benchmark_evaluate.__file__) == digest(Path(original_source["source"])/"as_jepa_tools/benchmark_evaluate.py")
    native_argv = [benchmark_evaluate.__file__]
    for key in ["manifest", "checkpoint", "data", "output"]: native_argv.extend(["--"+key, getattr(a, key)])
    argv = sys.argv
    try:
        sys.argv = native_argv
        with install_objective(matrices, a.metric, recorder) as objectives: benchmark_evaluate.main()
    finally: sys.argv = argv
    assert len(objectives) == 1
    audit = recorder.finish(); assert audit["manual_identity_max_abs_error"] == 0
    metrics = json.loads((out/"metrics.json").read_text())
    matched = {k:metrics[k] == reference[k] for k in ["manifest_sha256", "pairs", "solver", "plan", "protocol", "normalization", "split"]}
    actual, previous = metrics["metrics"]["episode_successes"], reference["metrics"]["episode_successes"]
    assert len(actual) == len(previous) == 50
    changed = [i for i, (x,y) in enumerate(zip(actual, previous)) if x != y]
    metrics["decoder_geometry"] = dict(selected=a.metric, metric_order=list(METRICS), source=source,
        source_audit=source_audit, native_objective=objectives[0], calibration=a.calibration,
        calibration_sha256=a.calibration_sha256, matrices_sha256=report["artifacts"]["matrices.npz"],
        candidate_audit=audit, reference=dict(path=a.reference, sha256=a.reference_sha256),
        compatibility=matched, identity_replay=dict(applicable=a.metric == "identity", success_exact=not changed,
            changed_case_indices=changed, current_episode_successes=actual, previous_episode_successes=previous),
        source_record_sha256=digest(a.source_record), wrapper_sha256=digest(__file__),
        scope="Original command actions, transforms, raw-state rollout and CEM unchanged. Fixed decoder/rotated factors only. Identity successes are freshly evaluated; any discrepancy is preserved. Subsequent CEM banks may diverge.")
    (out/"metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False)+"\n")
    assert all(matched.values()), matched
    print(json.dumps(dict(metric=a.metric, success=metrics["metrics"]["success_rate"],
        identity_replay_exact=(not changed) if a.metric == "identity" else None, candidate_audit=audit)), flush=True)


if __name__ == "__main__": main()
