"""Prepare eight held calibration candidates. No launch, training, or evaluation."""
import argparse
import json
from pathlib import Path
import shutil

from experiments.carrier_progress_20261031.metric_core import digest, verify_manifest

B = Path("/efs/prescottsun/SPT-JEPA")
PY = "/efs/prescottsun/project_jepa/envs/lewm/bin/python"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True)
    p.add_argument("--source-record", help="Omit only for an explicitly unlaunchable pre-freeze preview")
    a = p.parse_args(); root = Path(a.root)
    if root.exists(): raise FileExistsError("Preparation always requires a new root")
    source = json.loads(Path(a.source_record).read_text()) if a.source_record else None
    if source:
        m = json.loads(Path(source["manifest"]).read_text())
        for name in ["core.py", "calibrate.py", "prepare.py"]:
            relative = "experiments/decoder_geometry_20261031/"+name
            assert digest(Path(__file__).with_name(name)) == m["files"][relative]
        script = str(Path(source["source"])/"experiments/decoder_geometry_20261031/calibrate.py")
    else: script = "<COMMIT_AND_FREEZE_REQUIRED>/experiments/decoder_geometry_20261031/calibrate.py"
    parent = B/"runs/carrier_learn6000_20261031"
    cohorts = dict(cube=B/"runs/carrier_metric_20261031/train-clips128.json",
        tworoom=B/"reports/native-lewm600-20261031/source-bncal-v1/train128-tworoom.json")
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    jobs, records = [], []; root.mkdir(parents=True)
    for task in ["tworoom", "cube"]:
        manifest = root/f"train128-{task}.json"; shutil.copy2(cohorts[task], manifest)
        for family in ["carrier-identity", "temporal-forecast"]:
            for seed in [1, 2]:
                name = f"{task}-{family}-seed{seed}"; run = parent/name
                cfg = json.loads((run/"config.json").read_text())
                assert cfg["dataset_kind"] == "benchmark_hdf5" and cfg["model"]["window"] == 8
                assert cfg["model"].get("state_slots", 4)*cfg["model"].get("state_width", 16) == 64
                if family == "carrier-identity": assert cfg["model_class"] == "carrier" and cfg["model"]["carrier_chart"] == "identity"
                else: assert cfg.get("model_class", "robust") == "robust" and cfg["robust_variant"]
                dataset = BenchmarkWindows(cfg["data"], window=8, split="train")
                verify_manifest(dataset, json.loads(manifest.read_text()))
                checkpoint = run/"model-step0006000"; sha = digest(checkpoint/"weights.pt")
                output = root/("calibrate-"+name)
                command = [PY, "-B", "-u", script, "--checkpoint", str(checkpoint), "--manifest", str(manifest),
                    "--source-record", str(a.source_record or "<FROZEN_SOURCE_RECORD_REQUIRED>"), "--output", str(output),
                    "--expected-weights-sha256", sha, "--device", "cuda"]
                jobs.append(dict(id="calibrate-"+name, host="dev2", gpu=len(jobs), output=str(output),
                    phase="frozen-training-global-decoder-gram", task=task, family=family, seed=seed,
                    source=source, hold=True, command=command))
                records.append(dict(id=name, checkpoint=str(checkpoint), weights_sha256=sha,
                    config_sha256=digest(run/"config.json"), export_config_sha256=digest(checkpoint/"config.json"),
                    manifest=str(manifest), manifest_sha256=digest(manifest), original_manifest=str(cohorts[task])))
    env = json.loads((parent/"plan.json").read_text())["environment"]
    plan = dict(root=str(root), deadline=None, environment=env, jobs=jobs)
    (root/"candidate-plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    (root/"HOLD_DISPATCH").write_text("Candidate only. Root must authorize frozen calibration launch. No planning authorized.\n")
    protocol = dict(source=source, helper_requires_new_commit_freeze=source is None, records=records,
        seed=20261031, clips=128, probes=8, horizons="one predeclared uniform draw 1..7 per clip",
        times="native shifted logit-normal", rotation_seed=20261031, normalization="64*G/trace(G), common positive scalar only",
        estimator="mean of 1024 VJP outer products; no inverse, floor or identity mixture",
        candidate_metrics=["identity", "decoder", "rotated"], tasks=2, seeds=2, families=2, endpoints=8,
        gate="Review all 8 calibration integrity/spectra before any separate planning decision; no training or auto-continuation.")
    (root/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    print(json.dumps(dict(root=str(root), jobs=8, held=True, source_frozen=source is not None)))


if __name__ == "__main__": main()
