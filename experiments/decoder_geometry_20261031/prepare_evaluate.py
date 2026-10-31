"""48 held native-command evaluation candidates; no GPU or activation."""
import argparse
import json
from pathlib import Path
import shutil

from experiments.carrier_progress_20261031.metric_core import digest
from experiments.decoder_geometry_20261031.objective import METRICS, validate_matrices
import numpy as np

B = Path("/efs/prescottsun/SPT-JEPA")
CAL = B/"runs/decoder_geometry_calibration_20261031"
PARENT = B/"runs/carrier_learn6000_20261031"
PY = "/efs/prescottsun/project_jepa/envs/lewm/bin/python"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True)
    p.add_argument("--source-record", help="Omit only for an unlaunchable pre-freeze preview")
    a = p.parse_args(); root = Path(a.root)
    if root.exists(): raise FileExistsError("Preserve existing candidates")
    source = json.loads(Path(a.source_record).read_text()) if a.source_record else None
    if source:
        m = json.loads(Path(source["manifest"]).read_text())
        for name in ["objective.py", "evaluate.py", "prepare_evaluate.py"]:
            relative = "experiments/decoder_geometry_20261031/"+name
            assert digest(Path(__file__).with_name(name)) == m["files"][relative]
        script = str(Path(source["source"])/"experiments/decoder_geometry_20261031/evaluate.py")
    else: script = "<COMMIT_AND_FREEZE_REQUIRED>/experiments/decoder_geometry_20261031/evaluate.py"
    calibrations = json.loads((CAL/"plan.json").read_text())["jobs"]
    slots = [(host, gpu) for host in ["dev2", "dev13"] for gpu in range(8)]
    root.mkdir(parents=True); jobs, records = [], []; manifests = {}
    for caljob in calibrations:
        assert json.loads((CAL/"exits"/(caljob["id"]+".json")).read_text())["returncode"] == 0
        caldir = Path(caljob["output"]); report = json.loads((caldir/"calibration.json").read_text())
        assert report["passed"] and report["clips"] == 128 and report["probes_per_clip"] == 8
        assert digest(caldir/"matrices.npz") == report["artifacts"]["matrices.npz"]
        with np.load(caldir/"matrices.npz") as f: validate_matrices(dict(f))
        checkpoint = Path(report["checkpoint"]); cfg = json.loads((checkpoint.parent/"config.json").read_text())
        assert digest(checkpoint/"weights.pt") == report["weights_sha256"]
        name = checkpoint.parent.name
        for case, phase in [("primary", "planning"), ("fresh", "fresh-planning")]:
            reference = PARENT/(phase+"-"+name)/"metrics.json"
            old = json.loads(reference.read_text())
            assert old["checkpoint"] == str(checkpoint) and len(old["metrics"]["episode_successes"]) == 50
            original = Path(old["manifest"]); key = (caljob["task"], case)
            assert digest(original) == old["manifest_sha256"]
            manifest = root/("-".join(key)+"-50.json")
            if key not in manifests:
                shutil.copy2(original, manifest); manifests[key] = digest(manifest)
            assert digest(manifest) == old["manifest_sha256"] == manifests[key]
            for metric in METRICS:
                host, gpu = slots[len(jobs) % len(slots)]
                output = root/(case+"-"+metric+"-"+name)
                command = [PY, "-B", "-u", script, "--metric", metric, "--manifest", str(manifest),
                    "--checkpoint", str(checkpoint), "--data", cfg["data"], "--output", str(output),
                    "--calibration", str(caldir), "--calibration-sha256", digest(caldir/"calibration.json"),
                    "--reference", str(reference), "--reference-sha256", digest(reference),
                    "--source-record", str(a.source_record or "<FROZEN_SOURCE_RECORD_REQUIRED>")]
                jobs.append(dict(id=output.name, host=host, gpu=gpu, hold=True, source=source, output=str(output),
                    task=caljob["task"], family=caljob["family"], seed=caljob["seed"], case_set=case, metric=metric,
                    phase="fixed-decoder-metric-native-command-planning", environment={"MUJOCO_EGL_DEVICE_ID":str(gpu)}, command=command))
            records.append(dict(model=name, case_set=case, checkpoint=str(checkpoint), weights_sha256=report["weights_sha256"],
                calibration=str(caldir), calibration_sha256=digest(caldir/"calibration.json"),
                matrix_sha256=report["artifacts"]["matrices.npz"], manifest=str(manifest), manifest_sha256=digest(manifest),
                reference=str(reference), reference_sha256=digest(reference), reference_episode_successes=old["metrics"]["episode_successes"]))
    assert len(jobs) == 48 and len(records) == 16 and len(manifests) == 4
    assert all(sum((j["host"], j["gpu"]) == slot for j in jobs) == 3 for slot in slots)
    plan = dict(root=str(root), deadline=None, environment=json.loads((PARENT/"plan.json").read_text())["environment"], jobs=jobs)
    (root/"candidate-plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    (root/"HOLD_DISPATCH").write_text("48 candidates only. Freeze and root activation required; confirm occupied GPUs free.\n")
    protocol = dict(source=source, requires_new_freeze=source is None, records=records, jobs=48, metrics=list(METRICS),
        checkpoints=8, cases_per_job=50, case_sets=["primary", "fresh"], all_jobs_held=True,
        allocation="dev2/dev13 only:16 proposed slots, three serial jobs per slot. Chart gates are held complete; root must verify actual GPUs free. dev15 is reserved for the separate projection control.",
        actions="Original command actions and native transforms/bounds only; no projection or new clipping",
        identity="Original nativecost tensor returned directly. Fresh success vectors compared case-by-case to old6000; discrepancies retained, never filled from old results.",
        same_bank="All three metric costs/order/hash recorded for each identical bank; later CEM populations may differ across runs",
        scope="Bounded controller diagnostic. Same CEM/planner/sample/episode budgets and frozen models. No training, physical supervision or metric tuning.")
    (root/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    print(json.dumps(dict(root=str(root), jobs=48, metrics=list(METRICS), all_held=True, frozen=source is not None)))


if __name__ == "__main__": main()
