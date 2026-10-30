"""Write immutable CPU manifests/plans; scheduling is exclusively the parent's job."""
import argparse
import json
from pathlib import Path
from experiments.carrier_progress_20261031.metric_core import METRICS,digest,make_manifest

PY="/efs/prescottsun/project_jepa/envs/lewm/bin/python"


def write(path,value):
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True);p.add_argument("--source-record",required=True)
    p.add_argument("--calibration-root",required=True)
    p.add_argument("--dispatch-root",help="Parent's combined evaluation dispatcher/exits namespace")
    p.add_argument("--reference-root",default="/efs/prescottsun/SPT-JEPA/runs/ideas_validate600_20261031")
    args=p.parse_args();root=Path(args.root);reference=Path(args.reference_root)
    calibration_root=Path(args.calibration_root)
    assert calibration_root.resolve()!=root.resolve()
    source_record=Path(args.source_record);source=json.loads(source_record.read_text())
    script_dir=Path(source["source"])/"experiments/carrier_progress_20261031"
    for name in ["metric_core.py","metric_prepare.py","metric_calibrate.py","metric_evaluate.py"]:
        assert digest(script_dir/name)==digest(Path(__file__).with_name(name)),name
    if root.exists() or calibration_root.exists():raise FileExistsError("Choose fresh metric/calibration roots")
    root.mkdir(parents=True);calibration_root.mkdir(parents=True)
    environment=json.loads((reference/"plan.json").read_text())["environment"]
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    cfg=json.loads((reference/"cube-carrier-identity-seed1/config.json").read_text())
    dataset=BenchmarkWindows(cfg["data"],window=cfg["model"]["window"],split="train")
    manifest=root/"train-clips128.json";write(manifest,make_manifest(dataset))
    cal_jobs,eval_jobs,records=[],[],[]
    slots=[("dev13",i)for i in range(4,8)]+[("dev15",i)for i in range(8)]
    index=0
    for seed in [1,2]:
        for arm in ["identity","bounded"]:
            name=f"cube-carrier-{arm}-seed{seed}";run=reference/name
            current=json.loads((run/"config.json").read_text())
            assert current["data"]==cfg["data"] and current["model"]["window"]==dataset.window
            assert current["model_class"]=="carrier" and current["model"]["carrier_chart"]==arm
            cal=calibration_root/("calibration-"+name);cal.mkdir()
            common=dict(source=source,task="cube",training_seed=seed,arm=arm)
            cal_id="calibration-"+name
            cal_jobs.append(dict(**common,id=cal_id,host="dev13",gpu=4+len(cal_jobs),output=str(cal),
                phase="training-observation-calibration",command=[PY,"-B","-u",str(script_dir/"metric_calibrate.py"),
                    "--run",str(run),"--manifest",str(manifest),"--output",str(cal),
                    "--source-record",str(source_record),"--device","cuda"]))
            for metric in METRICS:
                host,gpu=slots[index];index+=1
                out=root/(metric+"-"+name)
                eval_jobs.append(dict(**common,id=out.name,host=host,gpu=gpu,output=str(out),phase="metric-planning",
                    metric=metric,calibration_id=cal_id,environment={"MUJOCO_EGL_DEVICE_ID":str(gpu)},
                    command=[PY,"-B","-u",str(script_dir/"metric_evaluate.py"),"--metric",metric,
                        "--calibration",str(cal),"--manifest",str(Path(cfg["data"])/"manifests/validation-strict.json"),
                        "--checkpoint",str(run/"model-step0000600"),"--data",cfg["data"],"--output",str(out)]))
            records.append(dict(id=name,run=str(run),seed=seed,arm=arm,config_sha256=digest(run/"config.json"),
                weights_sha256=digest(run/"model-step0000600/weights.pt"),calibration=str(cal)))
    write(calibration_root/"plan.json",dict(root=str(calibration_root),deadline=None,environment=environment,jobs=cal_jobs))
    # Parent must confirm all four calibrations before dispatching this separate plan.
    write(root/"plan.json",dict(root=str(root),deadline=None,environment=environment,jobs=eval_jobs))
    write(root/"protocol.json",dict(source=source,source_record_sha256=digest(source_record),runs=records,
        calibration_root=str(calibration_root),evaluation_dispatch_root=args.dispatch_root or str(root),
        training_manifest=str(manifest),training_manifest_sha256=digest(manifest),metrics=list(METRICS),
        calibration_jobs=4,evaluation_jobs=12,checkpoint_step=600,case_count=50,
        native_planning_reference=str(reference),native_evaluator="as_jepa_tools.benchmark_evaluate.main",
        gate="Do not dispatch evaluation-plan jobs before all four calibrations have exit0 and frozen artifacts verified.",
        scope="Diagnostic controller metric change, not new world-model learning. No updates or privileged/optimal-action labels. Native identity remains an exact implementation control; all three costs also score identical banks.",
        limitations=["Training clips include correlated frames; Ledoit-Wolf is a deterministic regularizer, not an iid risk guarantee.",
            "Inverse increment covariance upweights low-variation directions; it may emphasize nearly static nuisance, not semantic goals.",
            "CEM populations may diverge after the first update, so compare same-bank cost arrays separately from closed-loop success.",
            "Same600 checkpoints and repeated development validation do not establish unseen-case or training-seed generalization."]))
    print(json.dumps(dict(root=str(root),calibration_root=str(calibration_root),
        evaluation_dispatch_root=args.dispatch_root or str(root),calibration_jobs=4,evaluation_jobs=12,
        manifest_sha256=digest(manifest))))


if __name__=="__main__":main()
