"""Run the existing native evaluator with an explicitly named metric diagnostic."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

from experiments.carrier_progress_20261031.metric_core import METRICS,MetricRecorder,digest,install_objective


def main():
    p=argparse.ArgumentParser()
    for name in ["manifest","checkpoint","data","output","calibration"]:p.add_argument("--"+name,required=True)
    p.add_argument("--metric",choices=METRICS,required=True);p.add_argument("--limit",type=int)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/"metrics.json").exists():raise FileExistsError("Do not overwrite a metric evaluation")
    calibration=Path(args.calibration);report=json.loads((calibration/"calibration.json").read_text())
    assert report["step"]==600 and report["training_clips"]==128
    assert Path(args.checkpoint).resolve()==Path(report["checkpoint"]).resolve()
    assert digest(Path(args.checkpoint)/"weights.pt")==report["weights_sha256"]
    assert Path(args.data).resolve()==Path(report["data"]).resolve()
    assert digest(Path(args.data)/"dataset_card.json")==report["dataset_card_sha256"]
    assert digest(calibration/"matrices.npz")==report["matrices_sha256"]
    assert digest(report["train_manifest"])==report["train_manifest_sha256"]
    with np.load(calibration/"matrices.npz") as loaded:matrices=dict(loaded)
    recorder=MetricRecorder(out)
    from as_jepa_tools import benchmark_evaluate
    native_argv=[benchmark_evaluate.__file__]
    for key in ["manifest","checkpoint","data","output"]:native_argv.extend(["--"+key,getattr(args,key)])
    if args.limit is not None:native_argv.extend(["--limit",str(args.limit)])
    argv=sys.argv
    try:
        sys.argv=native_argv
        with install_objective(matrices,args.metric,recorder) as objectives:
            benchmark_evaluate.main()
    finally:
        sys.argv=argv
    if len(objectives)!=1:raise AssertionError("Expected exactly one native objective")
    audit=recorder.finish()
    assert audit["manual_identity_max_abs_error"]==0, audit
    metrics=json.loads((out/"metrics.json").read_text())
    metrics["metric_diagnostic"]=dict(selected=args.metric,native_objective=objectives[0],
        calibration=str(calibration),calibration_sha256=digest(calibration/"calibration.json"),
        train_manifest_sha256=report["train_manifest_sha256"],matrices_sha256=report["matrices_sha256"],
        native_evaluator_sha256=digest(benchmark_evaluate.__file__),wrapper_sha256=digest(__file__),
        core_sha256=digest(Path(__file__).with_name("metric_core.py")),candidate_audit=audit,
        scope="Explicit planner metric intervention. Native goal encoding, raw-state prediction, CEM configuration, case manifest and physical evaluator unchanged. This is not a training method or a novelty claim.")
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(metric=args.metric,candidate_audit=audit)),flush=True)


if __name__=="__main__":main()
