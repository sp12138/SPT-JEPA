"""Encode a frozen training-only clip manifest and fit two covariance metrics."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.carrier_progress_20261031.metric_core import digest, fit_metrics, verify_manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True);p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True);p.add_argument("--source-record",required=True)
    p.add_argument("--device",default="cpu");p.add_argument("--batch-size",type=int,default=4)
    args=p.parse_args()
    root,out=Path(args.run),Path(args.output)
    if (out/"calibration.json").exists() or (out/"matrices.npz").exists():
        raise FileExistsError("Do not overwrite frozen metric calibration")
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((root/"config.json").read_text())
    if cfg.get("model_class")!="carrier" or cfg["model"]["carrier_chart"] not in ["identity","bounded"]:
        raise ValueError("Only the predeclared frozen carrier checkpoints are in scope")
    weights=root/"model-step0000600"/"weights.pt"
    if json.loads((root/"status.json").read_text())["step"]!=600:
        raise ValueError("Calibration must use the immutable 600-step gate")
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    from as_jepa_lewm.factory import build_model
    from vforge.ops import OpsConfig,initialize_ops
    torch.set_num_threads(2);torch.manual_seed(1042);torch.set_float32_matmul_precision("highest")
    device=torch.device(args.device)
    initialize_ops(OpsConfig(backend="sdpa" if device.type=="cuda" else "sdpa_math",allow_fallback=False),device=device,parallel_state=None)
    model=build_model(cfg).to(device).eval();model.requires_grad_(False)
    model.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True),strict=True)
    ds=BenchmarkWindows(cfg["data"],window=model.window,split="train")
    manifest=json.loads(Path(args.manifest).read_text());verify_manifest(ds,manifest)
    loader=DataLoader(Subset(ds,[x["index"] for x in manifest["windows"]]),batch_size=args.batch_size,shuffle=False,num_workers=0)
    values=[];offset=0;started=time.time()
    with torch.no_grad():
        for batch in loader:
            for j,episode in enumerate(batch["episode"].tolist()):
                expected=manifest["windows"][offset+j]
                assert episode==expected["episode"] and batch["rows"][j].tolist()==expected["rows"]
            pixels=model.normalize(batch["pixels"].to(device))
            values.append(model.encode({"pixels":pixels})["emb"].float().cpu().numpy())
            offset+=len(batch["episode"])
    if ds.file is not None:ds.file.close()
    clips=np.concatenate(values);assert clips.shape==(128,manifest["window"],64)
    arrays,fit=fit_metrics(clips)
    np.savez_compressed(out/"matrices.npz",**arrays)
    np.savez_compressed(out/"training-embeddings.npz",clips=clips)
    import sklearn
    report=dict(run=str(root),checkpoint=str(weights.parent),step=600,weights_sha256=digest(weights),
        config_sha256=digest(root/"config.json"),data=cfg["data"],dataset_card_sha256=digest(ds.root/"dataset_card.json"),
        train_manifest=str(Path(args.manifest)),train_manifest_sha256=digest(args.manifest),
        source=json.loads(Path(args.source_record).read_text()),source_record_sha256=digest(args.source_record),
        helper_sha256=digest(__file__),core_sha256=digest(Path(__file__).with_name("metric_core.py")),
        dimensions=64,training_clips=128,frames_per_clip=manifest["window"],frameskip=manifest["frameskip"],
        observed_samples=128*manifest["window"],increment_samples=128*(manifest["window"]-1),
        fit=fit,sklearn_version=sklearn.__version__,numpy_version=np.__version__,torch_version=torch.__version__,
        device=str(device),input_precision="float32/highest",matrices_sha256=digest(out/"matrices.npz"),
        training_embeddings_sha256=digest(out/"training-embeddings.npz"),seconds=time.time()-started,
        scope="Frozen observed training RGB only. Identical clips for both covariance fits and every checkpoint. No privileged labels, goal labels, optimal actions, validation fitting, model update or extra policy supervision. Adjacent clip frames are correlated; Ledoit-Wolf is a fixed regularizer here, not a claimed iid-optimal guarantee.")
    (out/"calibration.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(out),fit=fit,seconds=report["seconds"])),flush=True)


if __name__=="__main__":main()
