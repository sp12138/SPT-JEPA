"""Estimate a goal metric covariance from training frames only,without pose labels."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from as_jepa_tools.evaluate import load_model
from as_jepa_tools.probes import features
from vforge.ops import OpsConfig,initialize_ops
@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--data",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.manual_seed(42)
 initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda",0),parallel_state=None)
 model=load_model(a.checkpoint);x,_,_=features(model,a.data,"train",20000,17,fp32=True)
 c=np.cov(x,rowvar=False);scale=np.trace(c)/len(c);c=c+np.eye(len(c))*max(scale,1e-8)*1e-5
 np.save(out/"covariance.npy",c);(out/"metrics.json").write_text(json.dumps(dict(train_frames=len(x),checkpoint=a.checkpoint,ridge=scale*1e-5,scope="Train-only feature covariance; no held-out data or pose target used."),indent=2))
if __name__=="__main__":main()
