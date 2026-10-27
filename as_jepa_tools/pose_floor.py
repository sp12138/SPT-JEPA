"""Measure the frozen pose evaluator on real frames in each controlled domain."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from as_jepa_tools.data import PushTWindows
from as_jepa_tools.evaluate import load_model,pose_net
from vforge.ops import initialize_ops,OpsConfig


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--data",required=True)
    p.add_argument("--pose",required=True);p.add_argument("--output",required=True);p.add_argument("--frames",type=int,default=10000)
    a=p.parse_args();torch.set_num_threads(2)
    initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda",0),parallel_state=None)
    model=load_model(a.checkpoint);pose=pose_net(a.pose);data=PushTWindows(a.data,window=1,gap=0,split="test")
    indices=np.random.default_rng(47).choice(len(data),min(a.frames,len(data)),replace=False)
    loader=DataLoader(Subset(data,indices),batch_size=128,num_workers=2,pin_memory=True)
    errors=[];started=time.time()
    for batch in loader:
        batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
        with torch.autocast("cuda",dtype=torch.bfloat16):
            _,real,_=model.prepare(batch);pred=pose(real[:,0]).float()
        truth=batch["state"][:,0,:4].float();errors.append((pred[:,:4]*512-truth).reshape(-1,2,2).norm(dim=-1).cpu())
    error=torch.cat(errors);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    result=dict(domain=model.domain,frames=len(error),agent_position_error=float(error[:,0].mean()),
                block_position_error=float(error[:,1].mean()),seconds=time.time()-started,
                scope="Episode-disjoint real-frame evaluator floor; no world-model generated frames")
    (out/"floor.json").write_text(json.dumps(result,indent=2));np.save(out/"per_frame_error.npy",error.numpy());print(json.dumps(result,indent=2))


if __name__=="__main__":main()
