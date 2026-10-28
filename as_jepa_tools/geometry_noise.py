"""Estimate a train-only diagonal renderer pullback for a declared noise control."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from as_jepa_tools.data import PushTWindows
from as_jepa_tools.evaluate import load_model
from vforge.models.as_jepa.pixels import pack
from vforge.ops import OpsConfig,initialize_ops
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--data",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.manual_seed(20261028)
 initialize_ops(OpsConfig(backend="sdpa_math",allow_fallback=False),device=torch.device("cuda",0),parallel_state=None)
 m=load_model(a.checkpoint);m.requires_grad_(False)
 data=PushTWindows(a.data,window=8,gap=8,split="train");indices=np.random.default_rng(17).choice(len(data),32,replace=False);fisher=torch.zeros(m.state_dim,device="cuda")
 for j,index in enumerate(indices):
  batch=data[int(index)];batch={k:v[None].cuda() if torch.is_tensor(v) else torch.tensor([v],device="cuda") for k,v in batch.items()}
  with torch.no_grad():
   pixels,frames,anchor=m.prepare(batch);mu=m.encode(dict(pixels=pixels))["emb"][:,3].float()
  state=mu.detach().unflatten(-1,(m.state_slots,m.state_width)).requires_grad_();y=pack(frames[:,3]);noise=torch.randn_like(y);time=torch.full((1,1),[.2,.5,.8][j%3],device="cuda")
  with m.renderer.anchored(pack(anchor)):
   velocity=m.renderer((1-time.reshape(1,1,1,1,1))*y+time.reshape(1,1,1,1,1)*noise,time,condition=state)
  for trial in range(4):
   direction=torch.randint(0,2,velocity.shape,device="cuda").float()*2-1
   gradient=torch.autograd.grad((velocity.float()*direction).sum()/np.sqrt(velocity[0].numel()),state,retain_graph=trial<3)[0].flatten()
   fisher+=gradient.square()/128
 floor=float(fisher.mean())*.05;diagonal=(fisher+floor).reciprocal();diagonal=diagonal/diagonal.mean();diagonal=diagonal.clamp(.05,20);diagonal=diagonal/diagonal.mean()
 report=dict(checkpoint=a.checkpoint,weights_sha256=hashlib.sha256((Path(a.checkpoint)/"weights.pt").read_bytes()).hexdigest(),training_window_indices=indices.tolist(),fisher_diagonal=fisher.cpu().tolist(),noise_diagonal=diagonal.cpu().tolist(),scope="FP32 train-only Hutchinson diagonal of renderer velocity Jacobian; diagonal approximation; trained-reference cost must be counted; no full-metric or optimality claim.")
 (out/"metrics.json").write_text(json.dumps(report,indent=2));print(json.dumps(dict(min=float(diagonal.min()),max=float(diagonal.max()))))
if __name__=="__main__":main()
