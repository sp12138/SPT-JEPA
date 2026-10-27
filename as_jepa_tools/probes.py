"""Episode-disjoint dynamics and nuisance probes, using training features only to fit."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from as_jepa_tools.data import PushTWindows
from as_jepa_tools.evaluate import load_model
from vforge.ops import initialize_ops,OpsConfig

@torch.no_grad()
def features(model,root,split,count,seed,noisy=False,fp32=False):
 data=PushTWindows(root,window=1,gap=0,split=split);rng=np.random.default_rng(seed);indices=rng.choice(len(data),size=min(count,len(data)),replace=False)
 loader=DataLoader(Subset(data,indices),batch_size=128,num_workers=3,pin_memory=True)
 xs=[];ys=[];names=['agent_x','agent_y','block_x','block_y','sin_block_angle','cos_block_angle','agent_vx','agent_vy']
 for batch in loader:
  batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not fp32):
   pixels,_,_=model.prepare(batch);mu=model.encode(dict(pixels=pixels))['emb'][:,0].float()
   if noisy and model.mode in {"L2","S2"}:mu=mu+model.sigma0*torch.randn_like(mu)
  state=batch['state'][:,0].float();target=torch.cat([state[:,:4],state[:,4:5].sin(),state[:,4:5].cos(),state[:,5:7]],1)
  if model.domain=='color':
   ep=batch['episode'].float();colors=torch.stack([.35+.65*torch.remainder(torch.sin(ep*12.9898+k*78.233)*43758.5453,1) for k in range(3)],1)
   bias=torch.stack([35*torch.remainder(torch.sin(ep*9.125+k*23.77)*12781.13,1) for k in range(3)],1);target=torch.cat([target,colors,bias],1)
  if model.domain=='dot':
   key=batch['rows'][:,0].float();nuisance=torch.stack([torch.frac(torch.sin(key*12.9898+k*78.233)*43758.5453).abs() for k in range(1,6)],1);target=torch.cat([target,nuisance],1)
  xs.append(mu.cpu().numpy());ys.append(target.cpu().numpy())
 if model.domain=='color':names+=['color_scale_r','color_scale_g','color_scale_b','color_bias_r','color_bias_g','color_bias_b']
 if model.domain=='dot':names+=['dot_x','dot_y','dot_r','dot_g','dot_b']
 return np.concatenate(xs),np.concatenate(ys),names

def main(argv=None,initialize=True):
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--train',type=int,default=20000);p.add_argument('--test',type=int,default=5000);p.add_argument('--noisy',action='store_true');p.add_argument('--seed',type=int,default=20261027);p.add_argument('--fp32',action='store_true');a=p.parse_args(argv)
 torch.set_num_threads(2);torch.manual_seed(a.seed);torch.set_float32_matmul_precision('highest')
 if initialize:initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=torch.device('cuda',0),parallel_state=None)
 model=load_model(a.checkpoint);started=time.time();train_x,train_y,names=features(model,a.data,'train',a.train,17,a.noisy,a.fp32);test_x,test_y,_=features(model,a.data,'test',a.test,18,a.noisy,a.fp32)
 scaler=StandardScaler().fit(train_x);yscaler=StandardScaler().fit(train_y);ridge=RidgeCV(alphas=np.logspace(-4,4,9),cv=5).fit(scaler.transform(train_x),yscaler.transform(train_y));prediction=yscaler.inverse_transform(ridge.predict(scaler.transform(test_x)))
 scores=r2_score(test_y,prediction,multioutput='raw_values');cov=np.cov(train_x,rowvar=False);eigen=np.clip(np.linalg.eigvalsh(cov),0,None);prob=eigen/eigen.sum().clip(1e-30)
 rank=float(np.exp(-(prob*np.log(prob.clip(1e-30))).sum()))
 result=dict(feature_precision='FP32' if a.fp32 else 'BF16 autocast',float32_matmul_precision=torch.get_float32_matmul_precision(),checkpoint=str(Path(a.checkpoint).resolve()),noise_seed=a.seed,feature_type='sampled noisy code' if a.noisy else 'posterior mean / deterministic embedding',r2={name:float(value) for name,value in zip(names,scores)},agent_r2=float(scores[:2].mean()),block_position_r2=float(scores[2:4].mean()),block_angle_r2=float(scores[4:6].mean()),velocity_r2=float(scores[6:8].mean()),feature_effective_rank=rank,feature_std=float(train_x.std(0).mean()),selected_alpha=float(ridge.alpha_),train_frames=len(train_x),test_frames=len(test_x),seconds=time.time()-started,split='episode-disjoint; ridge regularization selected only on training features',scope='single trained seed; exploratory linear decodability')
 if len(scores)>8:result['nuisance_r2']=float(scores[8:].mean())
 if not a.noisy and model.mode in {"L2","S2"}:
  test_cov=np.cov(test_x,rowvar=False,bias=True);test_eigen=np.clip(np.linalg.eigvalsh(test_cov),0,None)
  result["heldout_noisy_channel_covariance_bound_nats"]=float(.5*np.log1p(test_eigen/model.sigma0**2).sum())
  result["heldout_noisy_channel_diagonal_bound_nats"]=float(.5*np.log1p(np.diag(test_cov)/model.sigma0**2).sum())
  result["heldout_covariance_eigenvalues"]=test_eigen.tolist()
  result["information_bound_scope"]="Empirical covariance Gaussian-noise channel upper bound on uniformly sampled test frames; not task mutual information."

 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'probes.json').write_text(json.dumps(result,indent=2));np.savez(out/'probe_predictions.npz',truth=test_y,prediction=prediction);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
