"""Factual held-out rollout versus persistence; no model updates or optimal actions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.metaworld_baselines_20261107.models import PublicBaseline
from experiments.metaworld_baselines_20261107.train_native import digest, save, Windows


def main(config,checkpoint,output):
    cfg=json.loads(Path(config).read_text());torch.set_num_threads(2)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=PublicBaseline(cfg['method'],cfg['sources'])
    model.load_state_dict(state['model'],strict=True);step=state['step'];del state
    model=model.cuda().eval().requires_grad_(False)
    before=digest(model)
    data=Windows(cfg['data'],8,cfg['seed'],'test')
    task_ids=np.load(Path(cfg['data'])/'episode_task_ids.npy') if (Path(cfg['data'])/'episode_task_ids.npy').exists() else None
    # Common deterministic indices, independent of outcomes, same across methods.
    indices=np.linspace(0,len(data)-1,min(400,len(data)),dtype=np.int64)
    rows=[]
    for lo in range(0,len(indices),8):
        selected=indices[lo:lo+8]
        batch=torch.utils.data.default_collate([data[int(i)] for i in selected])
        batch={k:v.cuda() for k,v in batch.items()}
        with torch.inference_mode():
            emb=model.encode(dict(pixels=model.normalize(batch['pixels'])))['emb'].float()
            target=emb[:,2:]
            copy=(target-emb[:,1:2]).square().mean(-1)
            z=emb.flatten(0,1);zc=z-z.mean(0)
            cov=zc.T@zc/max(1,len(zc)-1)
            pr=(torch.trace(cov).square()/cov.square().sum().clamp_min(1e-20)).item()
            for kind in ['factual','zero','reverse']:
                actions=batch['action'][:,1:7].clone()
                if kind=='zero':
                    # Physical zero action, not normalized mean action.
                    mean=actions.new_tensor(data.base.mean);std=actions.new_tensor(data.base.std)
                    actions[:]=(-mean/std)
                elif kind=='reverse':actions=actions.flip(1)
                info=dict(pixels=model.normalize(batch['pixels'][:,:2])[:,None],
                    action_history=batch['action'][:,:1,None].transpose(1,2))
                predicted=model.rollout(info,actions[:,None])['predicted_emb'][:,0,2:].float()
                errors=(predicted-target).square().mean(-1)
                for j,index in enumerate(selected):
                    ep,start=data.base.locate(int(index))
                    rows.append(dict(index=int(index),episode=int(data.base.episode_ids[ep]),start=int(start),
                        kind=kind,forecast_mse=errors[j].cpu().tolist(),persistence_mse=copy[j].cpu().tolist(),
                        batch_latent_std=float(z.std(0).mean()),batch_cov_participation_rank=pr))
    assert digest(model)==before
    summary={}
    for kind in ['factual','zero','reverse']:
        r=[row for row in rows if row['kind']==kind]
        errors=np.asarray([row['forecast_mse'] for row in r]);copy=np.asarray([row['persistence_mse'] for row in r])
        summary[kind]=dict(mean_forecast_mse_by_horizon=errors.mean(0).tolist(),
            mean_persistence_mse_by_horizon=copy.mean(0).tolist(),
            forecast_over_persistence=(errors.mean(0)/np.maximum(copy.mean(0),1e-20)).tolist())
    save(output,dict(passed=True,method=cfg['method'],step=step,unique_windows=len(indices),rows=rows,summary=summary,
        model_state_before=before,model_state_after=digest(model),model_updates=0,
        scope='Factual actions are observed executed controls, never presumed optimal. '
        'Zero/reversed forecasts are sensitivity checks, not realized counterfactual control. '
        'Latent MSE units differ across methods; no cross-model physical-error/causal claim.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['config','checkpoint','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();main(a.config,a.checkpoint,a.output)
