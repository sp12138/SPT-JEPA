"""Frozen factual RGB prediction and cached own-intervention replay,0SGD.

Intervention controls were independently chosen by our diagnostic,never drawn
from demonstrations.Their observed images are diagnostic targets only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset

from experiments.metaworld_eval_20261103.evaluate import model_sha


@torch.no_grad()
def rollout(model,current,commands,steps):
    base=current;mapping=model.grid.permute(0,3,1,2).expand(len(current),-1,-1,-1).clone()
    past=current.new_zeros(len(current),4);valid=current.new_zeros(len(current));role=valid.clone()
    for _ in range(steps):
        drift,effect,_,_=model.fields(current,torch.zeros_like(current),role,past,valid)
        flow=drift+torch.einsum('na,nachw->nchw',torch.cat([commands,commands.abs()],-1),effect)
        mapping=model.warp(mapping,flow)
        current=F.grid_sample(base,mapping.permute(0,2,3,1),mode='bilinear',padding_mode='border',align_corners=False)
        past=commands;valid=valid.new_ones(len(valid))
    return current


def run(a):
    from as_jepa_lewm.factory import build_model
    from as_jepa_tools.native_rgb_pairs import NativeRGBPairs
    torch.set_num_threads(a.threads);torch.set_float32_matmul_precision('highest')
    a.output.mkdir(parents=True,exist_ok=False);cfg=json.loads((a.run/'config.json').read_text())
    if a.cache:cfg['native_rgb_cache']=str(a.cache)
    model=build_model(cfg).to(a.device).eval().requires_grad_(False)
    weight=a.run/f'model-step{a.step:07d}/weights.pt';model.load_state_dict(torch.load(weight,map_location=a.device,weights_only=True));before=model_sha(model)
    data=NativeRGBPairs(cfg,'test');chosen=np.linspace(0,len(data)-1,min(1024,len(data)),dtype=int).tolist();factual=[]
    with torch.inference_mode():
        for batch in DataLoader(Subset(data,chosen),batch_size=16):
            batch={k:v.to(a.device)if torch.is_tensor(v)else v for k,v in batch.items()}
            predicted,var=model.distribution(batch);target=model.encoder(batch['next_pixels']);current=model.encoder(batch['current_pixels'])
            copy=(target-current).square().mean((1,2,3));error=(predicted-target).square().mean((1,2,3))
            altered=dict(batch);altered['executed_control']=torch.zeros_like(batch['executed_control']);zero,_=model.distribution(altered)
            zero_error=(zero-target).square().mean((1,2,3))
            for j in range(len(copy)):
                factual.append(dict(boundary=bool(batch['boundary'][j]),episode=int(batch['episode'][j]),start=int(batch['start'][j]),
                    mse=float(error[j]),copy_mse=float(copy[j]),zero_control_mse=float(zero_error[j]),variance=float(var[j].flatten()[0])))
        physical=[]
        for path in sorted(a.interventions.glob('*.npz')):
            raw=np.load(path);initial=torch.from_numpy(raw['initial']).permute(2,0,1)[None].to(a.device)
            actual=torch.from_numpy(raw['future']).permute(0,3,1,2).to(a.device)
            initial=model.encoder(initial);actual=model.encoder(actual)
            commands=torch.as_tensor(raw['commands'],device=a.device);predicted=rollout(model,initial.expand(9,-1,-1,-1),commands,5)
            err=(predicted-actual).square().mean((1,2,3));copy=(actual-initial).square().mean((1,2,3));axes=[]
            for k in range(4):
                odd=(actual[1+2*k]-actual[2+2*k])*.5;predodd=(predicted[1+2*k]-predicted[2+2*k])*.5
                even=(actual[1+2*k]+actual[2+2*k])*.5-actual[0];predeven=(predicted[1+2*k]+predicted[2+2*k])*.5-predicted[0]
                axes.append(dict(axis=k,actual_odd_rms=float(odd.square().mean().sqrt()),actual_even_rms=float(even.square().mean().sqrt()),
                    odd_cosine=float(F.cosine_similarity(odd.flatten(),predodd.flatten(),dim=0)),
                    odd_mse=float((odd-predodd).square().mean()),even_mse=float((even-predeven).square().mean())))
            output=a.output/path.name
            np.savez_compressed(output,initial=(initial[0]*255).round().byte().permute(1,2,0).cpu().numpy(),
                predicted=(predicted*255).round().clamp(0,255).byte().permute(0,2,3,1).cpu().numpy(),
                actual=(actual*255).round().clamp(0,255).byte().permute(0,2,3,1).cpu().numpy())
            physical.append(dict(case=path.stem,source_trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                mse=err.cpu().tolist(),copy_mse=copy.cpu().tolist(),axes=axes,output=str(output)))
    assert model_sha(model)==before
    summary={}
    for role in [False,True]:
        rows=[r for r in factual if r['boundary']==role]
        summary['goal'if role else'forward']=dict(windows=len(rows),mse=float(np.mean([r['mse']for r in rows])),
            copy_mse=float(np.mean([r['copy_mse']for r in rows])),zero_control_mse=float(np.mean([r['zero_control_mse']for r in rows])),
            beats_copy=sum(r['mse']<r['copy_mse']for r in rows))
    summary['own_interventions']=dict(cases=len(physical),interventions=9*len(physical),
        mse=float(np.mean([r['mse']for r in physical])),copy_mse=float(np.mean([r['copy_mse']for r in physical])),
        median_odd_cosine=np.median([[axis['odd_cosine']for axis in r['axes']]for r in physical],axis=0).tolist())
    report=dict(passed=True,step=a.step,model_unchanged=True,model_sha256=before,weights_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
        model_updates=0,summary=summary,factual=factual,physical=physical,
        scope='Heldout factual observations and prior9self-chosen interventions on32sharedphysicalcases only.No inverse training/actionlabels/extra trainingdata.No policy success inference from MSE.')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['run','interventions','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cache',type=Path);p.add_argument('--step',type=int,required=True);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    run(p.parse_args())
