"""Finite CPU-only factual prediction tests; no fitting, goals, or simulation.

Actions in all arms are recorded forward inputs, not desired policy targets.
Changed-input predictions have NO realized counterfactual ground truth; their
error against the factual outcome only tests conditional predictive dependence.
"""
import argparse
import collections
import datetime
import json
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from torch.utils.data import default_collate
from .common import ROOT,read,save,sha,tensor_sha,now
from .data import FactualWindows
from .train import build

HORIZONS=[1,5,15,30,60]
MODES=['factual_open','zero_open','reversed_open','observe2','observe15']


def manifest(dataset,count=16):
    eligible=[];start=0
    for r,end in zip(dataset.records,dataset.ends):
        if r['length']>=68:eligible.append((start,r))
        start=int(end)
    assert len(eligible)>=count
    chosen=np.linspace(0,len(eligible)-1,count,dtype=int)
    rng=np.random.default_rng(932010);rows=[]
    for i in chosen:
        off,r=eligible[i];t=int(rng.integers(7,r['length']-60))
        rows.append(dict(index=off+t,episode=r['episode'],start=t,episode_length=r['length']))
    assert len({r['episode'] for r in rows})==count
    return rows


@torch.no_grad()
def evaluate(job,step=10000,count=16):
    cfg=read(job['config']);torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    assert not torch.cuda.is_available(),'This diagnostic must be CPU-only'
    model=build(cfg).eval().requires_grad_(False)
    weights=Path(job['output'])/f'model-step{step:07d}/weights.pt'
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    before=tensor_sha(model);dataset=FactualWindows(cfg['data'],'test',maximum_horizon=60)
    selected=manifest(dataset,count);rows=[];started=time.monotonic()
    for start in range(0,len(selected),2):
        subset=selected[start:start+2];batch=default_collate([dataset[r['index']] for r in subset])
        assert batch['future_valid'].all()
        hidden,current=model.prepare(batch)
        truth=model.pixels(batch['future_pixels'].flatten(0,1)).unflatten(0,(len(subset),60))
        recorded=batch['actions'];copy=(truth-current[:,None]).square().mean((2,3,4))
        per=[dict(r,copy={str(h):float(copy[i,h-1]) for h in HORIZONS},modes={},restarts={}) for i,r in enumerate(subset)]
        for mode in MODES:
            state=hidden.clone();image=current.clone();errors={}
            controls=recorded if mode not in ['zero_open','reversed_open'] else (torch.zeros_like(recorded) if mode=='zero_open' else recorded.flip(1))
            for t in range(60):
                state,image=model.predict(state,image,controls[:,t])
                if t+1 in HORIZONS:errors[str(t+1)]=(image-truth[:,t]).square().mean((1,2,3)).tolist()
                interval=int(mode[7:]) if mode.startswith('observe') else 0
                if interval and (t+1)%interval==0:
                    image=truth[:,t];state=model.observe(state,image)
                if t==29 and interval:
                    # Both receive the same current factual image. The reset arm
                    # discards history; it receives no extra observation/action.
                    variants={'filtered':state,'current_only_reset':model.observe(model.initial(len(image)),image)}
                    for name,belief in variants.items():
                        imagined=image
                        for j in range(30,45):belief,imagined=model.predict(belief,imagined,recorded[:,j])
                        mse=(imagined-truth[:,44]).square().mean((1,2,3)).tolist()
                        for i in range(len(subset)):per[i]['restarts'][mode+'-'+name]=mse[i]
            for i in range(len(subset)):per[i]['modes'][mode]={h:v[i] for h,v in errors.items()}
        rows.extend(per)
    assert tensor_sha(model)==before
    aggregate={}
    for mode in MODES:
        aggregate[mode]={}
        for h in map(str,HORIZONS):
            values=np.array([r['modes'][mode][h] for r in rows]);copies=np.array([r['copy'][h] for r in rows])
            assert np.isfinite(values).all()
            aggregate[mode][h]=dict(mse=float(values.mean()),copy_mse=float(copies.mean()),
                sum_error_over_copy=float(values.sum()/max(copies.sum(),1e-20)),
                median_window_error_over_copy=float(np.median(values/np.maximum(copies,1e-20))))
    restarts={key:float(np.mean([r['restarts'][key] for r in rows])) for key in rows[0]['restarts']}
    result=dict(passed=True,utc=now(),job=job['name'],step=step,weights_sha256=sha(weights),
        state_before=before,state_after=tensor_sha(model),selected_heldout_windows=selected,
        windows=len(rows),rows=rows,aggregate=aggregate,restarts=restarts,seconds=time.monotonic()-started,
        optimizer_updates=0,gpu_contexts=0,physics_steps=0,desired_goals_used=False,
        interpretation='16heldoutoriginalepisodes,long-enoughsubset;notcontrolsuccess. All errorunitsareRGBMSE. '
            'observe2/15errorsareforecastsBEFOREassimilatingthatendpoint. Resetcomparisonatnative30 predicts15morewithsamefactualinputs. '
            'Changed-inputerrorsagainstfactualoutcomesdoNOTestablishactualcounterfactualaccuracy oractionoptimality.')
    return result


def run(host):
    plan=read(ROOT/'plan.json');output=ROOT/'factual-diagnostics';output.mkdir(exist_ok=True)
    assert not (output/f'launch-{host}.json').exists()
    save(output/f'launch-{host}.json',dict(pid=os.getpid(),host=host,utc=now(),new_gpu_contexts=0,new_optimizer_updates=0))
    for job in [j for j in plan['jobs'] if j['host']==host]:
        sealed=Path(job['output'])/'model-step0010000/SEALED.json'
        while not sealed.exists():
            if (ROOT/'STOP').exists():return
            assert time.time()<plan['deadline'];time.sleep(5)
        dest=output/(job['name']+'.json');assert not dest.exists()
        try:
            result=evaluate(job);save(dest,result)
            print(job['name'],result['seconds'],result['aggregate']['factual_open']['60'],flush=True)
        except BaseException:
            save(output/f'ERROR-{job["name"]}.json',dict(utc=now(),traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);a=p.parse_args();run(a.host)
