"""Matched 2000-update continuation from admissible factual-only parents."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader,default_collate
from experiments.metaworld_transition_only_20261109 import train as original
from experiments.metaworld_transition_only_20261109.common import read,save,sha,tensor_sha,now
from experiments.metaworld_transition_only_20261109.data import FactualWindows,WindowSampler
from .model import build


def sources():
    import experiments.metaworld_transition_only_20261109.model as base
    import experiments.metaworld_transition_only_20261109.data as data
    from . import model
    return {str(Path(p).resolve()):sha(p) for p in [__file__,model.__file__,base.__file__,data.__file__,original.__file__]}


def checkpoint(cfg,model,opt,scheduler,step,samples):
    out=Path(cfg['output']);payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
        step=step,samples=samples,config=cfg,source=sources(),rng=dict(python=random.getstate(),numpy=np.random.get_state(),
        torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
    dest=out/f'checkpoint-{step:07d}.pt';tmp=dest.with_name(dest.name+f'.{os.getpid()}.tmp')
    torch.save(payload,tmp);os.replace(tmp,dest)
    portable=out/f'model-step{step:07d}';portable.mkdir(exist_ok=True)
    tmp=portable/f'weights.pt.{os.getpid()}.tmp';torch.save(model.state_dict(),tmp);os.replace(tmp,portable/'weights.pt')
    save(portable/'config.json',cfg);save(portable/'SEALED.json',dict(step=step,samples=samples,
        weights_sha256=sha(portable/'weights.pt'),checkpoint_sha256=sha(dest),source=sources(),utc=now()))
    save(out/'last.json',dict(step=step,samples=samples,path=str(dest),utc=now()))


def load(cfg,model,opt,scheduler,path,device,restore_rng=True):
    p=torch.load(path,map_location='cpu',weights_only=False);assert p['source']==sources()
    for key in ['seed','horizon','width','inverse_weight','data','batch_size','lr_horizon','supervision_contract','parent_sha256']:
        assert p['config'][key]==cfg[key],key
    assert p['samples']==p['step']*cfg['batch_size'];model.load_state_dict(p['model'],strict=True)
    opt.load_state_dict(p['optimizer']);scheduler.load_state_dict(p['scheduler'])
    if device.type=='cuda':
        for state in opt.state.values():
            for k,v in state.items():
                if torch.is_tensor(v) and k!='step':state[k]=v.to(device)
    if restore_rng:
        r=p['rng'];random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(r['cuda'])
    return p['step'],p['samples']


def setup(cfg,cuda=True):
    original.configure(cfg);device=torch.device('cuda' if cuda else 'cpu')
    if cuda:torch.cuda.set_per_process_memory_fraction(.30)
    model=build(cfg).to(device);opt,scheduler=original.optimizers(model,cfg)
    return device,model,opt,scheduler


def initialize(cfg):
    device,model,opt,scheduler=setup(cfg);out=Path(cfg['output']);assert not (out/'ORIGIN.json').exists()
    parent=Path(cfg['parent']);assert sha(parent)==cfg['parent_sha256']
    parent_cfg=read(parent.parent/'config.json');assert parent_cfg['horizon']==15
    for k in ['seed','width','inverse_weight','data','batch_size','lr_horizon','scratch','supervision_contract']:
        assert cfg[k]==parent_cfg[k],k
    assert not read(parent.parent/'initial.json')['goal_path_supervised_ancestry']
    step,samples=original.restore(parent,model,opt,scheduler,parent_cfg,device)
    assert step==10000 and samples==640000
    save(out/'ORIGIN.json',dict(passed=True,parent=str(parent),parent_sha256=sha(parent),step=step,samples=samples,
        full_optimizer_scheduler_rng_restored=True,state_sha256=tensor_sha(model),parameters_sha256=tensor_sha(model,True),
        ancestry='Ownrandominit-factual-onlyH15inverseparent;no goal-query/external/policy weights',
        intentional_change='Futureloss horizon15or60;bothreceive same60futurearrays,allactualnativeanchors,history8,cursor640000',source=sources(),utc=now()))
    checkpoint(cfg,model,opt,scheduler,step,samples)


def audit(cfg,step):
    assert not torch.cuda.is_available();device,model,opt,scheduler=setup(cfg,False)
    path=Path(cfg['output'])/f'checkpoint-{step:07d}.pt';s,n=load(cfg,model,opt,scheduler,path,device,False)
    before=tensor_sha(model);data=FactualWindows(cfg['data'],'test',maximum_horizon=60)
    batch=default_collate([data[i] for i in [0,17]]);loss,_=model.loss(batch);assert torch.isfinite(loss);loss.backward()
    g=original.gradients(model);assert all(math.isfinite(v) and v>0 for v in g.values())
    assert tensor_sha(model)==before and len(opt.state)>0
    save(Path(cfg['output'])/f'audit-{step:07d}.json',dict(passed=True,step=s,samples=n,cpu_full_restore=True,
        positive_fp32_gradients=g,model_unchanged=True,optimizer_updates=0,gpu_contexts=0,utc=now()))


def train(cfg,stop,resume):
    device,model,opt,scheduler=setup(cfg);step,samples=load(cfg,model,opt,scheduler,resume,device)
    out=Path(cfg['output']);assert step<stop<=12000
    data=FactualWindows(cfg['data'],maximum_horizon=60);sampler=WindowSampler(data,samples,cfg['seed'],64)
    loader=DataLoader(data,batch_size=64,sampler=sampler,num_workers=0,pin_memory=True,drop_last=True,
        generator=torch.Generator().manual_seed(cfg['seed']+999))
    save(out/'data-contract.json',dict(episodes=data.episode_ids,starts=len(data),history=8,maximum_horizon=60))
    model.train()
    while step<stop:
        sampler.start=samples
        for batch in loader:
            if (out.parent.parent/'STOP').exists():raise RuntimeError('Explicit pilot STOP')
            indices=dict(episode=batch['episode'].tolist(),start=batch['start'].tolist())
            batch=original.move(batch,device);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=model.loss(batch)
            assert torch.isfinite(loss);loss.backward();g=original.gradients(model) if step<10003 or (step+1)%100==0 else None
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();scheduler.step();step+=1;samples+=64
            if step<=10003 or step%100==0 or step==stop:
                if g is not None:assert all(math.isfinite(v) and v>0 for v in g.values())
                row=dict(step=step,samples=samples,utc=now(),loss=float(loss.detach()),gradients=g,
                    gradient_norm_before_clip=float(norm),metrics={k:float(v.detach()) for k,v in metrics.items()},
                    gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,lr=opt.param_groups[0]['lr'])
                save(out/'status.json',row)
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step<=10002:
                with (out/'sample-trace.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,**indices))+'\n')
            if step%500==0 or step==stop:checkpoint(cfg,model,opt,scheduler,step,samples)
            if step>=stop:break
    save(out/f'trained-{stop:07d}.json',dict(passed=True,step=step,samples=samples,utc=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['init','train','audit']);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--step',type=int);p.add_argument('--stop',type=int);p.add_argument('--resume',type=Path);a=p.parse_args();cfg=read(a.config)
    try:
        if a.mode=='init':initialize(cfg)
        elif a.mode=='audit':audit(cfg,a.step)
        else:train(cfg,a.stop,a.resume)
    except BaseException:
        save(Path(cfg['output'])/f'ERROR-{a.mode}-{os.getpid()}.json',dict(utc=now(),traceback=traceback.format_exc()));raise
