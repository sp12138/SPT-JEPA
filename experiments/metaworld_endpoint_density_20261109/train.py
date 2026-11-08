"""Paired endpoint heads, frozen own factual parent, complete resumable states."""
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
from experiments.metaworld_transition_only_20261109.common import read,save,sha,tensor_sha,now
from experiments.metaworld_transition_only_20261109.data import FactualWindows,WindowSampler
from experiments.metaworld_horizon_match_20261109.model import build
from .model import EndpointRenderer,factual_condition,objective,advance


def sources():
    from . import model,energy
    return {str(Path(p).resolve()):sha(p) for p in [__file__,model.__file__,energy.__file__]}


def setup(cfg,cuda=True):
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(910000+cfg['seed'])
    device=torch.device('cuda' if cuda else 'cpu')
    if cuda:torch.cuda.set_per_process_memory_fraction(.30);torch.cuda.manual_seed_all(910000+cfg['seed'])
    parent=build(read(cfg['parent_config'])).to(device).eval().requires_grad_(False)
    assert sha(cfg['parent_weights'])==cfg['parent_weights_sha256']
    parent.load_state_dict(torch.load(cfg['parent_weights'],map_location=device,weights_only=True),strict=True)
    # Isolate head initialization from all other module allocations.
    torch.manual_seed(920000+cfg['seed'])
    renderer=EndpointRenderer().to(device)
    opt=torch.optim.AdamW(renderer.parameters(),lr=3e-4,weight_decay=1e-3)
    def schedule(step):return min(1.,(step+1)/100)*.5*(1+math.cos(math.pi*min(step,6000)/6000))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,schedule)
    return device,parent,renderer,opt,scheduler


def random_terms(batch):
    lengths=batch['future_valid'].sum(1)
    durations=(torch.rand(len(lengths),device=lengths.device)*lengths).floor().long()+1
    uniform=torch.rand(len(lengths),device=lengths.device).clamp(1e-5,1-1e-5)
    noise=torch.randn(len(lengths),3,128,128,device=lengths.device)
    return durations,uniform,noise


def move(batch,device):return {k:v.to(device) for k,v in batch.items()}


def gradients(renderer):
    values={}
    for name in ['time','duration','down0','down3','context','middle','up0','output']:
        params=getattr(renderer,name).parameters()
        value=sum(float(p.grad.detach().float().square().sum()) for p in params if p.grad is not None)**.5
        assert math.isfinite(value) and value>0,(name,value)
        values[name]=value
    return values


def checkpoint(cfg,parent,head,opt,scheduler,step,samples,parent_hash):
    assert tensor_sha(parent)==parent_hash
    out=Path(cfg['output']);payload=dict(head=head.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
        step=step,samples=samples,config=cfg,source=sources(),parent_state_sha256=parent_hash,
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
    path=out/f'checkpoint-{step:07d}.pt';temp=path.with_name(path.name+f'.{os.getpid()}.tmp');torch.save(payload,temp);os.replace(temp,path)
    weights=out/f'head-step{step:07d}';weights.mkdir(exist_ok=True)
    temp=weights/f'weights.pt.{os.getpid()}.tmp';torch.save(head.state_dict(),temp);os.replace(temp,weights/'weights.pt')
    save(weights/'SEALED.json',dict(step=step,samples=samples,weights_sha256=sha(weights/'weights.pt'),checkpoint_sha256=sha(path),
        parent_state_sha256=parent_hash,source=sources(),utc=now()))
    save(out/'last.json',dict(step=step,samples=samples,path=str(path),utc=now()))


def restore(cfg,parent,head,opt,scheduler,path,device,rng=True):
    d=torch.load(path,map_location='cpu',weights_only=False)
    assert d['source']==sources() and d['config']==cfg and d['samples']==64*d['step']
    assert tensor_sha(parent)==d['parent_state_sha256']
    head.load_state_dict(d['head'],strict=True);opt.load_state_dict(d['optimizer']);scheduler.load_state_dict(d['scheduler'])
    if device.type=='cuda':
        for state in opt.state.values():
            for k,v in state.items():
                if torch.is_tensor(v) and k!='step':state[k]=v.to(device)
    if rng:
        r=d['rng'];random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(r['cuda'])
    return d['step'],d['samples']


def train(cfg,stop,resume=None):
    device,parent,head,opt,scheduler=setup(cfg);out=Path(cfg['output']);parent_hash=tensor_sha(parent)
    step=samples=0
    if resume:step,samples=restore(cfg,parent,head,opt,scheduler,resume,device)
    else:
        assert not (out/'ORIGIN.json').exists()
        save(out/'ORIGIN.json',dict(utc=now(),head_initial_parameters_sha256=tensor_sha(head,True),
            parent_state_sha256=parent_hash,parent_weights_sha256=sha(cfg['parent_weights']),parent_original_updates=12000,
            parent_goal_path_or_policy_ancestry=False,parent_frozen=True,new_head_random_init=True,
            head_parameters=sum(p.numel() for p in head.parameters()),source=sources()))
        checkpoint(cfg,parent,head,opt,scheduler,0,0,parent_hash)
    assert step<stop<=2000
    data=FactualWindows(cfg['data'],'train',maximum_horizon=60);sampler=WindowSampler(data,samples,cfg['seed'],64)
    loader=DataLoader(data,batch_size=64,sampler=sampler,num_workers=0,pin_memory=True,drop_last=True,
        generator=torch.Generator().manual_seed(930000+cfg['seed']))
    save(out/'DATA_CONTRACT.json',dict(train_episodes=data.episode_ids,train_starts=len(data),history=8,maximum_horizon=60,
        all_native_starts_retained=True,supervision='ActualRGB/knownexecutedcontrols only;variable1..validfuture duration. No goals/path/policy/actiontargets.'))
    head.train()
    while step<stop:
        sampler.start=samples
        for batch in loader:
            if (out.parent.parent/'STOP').exists():raise RuntimeError('Explicit endpoint pilot STOP')
            indices=dict(episode=batch['episode'].tolist(),start=batch['start'].tolist());batch=move(batch,device)
            durations,uniform,noise=random_terms(batch);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                hidden,current,target=factual_condition(parent,batch,durations)
                loss,metrics=objective(head,hidden,current,target,durations,uniform,noise,cfg['mode'])
            assert torch.isfinite(loss);loss.backward();g=gradients(head) if step<2 or (step+1)%100==0 else None
            assert all(p.grad is None for p in parent.parameters())
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            opt.step();scheduler.step();step+=1;samples+=64
            if step<=2:
                with (out/'sample-trace.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,**indices,
                    durations=durations.cpu().tolist(),uniform_sha256=__import__('hashlib').sha256(uniform.cpu().numpy().tobytes()).hexdigest(),
                    noise_sha256=__import__('hashlib').sha256(noise.cpu().numpy().tobytes()).hexdigest()))+'\n')
            if step<=2 or step%100==0 or step==stop:
                save(out/'status.json',dict(utc=now(),step=step,samples=samples,mode=cfg['mode'],loss=float(loss.detach()),
                    metrics={k:float(v.detach()) for k,v in metrics.items()},positive_gradients=g,
                    gradient_norm=float(norm),parent_frozen=True,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30))
            if step%250==0 or step==stop:checkpoint(cfg,parent,head,opt,scheduler,step,samples,parent_hash)
            if step>=stop:break
    save(out/f'TRAINED-{stop:07d}.json',dict(passed=True,utc=now(),step=step,parent_frozen=True))


def audit(cfg,step):
    assert not torch.cuda.is_available()
    device,parent,head,opt,scheduler=setup(cfg,False)
    s,n=restore(cfg,parent,head,opt,scheduler,Path(cfg['output'])/f'checkpoint-{step:07d}.pt',device,False)
    parent_hash=tensor_sha(parent);before=tensor_sha(head)
    data=FactualWindows(cfg['data'],'test',maximum_horizon=60)
    batch=default_collate([data[0],data[31]]);durations,uniform,noise=random_terms(batch)
    hidden,current,target=factual_condition(parent,batch,durations)
    loss,_=objective(head,hidden,current,target,durations,uniform,noise,cfg['mode']);assert torch.isfinite(loss);loss.backward()
    g=gradients(head);assert tensor_sha(head)==before and tensor_sha(parent)==parent_hash and len(opt.state)>0
    # Exact hidden-operator parity, independent of the new renderer.
    h,x=parent.prepare(batch)
    direct,_=parent.predict(h,x,batch['actions'][:,0]);fast=advance(parent,h,batch['actions'][:,0])
    assert torch.equal(direct,fast)
    save(Path(cfg['output'])/f'AUDIT-{step:07d}.json',dict(passed=True,utc=now(),step=s,samples=n,
        cpu_full_restore_fp32_backward=True,positive_head_gradients=g,parent_unchanged=True,head_unchanged=True,
        exact_parent_hidden_operator=True,optimizer_updates=0,gpu_contexts=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','audit']);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stop',type=int);p.add_argument('--resume',type=Path);p.add_argument('--step',type=int);a=p.parse_args();cfg=read(a.config)
    try:
        if a.mode=='train':train(cfg,a.stop,a.resume)
        else:audit(cfg,a.step)
    except BaseException:
        save(Path(cfg['output'])/f'ERROR-{a.mode}-{os.getpid()}.json',dict(utc=now(),traceback=traceback.format_exc()));raise
