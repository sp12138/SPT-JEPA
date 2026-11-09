"""Paired joint-world updates with complete states and factual-only ancestry."""
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
from experiments.metaworld_endpoint_density_20261109.train import setup as old_setup
from .model import JointFlow,random_terms
from .data import EndpointWindows,add_endpoint


def sources():
    from . import model,evaluate,data
    return {str(Path(p).resolve()):sha(p) for p in [__file__,model.__file__,evaluate.__file__,data.__file__]}


def setup(cfg,cuda=True):
    origin=read(cfg['head_config']);device,parent,renderer,_,_=old_setup(origin,cuda)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if cuda:torch.cuda.set_per_process_memory_fraction(.85)
    assert sha(cfg['head_weights'])==cfg['head_sha256']
    renderer.load_state_dict(torch.load(cfg['head_weights'],map_location=device,weights_only=True),strict=True)
    torch.manual_seed(940000+cfg['seed'])
    model=JointFlow(parent,renderer,cfg['mode']).to(device)
    model.posterior_scale.requires_grad_(False)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-3)
    def schedule(step):return min(1.,(step+1)/100)*.5*(1+math.cos(math.pi*min(step,6000)/6000))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,schedule)
    return device,model,opt,scheduler


def gradients(model):
    groups=dict(vision=model.parent.vision,observer=model.parent.observer,control=model.parent.control,
        transition=model.parent.transition,renderer=model.renderer,prior_scale=model.prior_scale)
    
    values={}
    for name,module in groups.items():
        value=sum(float(p.grad.detach().float().square().sum()) for p in module.parameters() if p.grad is not None)**.5
        assert math.isfinite(value) and value>0,(name,value)
        values[name]=value
    assert all(p.grad is None for module in [model.parent.decoder,model.parent.rgb,model.parent.inverse] for p in module.parameters())
    return values


def checkpoint(cfg,model,opt,scheduler,step,samples):
    out=Path(cfg['output']);payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
        step=step,samples=samples,config=cfg,source=sources(),
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
    path=out/f'checkpoint-{step:07d}.pt';temp=path.with_name(path.name+f'.{os.getpid()}.tmp');torch.save(payload,temp);os.replace(temp,path)
    portable=out/f'model-step{step:07d}';portable.mkdir(exist_ok=True)
    temp=portable/f'weights.pt.{os.getpid()}.tmp';torch.save(model.state_dict(),temp);os.replace(temp,portable/'weights.pt')
    save(portable/'SEALED.json',dict(step=step,samples=samples,weights_sha256=sha(portable/'weights.pt'),checkpoint_sha256=sha(path),source=sources(),utc=now()))
    save(out/'last.json',dict(step=step,samples=samples,path=str(path),utc=now()))


def restore(cfg,model,opt,scheduler,path,device,rng=True):
    d=torch.load(path,map_location='cpu',weights_only=False)
    assert d['source']==sources() and d['config']==cfg and d['samples']==64*d['step']
    model.load_state_dict(d['model'],strict=True);opt.load_state_dict(d['optimizer']);scheduler.load_state_dict(d['scheduler'])
    if device.type=='cuda':
        for state in opt.state.values():
            for k,v in state.items():
                if torch.is_tensor(v) and k!='step':state[k]=v.to(device)
    if rng:
        r=d['rng'];random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(r['cuda'])
    return d['step'],d['samples']


def train(cfg,stop,resume=None):
    device,model,opt,scheduler=setup(cfg);out=Path(cfg['output']);step=samples=0
    if resume:step,samples=restore(cfg,model,opt,scheduler,resume,device)
    else:
        assert not (out/'ORIGIN.json').exists()
        save(out/'ORIGIN.json',dict(utc=now(),initial_parameters_sha256=tensor_sha(model,True),
            parent_weights_sha256=read(cfg['head_config'])['parent_weights_sha256'],head_sha256=sha(cfg['head_weights']),
            parent_updates=12000,head_updates=2000,new_optimizer_intentional=True,
            pretrained_external_or_policy_or_goalpath=False,loss='Sample-leveljointendpointflowmatching,twoRGBnoisequeries. Pure-noise,uniform,highnoiseodds4. No posteriorlatent/goal/actiontargets.',source=sources()))
        checkpoint(cfg,model,opt,scheduler,0,0)
    assert step<stop<=2000
    data=EndpointWindows(cfg['data'],'train',maximum_horizon=60);sampler=WindowSampler(data,samples,cfg['seed'],64)
    loader=DataLoader(data,batch_size=64,sampler=sampler,num_workers=0,pin_memory=True,drop_last=True,
        generator=torch.Generator().manual_seed(950000+cfg['seed']))
    save(out/'DATA_CONTRACT.json',dict(train_episodes=data.episode_ids,train_starts=len(data),history=8,maximum_horizon=60,
        all_native_starts_retained=True,supervision='ActualfutureRGB andknownforwardcontrols only;NOgoal/path/policy/actiontargets.'))
    model.train()
    assert torch.cuda.device_count()==cfg['training_gpus']==4
    parallel=torch.nn.DataParallel(model,device_ids=list(range(4)))
    while step<stop:
        sampler.start=samples
        for batch in loader:
            if (out.parent.parent/'STOP').exists():raise RuntimeError('Explicit joint-flow pilot STOP')
            indices=dict(episode=batch['episode'].tolist(),start=batch['start'].tolist());batch={k:v.to(device) for k,v in batch.items()}
            terms=random_terms(batch);add_endpoint(data,batch,terms[0],device);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss,metrics=parallel(batch,terms);loss=loss.mean();metrics={k:v.mean() for k,v in metrics.items()}
            assert torch.isfinite(loss);loss.backward()
            g=gradients(model) if step<2 or (step+1)%100==0 else None
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            opt.step();scheduler.step();step+=1;samples+=64
            if step<=2:
                with (out/'sample-trace.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,**indices,
                    terms_sha256=[__import__('hashlib').sha256(t.cpu().numpy().tobytes()).hexdigest() for t in terms]))+'\n')
            if step<=2 or step%100==0 or step==stop:
                save(out/'status.json',dict(utc=now(),step=step,samples=samples,mode=cfg['mode'],loss=float(loss.detach()),
                    metrics={k:float(v.detach()) for k,v in metrics.items()},positive_gradients=g,
                    gradient_norm=float(norm),training_gpus=4,gpu_peak_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(4)]))
            if step%250==0 or step==stop:checkpoint(cfg,model,opt,scheduler,step,samples)
            if step>=stop:break
    save(out/f'TRAINED-{stop:07d}.json',dict(passed=True,utc=now(),step=step))


def audit(cfg,step):
    assert not torch.cuda.is_available();device,model,opt,scheduler=setup(cfg,False)
    s,n=restore(cfg,model,opt,scheduler,Path(cfg['output'])/f'checkpoint-{step:07d}.pt',device,False)
    before=tensor_sha(model);data=EndpointWindows(cfg['data'],'test',maximum_horizon=60)
    batch=default_collate([data[0],data[31]]);terms=random_terms(batch);add_endpoint(data,batch,terms[0],device)
    model.train();loss,metrics=model.objective(batch,terms);assert torch.isfinite(loss);loss.backward()
    g=gradients(model);assert tensor_sha(model)==before and len(opt.state)>0
    assert abs(float(model.kl(torch.zeros(2,128,16,16),torch.zeros(2,128,16,16),torch.zeros(2,128,16,16),torch.zeros(2,128,16,16)).max()))<1e-7
    save(Path(cfg['output'])/f'AUDIT-{step:07d}.json',dict(passed=True,utc=now(),step=s,samples=n,
        cpu_full_restore_fp32_backward=True,positive_gradients=g,model_unchanged=True,zero_identical_gaussian_KL=True,
        optimizer_updates=0,gpu_contexts=0,metrics={k:float(v.detach()) for k,v in metrics.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','audit']);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stop',type=int);p.add_argument('--resume',type=Path);p.add_argument('--step',type=int);a=p.parse_args();cfg=read(a.config)
    try:
        if a.mode=='train':train(cfg,a.stop,a.resume)
        else:audit(cfg,a.step)
    except BaseException:
        save(Path(cfg['output'])/f'ERROR-{a.mode}-{os.getpid()}.json',dict(utc=now(),traceback=traceback.format_exc()));raise
