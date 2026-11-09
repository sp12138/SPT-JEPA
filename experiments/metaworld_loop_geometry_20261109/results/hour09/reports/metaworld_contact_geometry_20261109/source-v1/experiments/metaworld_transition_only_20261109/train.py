"""Single-GPU factual training with full-state resumable checkpoints."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
from .common import read, save, sha, tensor_sha, now
from .data import FactualWindows, WindowSampler
from .model import FactualRecurrent


def configure(cfg):
    assert cfg['scratch'] and cfg['supervision_contract'] == 'factual_forward_and_same_input_inverse'
    assert not any(cfg.get(k) for k in ['init_checkpoint','teacher_checkpoint','goal_path_data','initial_encoder_checkpoint'])
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(cfg['seed'])


def build(cfg):
    return FactualRecurrent(horizon=cfg['horizon'],width=cfg['width'],inverse_weight=cfg['inverse_weight'])


def optimizers(model,cfg):
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    warmup=300; total=cfg['lr_horizon']
    def schedule(step):
        if step < warmup:return max(1,step)/warmup
        return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/(total-warmup))))
    return opt,torch.optim.lr_scheduler.LambdaLR(opt,schedule)


def move(batch,device):
    return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


def save_checkpoint(root,model,opt,scheduler,step,samples,cfg):
    payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
        step=step,samples=samples,config=cfg,rng=dict(python=random.getstate(),numpy=np.random.get_state(),
        torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),source={n:sha(Path(__file__).with_name(n))
            for n in ['model.py','data.py','train.py']})
    path=root/f'checkpoint-{step:07d}.pt';temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(payload,temp); os.replace(temp,path)
    dest=root/f'model-step{step:07d}';dest.mkdir(exist_ok=True)
    temp=dest/f'weights.pt.{os.getpid()}.tmp';torch.save(model.state_dict(),temp);os.replace(temp,dest/'weights.pt')
    save(dest/'config.json',cfg)
    save(dest/'SEALED.json',dict(step=step,samples=samples,weights_sha256=sha(dest/'weights.pt'),
        checkpoint_sha256=sha(path),utc=now()))
    save(root/'last.json',dict(step=step,samples=samples,path=str(path),utc=now()))


def restore(path,model,opt,scheduler,cfg,device,restore_rng=True):
    value=torch.load(path,map_location='cpu',weights_only=False)
    for key in ['seed','horizon','width','inverse_weight','data','batch_size','lr_horizon','supervision_contract']:
        assert value['config'][key] == cfg[key],key
    assert value['samples'] == value['step']*cfg['batch_size']
    assert value['source'] == {n:sha(Path(__file__).with_name(n)) for n in value['source']}
    model.load_state_dict(value['model'],strict=True)
    opt.load_state_dict(value['optimizer']);scheduler.load_state_dict(value['scheduler'])
    if device.type=='cuda':
        for state in opt.state.values():
            for k,v in state.items():
                if torch.is_tensor(v) and k!='step':state[k]=v.to(device)
    if restore_rng:
        rng=value['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(rng['cuda'])
    return value['step'],value['samples']


def gradients(model):
    result={}
    for name in ['vision','observer','control','transition','decoder','rgb','inverse']:
        params=getattr(model,name).parameters()
        total=sum(float(p.grad.detach().float().square().sum()) for p in params if p.grad is not None)
        result[name]=math.sqrt(total)
    return result


def train(cfg,stop,resume=None):
    configure(cfg);root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda');model=build(cfg).to(device);opt,scheduler=optimizers(model,cfg)
    step=samples=0
    if resume:
        step,samples=restore(resume,model,opt,scheduler,cfg,device)
        save(root/f'resume-{step:07d}-{os.getpid()}.json',dict(passed=True,step=step,samples=samples,
            source=str(resume),cpu_full_model_optimizer_scheduler_restore=True,rng_restored=True,utc=now()))
    else:
        assert not (root/'initial.json').exists()
        save(root/'initial.json',dict(passed=True,all_parameters_random=True,external_weights_loaded=False,
            goal_path_supervised_ancestry=False,parameters_sha256=tensor_sha(model,True),
            parameters=sum(p.numel() for p in model.parameters()),seed=cfg['seed'],utc=now()))
        save_checkpoint(root,model,opt,scheduler,0,0,cfg)
    assert step < stop <= cfg['short_stop']
    dataset=FactualWindows(cfg['data']);start_time=time.monotonic();previous_time=start_time;previous_step=step
    sampler=WindowSampler(dataset,samples,cfg['seed'],cfg['batch_size'])
    loader=DataLoader(dataset,batch_size=cfg['batch_size'],sampler=sampler,num_workers=0,
        pin_memory=True,drop_last=True,generator=torch.Generator().manual_seed(cfg['seed']+999))
    before_manifest=dict(episodes=dataset.episode_ids,starts=len(dataset),history=8,maximum_horizon=15)
    save(root/'data-contract.json',before_manifest)
    model.train()
    while step < stop:
        sampler.start=samples
        for batch in loader:
            if (root/'STOP').exists() or (root.parent.parent/'STOP').exists():raise RuntimeError('Explicit experiment STOP;no completion claim')
            indices=dict(episode=batch['episode'].tolist(),start=batch['start'].tolist())
            batch=move(batch,device);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=model.loss(batch)
            if not torch.isfinite(loss):raise FloatingPointError((step,float(loss)))
            loss.backward();grad=gradients(model) if step<3 or (step+1)%100==0 else None
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();scheduler.step();step+=1;samples+=cfg['batch_size']
            if step<=3 or step%100==0 or step==stop:
                t=time.monotonic();row=dict(step=step,samples=samples,utc=now(),pid=os.getpid(),
                    loss=float(loss.detach()),**{k:float(v.detach()) for k,v in metrics.items() if k!='loss'},
                    gradients=grad,gradient_norm_before_clip=float(norm),lr=opt.param_groups[0]['lr'],
                    seconds_per_step=(t-previous_time)/max(1,step-previous_step),
                    gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,source_sha256=sha(Path(__file__).with_name('model.py')))
                save(root/'status.json',row)
                with (root/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                if step<=3 or step%1000==0:
                    with (root/'sample-trace.jsonl').open('a') as stream:stream.write(json.dumps(dict(step=step,**indices))+'\n')
                previous_time=t;previous_step=step
            if step%500==0 or step==stop:save_checkpoint(root,model,opt,scheduler,step,samples,cfg)
            if step>=stop:break
        if time.monotonic()-start_time>cfg['max_train_seconds']:raise TimeoutError('Finite train stage limit')
    save(root/f'trained-{stop:07d}.json',dict(passed=True,step=step,samples=samples,utc=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--stop',required=True,type=int)
    p.add_argument('--resume',type=Path);a=p.parse_args();cfg=read(a.config)
    try:train(cfg,a.stop,a.resume)
    except BaseException:
        save(Path(cfg['output'])/f'ERROR-train-{a.stop:07d}-{os.getpid()}.json',dict(utc=now(),traceback=traceback.format_exc()));raise
