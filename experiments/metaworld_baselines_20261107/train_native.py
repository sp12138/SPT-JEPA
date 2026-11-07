"""Finite DDP harness for native public losses, with full-state stage resumes."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from experiments.metaworld_baselines_20261107.models import PublicBaseline


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return value


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def digest(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(str((tuple(value.shape),value.dtype)).encode())
        h.update(value.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


class Windows(torch.utils.data.Dataset):
    def __init__(self,data,frames,seed,split='train'):
        from experiments.metaworld_baselines_20261107.models import load_file
        from as_jepa_tools.benchmark_data import BenchmarkWindows
        io=load_file('audited_local_cache',Path('/efs/prescottsun/SPT-JEPA/reports/metaworld_input_inverse_20261107/helper-io-v2/io_stage.py'))
        io.install_cache()
        self.base=BenchmarkWindows(data,window=frames,seed=seed,split=split)
        self.frames=frames

    def __len__(self):return len(self.base)

    def __getitem__(self,index):
        item=self.base[index]
        return dict(pixels=item['pixels'],action=item['action'])


class Batches:
    def __init__(self,length,seed,start,stop,rank,world,batch):
        self.length,self.seed,self.start,self.stop,self.rank,self.world,self.batch=length,seed,start,stop,rank,world,batch
        self.global_batch=world*batch
        self.per_epoch=length//self.global_batch

    def __len__(self):return self.stop-self.start

    def __iter__(self):
        epoch=-1
        for step in range(self.start,self.stop):
            e,k=divmod(step,self.per_epoch)
            if e!=epoch:
                generator=torch.Generator().manual_seed(self.seed+e*1000003)
                order=torch.randperm(self.length,generator=generator)
                epoch=e
            lo=k*self.global_batch+self.rank*self.batch
            yield order[lo:lo+self.batch].tolist()


def gradients(model):
    out={}
    for prefix in model.gradient_groups():
        values=[p.grad.detach().float().square().sum() for n,p in model.named_parameters()
            if n.startswith(prefix+'.') and p.grad is not None]
        out[prefix]=float(torch.stack(values).sum().sqrt()) if values else 0.
    return out


def optimizer(model,total):
    from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR
    groups,kwargs,warm=model.optimizer_spec()
    opt=torch.optim.AdamW(groups,**kwargs)
    sch=LinearWarmupCosineAnnealingLR(opt,warmup_steps=max(1,int(warm*total)),max_steps=total)
    return opt,sch


def extended_config(parent, output):
    """Keep the completed100epoch experiment intact; add one200epoch cycle."""
    base=parent['steps'];per=base//100
    assert base==per*100
    result=dict(parent,output=str(output),steps=base*3,epochs=300,
        milestones=sorted(set(parent['milestones']+[base+3,base+100,*[per*e for e in [150,200,250,300]]])),
        scaling_schedule=dict(start_step=base,additional_steps=2*base,
            kind='native_warmup_cosine_restart',peak_lr_scale=1.,version=1))
    return result


def extension_optimizer(model,state,cfg):
    """Restore AdamW exactly, then restart only the native LR schedule."""
    from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR
    opt,old=optimizer(model,state['config']['steps'])
    opt.load_state_dict(state['optimizer']);old.load_state_dict(state['scheduler'])
    assert equal(cpu(opt.state_dict()),state['optimizer']) and equal(old.state_dict(),state['scheduler'])
    before=[g['lr'] for g in opt.param_groups]
    pointers=[v.data_ptr() for s in opt.state.values() for v in s.values() if torch.is_tensor(v)]
    _,_,warm=model.optimizer_spec();duration=cfg['scaling_schedule']['additional_steps']
    for group,peak in zip(opt.param_groups,old.base_lrs):
        group['lr']=peak;group['initial_lr']=peak
    sch=LinearWarmupCosineAnnealingLR(opt,warmup_steps=max(1,int(warm*duration)),max_steps=duration)
    assert pointers==[v.data_ptr() for s in opt.state.values() for v in s.values() if torch.is_tensor(v)]
    return opt,sch,dict(parent_lrs=before,restart_lrs=[g['lr'] for g in opt.param_groups],
        native_peak_lrs=list(sch.base_lrs),native_warmup_fraction=warm,additional_steps=duration,
        actual_parent_adamw_and_scheduler_restore=True,adamw_moment_tensor_storage_preserved=True)


def checkpoint(model,opt,sch,step,out,cfg,rank,world):
    rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
    rngs=[None]*world
    dist.all_gather_object(rngs,rng)
    if rank==0:
        state=dict(model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),scheduler=sch.state_dict(),
            step=step,samples=step*cfg['global_batch'],rng_by_rank=rngs,config=cfg,
            parameter_names=[name for name,p in model.named_parameters()],utc=now())
        temp=out/f'checkpoint-{step:07d}.tmp'
        torch.save(state,temp)
        target=out/f'checkpoint-{step:07d}.pt'
        temp.replace(target)
        portable=out/f'model-step{step:07d}'
        portable.mkdir(exist_ok=False)
        # Same loader surface as the common physical evaluator; state is native.
        save(portable/'config.json',dict(_target_='experiments.metaworld_baselines_20261107.models.PublicBaseline',
            method=cfg['method'],sources=cfg['sources']))
        torch.save(state['model'],portable/'weights.pt')
        save(out/f'ready-{step:07d}.json',dict(passed=True,step=step,samples=state['samples'],
            checkpoint=str(target),portable=str(portable),model_state_sha256=digest(model),utc=now()))
        save(out/'latest.json',dict(step=step,checkpoint=str(target),portable=str(portable),utc=now()))
    dist.barrier()


def run(args):
    cfg=json.loads(Path(args.config).read_text())
    rank,world,local=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']),int(os.environ['LOCAL_RANK'])
    assert world==2 and cfg['global_batch']==256
    torch.cuda.set_device(local)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=20))
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    model=PublicBaseline(cfg['method'],cfg['sources'])
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    start=0;state=None;extension=None
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=False)
        expected=dict(state['config'])
        if args.allow_budget_extension:
            fork=json.loads((out/'FORK.json').read_text())
            assert Path(args.resume)==Path(fork['parent_checkpoint'])
            assert hashlib.sha256(Path(args.resume).read_bytes()).hexdigest()==fork['parent_checkpoint_sha256']
            assert state['step']==state['config']['steps']
            expected=extended_config(state['config'],cfg['output'])
        elif args.allow_output_transfer:
            fork=json.loads((out/'FORK.json').read_text())
            assert Path(args.resume)==Path(fork['parent_checkpoint'])
            assert hashlib.sha256(Path(args.resume).read_bytes()).hexdigest()==fork['parent_checkpoint_sha256']
            expected['output']=cfg['output']
        assert expected==cfg and state['samples']==state['step']*256
        model.load_state_dict(state['model'],strict=True)
        start=state['step']
    # H-JEPA's custom variable-length queue loader replaces its buffer using
    # empty_like(checkpoint_tensor). Restore on CPU BEFORE moving every buffer
    # to CUDA; otherwise its queue remains on CPU and NCCL rejects DDP restore.
    model=model.cuda()
    assert all(t.device.type=='cuda' for t in list(model.parameters())+list(model.buffers()))
    if args.allow_budget_extension:
        assert state is not None
        opt,sch,extension=extension_optimizer(model,state,cfg)
    else:opt,sch=optimizer(model,cfg['steps'])
    if state is not None and not args.allow_budget_extension:
        opt.load_state_dict(state['optimizer']);sch.load_state_dict(state['scheduler'])
    if rank==0:
        entry=out/f'entry-{start:07d}.json'
        if entry.exists():entry=out/f'entry-{start:07d}-attempt-{uuid.uuid4().hex}.json'
        save(entry,dict(method=cfg['method'],seed=cfg['seed'],start_step=start,
            random_initialization=(start==0),external_pretrained=False,parameters=sum(p.numel() for p in model.parameters()),
            model_state_sha256=digest(model),frames_per_window=model.frames,global_batch=256,local_batch=128,
            input_keys=['pixels','action'],source_config=OmegaPlain(model.cfg),
            runtime_train_file=str(Path(__file__).resolve()),runtime_train_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            all_model_buffers_cuda=True,utc=now()))
        if extension is not None:
            save(out/'BUDGET_EXTENSION_ENTRY.json',dict(extension,parent_step=start,sample_cursor=state['samples'],
                parent_checkpoint=args.resume,model_state_sha256=digest(model),completed_sgd_repeated=0,utc=now()))
    dataset=Windows(cfg['data'],model.frames,cfg['seed'])
    assert len(dataset)==cfg.get('natural_windows',105486)
    batches=Batches(len(dataset),cfg['seed'],start,args.stop,rank,world,128)
    loader=DataLoader(dataset,batch_sampler=batches,num_workers=4,persistent_workers=True,prefetch_factor=2,
        pin_memory=True,generator=torch.Generator().manual_seed(1777+cfg['seed']+rank))
    ddp=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=True,find_unused_parameters=False)
    if state is not None:
        rng=state['rng_by_rank'][rank]
        torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
        del state
    ddp.train();started=time.monotonic()
    for step,batch in enumerate(loader,start=start+1):
        batch={key:value.cuda(non_blocking=True) for key,value in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,metrics=ddp(batch,(step-1)//batches.per_epoch)
        assert torch.isfinite(loss),('nonfinite',step,float(loss))
        loss.backward()
        g=None
        entry_probe=bool(cfg.get('scaling_schedule')) and step<=start+3
        if step<=3 or step%100==0 or entry_probe:
            g=gradients(model)
            assert all(math.isfinite(x) for x in g.values()),(step,g)
            if step>=3:assert all(x>0 for x in g.values()),(step,'required gradient missing',g)
        # H-JEPA public recipe clips each level separately; all others globally.
        if cfg['method']=='h-jepa':
            norms=[torch.nn.utils.clip_grad_norm_(j.parameters(),1.) for j in model.core.jepas]
        else:norms=[torch.nn.utils.clip_grad_norm_(model.parameters(),1.)]
        assert all(torch.isfinite(n) for n in norms)
        opt.step();sch.step()
        if step<=3 or step%100==0 or entry_probe:
            row=dict(step=step,epoch_exposure=step/batches.per_epoch,loss=float(loss.detach()),
                metrics={k:float(v) for k,v in metrics.items()},gradients=g,rank=rank,world_size=world,
                lrs=[p['lr'] for p in opt.param_groups],elapsed_seconds=time.monotonic()-started,
                peak_memory_bytes=torch.cuda.max_memory_allocated(),utc=now())
            save(out/f'health-rank{rank}.json',row)
            with (out/f'learning-rank{rank}.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if step==args.stop or step in cfg['milestones'] or step%1000==0:
            checkpoint(model,opt,sch,step,out,cfg,rank,world)
    if rank==0:save(out/f'stage-{args.stop:07d}.json',dict(passed=True,initial_step=start,step=args.stop,utc=now()))
    dist.destroy_process_group()


def OmegaPlain(cfg):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(cfg,resolve=True)


def audit(args):
    cfg=json.loads(Path(args.config).read_text());out=Path(cfg['output'])
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    state=torch.load(args.resume,map_location='cpu',weights_only=False)
    model=PublicBaseline(cfg['method'],cfg['sources'])
    model.load_state_dict(state['model'],strict=True)
    assert equal(model.state_dict(),state['model'])
    opt,sch=optimizer(model,cfg['steps']);opt.load_state_dict(state['optimizer']);sch.load_state_dict(state['scheduler'])
    assert equal(opt.state_dict(),state['optimizer']) and equal(sch.state_dict(),state['scheduler'])
    assert [n for n,p in model.named_parameters()]==state['parameter_names']
    assert state['samples']==state['step']*256 and len(state['rng_by_rank'])==2
    step=state['step'];del state,opt,sch
    model=model.cuda().eval();before=digest(model)
    dataset=Windows(cfg['data'],model.frames,cfg['seed'],'test')
    data=torch.utils.data.default_collate([dataset[0],dataset[min(31,len(dataset)-1)]])
    data={k:v.cuda() for k,v in data.items()}
    rng=torch.get_rng_state(),torch.cuda.get_rng_state()
    loss,_=model(data)
    assert torch.isfinite(loss)
    loss.backward();g=gradients(model)
    assert all(math.isfinite(x) and x>0 for x in g.values()),g
    model.zero_grad(set_to_none=True)
    poisoned=dict(data,action=data['action'].clone());poisoned['action'][:,-1]=9876.
    torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1])
    with torch.no_grad():other,_=model(poisoned)
    assert torch.equal(loss.detach(),other),('padding target leaked',float(loss),float(other))
    assert digest(model)==before
    save(out/f'audit-{step:07d}.json',dict(passed=True,step=step,actual_cpu_model_optimizer_scheduler_restore=True,
        parameter_associations_checked=True,rng_rank_states=2,sample_cursor=step*256,heldout_fp32_backward=True,
        required_gradients=g,padding_target_poison_invariant=True,weights_unchanged=True,utc=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stop',type=int)
    p.add_argument('--resume');p.add_argument('--audit',action='store_true')
    p.add_argument('--allow-output-transfer',action='store_true')
    p.add_argument('--allow-budget-extension',action='store_true');args=p.parse_args()
    if args.audit:audit(args)
    else:run(args)
