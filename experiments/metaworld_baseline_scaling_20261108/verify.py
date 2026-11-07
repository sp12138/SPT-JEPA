"""CPU restoration/backward checks and a scalar100->300 LR boundary test.

Real native20epoch copies validate mechanics only; formal100epoch parents do
not yet exist. All model copies are discarded without optimizer updates.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from experiments.metaworld_baselines_20261107.models import PublicBaseline
from experiments.metaworld_baselines_20261107.train_native import (
    Windows,Batches,digest,equal,extended_config,extension_optimizer,gradients,save,
)


def schedule_test():
    from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR
    results=[]
    for warm in [.01,.1]:
        p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.AdamW([p],lr=.001)
        sch=LinearWarmupCosineAnnealingLR(opt,warmup_steps=max(1,int(100*warm)),max_steps=100)
        for i in range(100):
            opt.zero_grad();p.square().backward();opt.step();sch.step()
        assert abs(opt.param_groups[0]['lr'])<1e-15
        step=opt.state[p]['step'].clone();moment=opt.state[p]['exp_avg'].clone()
        opt.param_groups[0].update(lr=.001,initial_lr=.001)
        tail=LinearWarmupCosineAnnealingLR(opt,warmup_steps=max(1,int(200*warm)),max_steps=200)
        assert torch.equal(step,opt.state[p]['step']) and torch.equal(moment,opt.state[p]['exp_avg'])
        lrs=[]
        for i in range(200):
            opt.zero_grad();p.square().backward();opt.step();tail.step();lrs.append(opt.param_groups[0]['lr'])
            if i==2:
                resumed=torch.nn.Parameter(p.detach().clone());other=torch.optim.AdamW([resumed],lr=.001)
                other_sch=LinearWarmupCosineAnnealingLR(other,warmup_steps=max(1,int(300*warm)),max_steps=300)
                other.load_state_dict(copy.deepcopy(opt.state_dict()));other_sch.load_state_dict(copy.deepcopy(tail.state_dict()))
                assert other_sch.state_dict()==tail.state_dict()
            elif i>2:
                other.zero_grad();resumed.square().backward();other.step();other_sch.step()
                assert torch.equal(p,resumed) and equal(opt.state_dict(),other.state_dict())
                assert other_sch.state_dict()==tail.state_dict()
        assert opt.state[p]['step']==300 and max(lrs)>.00099 and lrs[0]>0 and abs(lrs[-1])<1e-15
        results.append(dict(native_warmup_fraction=warm,first_tail_lr=lrs[0],peak_lr=max(lrs),final_lr=lrs[-1],adam_steps=300,
                            resumed_at103_equals_uninterrupted_at_every_remaining_update=True))
    return results


def native_check(job,output):
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    step=job['epoch_steps']['20'];path=Path(job['output'])/f'checkpoint-{step:07d}.pt'
    state=torch.load(path,map_location='cpu',weights_only=False);cfg=extended_config(state['config'],output)
    assert state['step']==step and state['samples']==step*256
    model=PublicBaseline(cfg['method'],cfg['sources']);model.load_state_dict(state['model'],strict=True)
    assert equal(model.state_dict(),state['model'])
    assert [n for n,p in model.named_parameters()]==state['parameter_names']
    opt,sch,entry=extension_optimizer(model,state,cfg)
    assert all(float(s['step'])==float(state['optimizer']['state'][k]['step']) for k,s in opt.state_dict()['state'].items())
    del state,opt,sch
    dataset=Windows(cfg['data'],model.frames,cfg['seed'],'test')
    batch=torch.utils.data.default_collate([dataset[0],dataset[min(31,len(dataset)-1)]])
    model.eval();before=digest(model);loss,_=model(batch);assert torch.isfinite(loss);loss.backward()
    g=gradients(model);assert all(np.isfinite(v) and v>0 for v in g.values()),g
    assert digest(model)==before and not torch.cuda.is_initialized()
    count=cfg.get('natural_windows',105486);base=cfg['scaling_schedule']['start_step'];per=count//256
    assert base==100*per
    ranks=[list(Batches(count,cfg['seed'],base,base+2,i,2,128)) for i in [0,1]]
    actual=[ranks[0][i]+ranks[1][i] for i in [0,1]]
    order=torch.randperm(count,generator=torch.Generator().manual_seed(cfg['seed']+100*1000003))
    assert actual==[order[:256].tolist(),order[256:512].tolist()]
    report=dict(passed=True,method=job['method'],source_checkpoint=str(path),source_step=step,
        probe_scope='Actualnative20epoch copy,not a formal100->300continuation or new training result',
        cpu_full_model_adamw_scheduler_restore=True,required_positive_fp32_gradients=g,loss=float(loss.detach()),
        schedule_entry=entry,model_weights_unchanged=True,new_model_optimizer_updates=0,gpu_context_created=False,
        cumulative100epoch_sample_cursor_continuity=True,original_checkpoint_not_modified=True)
    save(output,report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.method:
        plan=json.loads(Path('/efs/prescottsun/SPT-JEPA/runs/metaworld_baselines_native_20261108/plan.json').read_text())
        job=next(j for j in plan['jobs'] if j['method']==args.method and j['seed']==1)
        result=native_check(job,args.output)
    else:
        result=dict(passed=True,scalar_schedule_tests=schedule_test(),scope='Syntheticscalar optimizer LR-cycle boundary test;not native performance.')
        save(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['required_positive_fp32_gradients','schedule_entry']}))
