"""Real-data supervision, sample independence, padding, and restore audits."""
import argparse
import hashlib
from pathlib import Path
import random
import math
import numpy as np
import torch
from torch.utils.data import default_collate
from .common import read,save,tensor_sha,sha,now
from .data import FactualWindows,WindowSampler
from .model import FactualRecurrent
from .train import configure,build,optimizers,restore,move,gradients


def calibration(cache,seed,output):
    cfg=dict(seed=seed,scratch=True,supervision_contract='factual_forward_and_same_input_inverse')
    configure(cfg);data=FactualWindows(cache)
    ids=np.random.default_rng(91019).choice(len(data),16,replace=False).tolist()
    batch=move(default_collate([data[i] for i in ids]),torch.device('cuda'))
    model=FactualRecurrent(horizon=15).cuda();before=tensor_sha(model,True)
    f,i=model.terms(batch);params=list(model.vision.parameters())
    gf=torch.autograd.grad(f.mean(),params,retain_graph=True)
    gi=torch.autograd.grad(i.mean(),params)
    nf=math.sqrt(sum(float(v.float().square().sum()) for v in gf));ni=math.sqrt(sum(float(v.float().square().sum()) for v in gi))
    assert math.isfinite(nf) and math.isfinite(ni) and nf>0 and ni>0
    weight=nf/ni
    assert before==tensor_sha(model,True)
    save(output,dict(passed=True,seed=seed,inverse_weight=weight,rule='One-time equality of initial shared-vision gradient norms,measured on the same16TRAINwindows atH15;fixed thereafter/alltasks/allhorizons.',
        forward_gradient_norm=nf,inverse_gradient_norm=ni,training_indices=ids,parameters_sha256=before,
        model_source_sha256=sha(Path(__file__).with_name('model.py')),optimizer_updates=0,utc=now()))
    print(dict(seed=seed,inverse_weight=weight,forward_norm=nf,inverse_norm=ni),flush=True)


def contract(cache,output):
    torch.set_num_threads(2);data=FactualWindows(cache);held=FactualWindows(cache,'test')
    assert len(data.episode_ids)==2250 and len(held.episode_ids)==250
    assert not set(data.episode_ids)&set(held.episode_ids)
    ids=np.random.default_rng(1239).choice(len(data),64,replace=False)
    for index in ids:
        r,t=data.locate(index);sample=data[index];source=data.shards[r['shard']];off=r['offset']
        assert set(sample)=={'history_pixels','history_actions','history_valid','future_pixels','actions','future_valid','episode','start'}
        assert np.array_equal(sample['history_pixels'][-1].permute(1,2,0).numpy(),source['pixels'][off+t])
        for j in range(15):
            if sample['future_valid'][j]:
                assert np.array_equal(sample['future_pixels'][j].permute(1,2,0).numpy(),source['pixels'][off+t+j+1])
                assert np.array_equal(sample['actions'][j].numpy(),source['inputs'][off+t+j])
            else:assert not sample['actions'][j].any()
    # Include the first and last eligible start of an episode to exercise masks.
    batch=default_collate([data[0],data[int(data.ends[0])-1]])
    torch.manual_seed(47);model=FactualRecurrent(horizon=15,inverse_weight=1.).eval()
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    with torch.no_grad():
        f,i=model.terms(batch)
        singles=[model.terms({k:v[j:j+1] for k,v in batch.items()}) for j in range(2)]
        assert torch.allclose(f,torch.cat([x[0] for x in singles]),rtol=1e-5,atol=1e-7)
        assert torch.allclose(i,torch.cat([x[1] for x in singles]),rtol=1e-5,atol=1e-7)
        poison={k:v.clone() for k,v in batch.items()}
        invalid=~poison['future_valid'];poison['future_pixels'][invalid]=219;poison['actions'][invalid]=-.91
        invalid_history=~poison['history_valid'];poison['history_pixels'][invalid_history]=173
        poison['history_actions'][~poison['history_valid'][:,:-1]]=.83
        fp,ip=model.terms(poison)
        assert torch.equal(f,fp) and torch.equal(i,ip),'Padding affected factual loss'
        try:model.terms(dict(batch,goal_pixels=torch.zeros(2,3,128,128)))
        except AssertionError:pass
        else:raise AssertionError('Goal entered training')
    # Sampler resume is independent of dataloader history.
    full=list(WindowSampler(data,0,19,8));tail=list(WindowSampler(data,24,19,8))
    assert full[24:128]==tail[:104]
    save(output,dict(passed=True,train_episodes=2250,test_episodes=250,train_windows=len(data),test_windows=len(held),
        real_native_transition_windows_checked=64,all_native_starts_included=True,sample_independence=True,
        future_and_history_padding_poison_invariant=True,training_goal_field_rejected=True,
        sampler_resume_exact=True,no_batchnorm=True,model_updates=0,gpu_contexts=0,utc=now(),
        source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['model.py','data.py','train.py','audit.py']}))
    print('Contract PASS',len(data),len(held),flush=True)


def checkpoint(config,path,output):
    cfg=read(config);configure(cfg);device=torch.device('cpu');model=build(cfg);opt,scheduler=optimizers(model,cfg)
    step,samples=restore(path,model,opt,scheduler,cfg,device,restore_rng=False);before=tensor_sha(model)
    dataset=FactualWindows(cfg['data'],'test');batch=default_collate([dataset[i] for i in [0,17]])
    loss,metrics=model.loss(batch);assert torch.isfinite(loss);loss.backward();g=gradients(model)
    required=['vision','observer','control','transition','decoder','rgb']+(['inverse'] if cfg['inverse_weight'] else [])
    assert all(math.isfinite(g[k]) and g[k]>0 for k in required),g
    assert before==tensor_sha(model)
    assert len(opt.state)>0 and step*cfg['batch_size']==samples
    save(output,dict(passed=True,step=step,samples=samples,cpu_full_model_optimizer_scheduler_restore=True,
        positive_fp32_heldout_gradients=g,weights_unchanged=True,weights_sha256=sha(path.parent/f'model-step{step:07d}/weights.pt'),
        optimizer_updates=0,gpu_contexts=0,utc=now()))
    print('Checkpoint PASS',step,g,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['contract','calibrate','checkpoint']);p.add_argument('--cache',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int);p.add_argument('--config',type=Path);p.add_argument('--checkpoint',type=Path)
    a=p.parse_args()
    if a.mode=='contract':contract(a.cache,a.output)
    elif a.mode=='calibrate':calibration(a.cache,a.seed,a.output)
    else:checkpoint(a.config,a.checkpoint,a.output)
