"""Read-only current-input identifiability audit on ordinary transition logs.

Only a0 is scored. All scored spans hold that SAME known forward input fixed;
the forward model actually receives it and predicts the observed outcome.
No future/optimal actions, policy targets, SGD, or planner proposals are created.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
from torch.utils.data import default_collate
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,tensor_sha,now,environment
from experiments.metaworld_transition_only_20261109.campaign import command
from experiments.metaworld_horizon_match_20261109.model import build

ROOT=BASE/'runs/metaworld_random_inverse_audit_20261109'
NEW=Path('/tmp/prescottsun-random-interventions-native128-20261109-v1')


def records(directory,split):
    arrays=[];rows=[]
    for shard in range(12):
        p=Path(directory)/str(shard);m=read(p/'manifest.json');assert m['passed'] and m['physical_stride']==1
        arrays.append(dict(pixels=np.load(p/'pixels.npy',mmap_mode='r'),inputs=np.load(p/'known_input.npy',mmap_mode='r')))
        rows.extend(dict(r,shard=shard) for r in m['records'] if r['split']==split)
    return arrays,rows


def prepare(source):
    ROOT.mkdir(exist_ok=False);old=read(BASE/'runs/metaworld_horizon_match_20261109/training/h60-seed1/config.json')['data'];pools={}
    for corpus,directory in [('ordinary_expert',old),('random_intervention',str(NEW))]:
        for split in ['train','test']:
            arrays,rs=records(directory,split);eligible=[]
            for r in rs:
                a=arrays[r['shard']]['inputs'];o=r['offset'];length=r['length']
                if length<6:continue
                if corpus=='random_intervention':times=[7]
                else:
                    inputs=np.asarray(a[o:o+length-1]);same=(inputs[1:]==inputs[:-1]).all(1).astype(np.int32)
                    times=np.flatnonzero(np.convolve(same,np.ones(4,np.int32),mode='valid')==4).tolist()
                eligible.extend(dict(record=r,time=int(t)) for t in times if t+5<length)
            count=len(eligible);rng=np.random.default_rng(1060001)
            if count>400:eligible=[eligible[int(i)] for i in rng.permutation(count)[:400]]
            assert eligible
            key=corpus+'-'+split;pools[key]=dict(directory=str(directory),rows=eligible,eligible_rows=count,selected_rows=len(eligible),corpus=corpus,split=split)
    jobs=[]
    for seed in [1,2]:
        for pool in pools:
            for span in [1,3,5]:
                index=len(jobs);name=f'{pool}-seed{seed}-span{span}'
                jobs.append(dict(name=name,seed=seed,pool=pool,span=span,host=['dev13','dev15','dev6'][index//8],gpu=index%8,output=str(ROOT/'reports'/(name+'.json'))))
    save(ROOT/'plan.json',dict(utc=now(),source=str(source),helper_sha256=sha(__file__),pools=pools,jobs=jobs,deadline=time.time()+3600,
        role='AuditONLY. TargetEXACTcurrentnativeinputa0alreadyfedforward;observedRGBt+span/heldSAMEa0. '
        'Span3/5OODfornative1trainedinverse:rawmagnitudedifferencesNOTunbiasedtime-calibratedmethodranking. '
        'Newgroups8identicalprestates/independentOURcontrolscontrastoutcomeinfoagainstcurrent-image-onlyablation. '
        'Oldvsnewalsohasdatasetshift,NOTuniquecausalexpert-shortcutproof. '
        'Allreportsprotected:NEVERtrain/teacher/proposals. No actionpredictionvectorsareexported.',optimizer_updates=0,physics_steps=0))


def metrics(pred,target):
    error=(pred-target)**2;bounded=(pred.clip(-1,1)-target)**2
    norm=np.linalg.norm(pred[:,:3],axis=1)*np.linalg.norm(target[:,:3],axis=1);valid=norm>1e-8
    cosine=float(((pred[:,:3]*target[:,:3]).sum(1)[valid]/norm[valid]).mean()) if valid.any() else None
    return dict(MSE=float(error.mean()),XYZ_MSE=float(error[:,:3].mean()),gripper_MSE=float(error[:,3].mean()),
        bounded_MSE=float(bounded.mean()),XYZ_direction_cosine=cosine)


@torch.no_grad()
def run(name):
    p=read(ROOT/'plan.json');j=next(j for j in p['jobs'] if j['name']==name);pool=p['pools'][j['pool']]
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    cfg=read(BASE/f'runs/metaworld_horizon_match_20261109/training/h60-seed{j["seed"]}/config.json')
    weights=Path(cfg['output'])/'model-step0012000/weights.pt';assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256']
    model=build(cfg).cuda().eval().requires_grad_(False);model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True),strict=True);before=tensor_sha(model)
    arrays,_=records(pool['directory'],pool['split']);predictions=[];erased=[];targets=[];prehashes=[];forward=[];copy=[];span=j['span']
    for start in range(0,len(pool['rows']),64):
        rows=[]
        for row in pool['rows'][start:start+64]:
            r=row['record'];t=row['time'];s=arrays[r['shard']];o=r['offset'];a0=np.asarray(s['inputs'][o+t]).copy()
            assert np.array_equal(np.asarray(s['inputs'][o+t:o+t+5]),np.broadcast_to(a0,(5,4)))
            def image(i):return torch.from_numpy(np.asarray(s['pixels'][o+i]).copy()).permute(2,0,1)
            indices=list(range(t-7,t+1));current=image(t);prehashes.append(hashlib.sha256(current.numpy().tobytes()).hexdigest())
            rows.append(dict(history_pixels=torch.stack([image(max(0,i)) for i in indices]),
                history_actions=torch.stack([torch.from_numpy(np.asarray(s['inputs'][o+i]).copy()).float() if i>=0 else torch.zeros(4) for i in indices[:-1]]),
                history_valid=torch.tensor([i>=0 for i in indices]),outcome_pixels=image(t+span),known_input_a0=torch.from_numpy(a0).float()))
        batch={k:v.cuda() for k,v in default_collate(rows).items()};hidden,current=model.prepare(batch);outcome=model.pixels(batch['outcome_pixels']);a0=batch['known_input_a0']
        f=model.vision(current);g=model.vision(outcome);pred=model.inverse(torch.cat([f,g,g-f],1));state_only=model.inverse(torch.cat([f,f,torch.zeros_like(f)],1))
        image=current
        for _ in range(span):hidden,image=model.predict(hidden,image,a0)
        predictions.append(pred.cpu().numpy());erased.append(state_only.cpu().numpy());targets.append(a0.cpu().numpy())
        forward.extend((image-outcome).square().mean((1,2,3)).cpu().tolist());copy.extend((current-outcome).square().mean((1,2,3)).cpu().tolist())
    pred=np.concatenate(predictions);state=np.concatenate(erased);target=np.concatenate(targets);groups=collections.defaultdict(list)
    for i,h in enumerate(prehashes):groups[h].append(i)
    within=[]
    for indices in groups.values():
        if len(indices)>1:
            a=target[indices];within.extend(((a-a.mean(0))**2).mean(1).tolist())
    assert before==tensor_sha(model)
    save(j['output'],dict(passed=True,utc=now(),job=j,rows=len(target),eligible_pool_rows=pool['eligible_rows'],
        actual_outcome_inverse=metrics(pred,target),future_erased_inverse=metrics(state,target),zero_action_baseline=metrics(np.zeros_like(target),target),
        duplicate_prestate_groups=sum(len(v)>1 for v in groups.values()),duplicate_prestate_sizes=sorted(len(v) for v in groups.values() if len(v)>1),
        empirical_within_identical_prestate_action_variance=None if not within else float(np.mean(within)),
        known_a0_forward_forecast_MSE=float(np.mean(forward)),copy_MSE=float(np.mean(copy)),
        same_known_a0_actually_fed_to_forward_for_all_span_steps=True,all5logged_inputs_equal_a0_verified=True,
        inverse_target_time='CURRENTa0 ONLY,never a1/unknownoptimal',weights_sha256=sha(weights),model_before=before,model_after=before,
        optimizer_updates=0,physics_steps=0,protected_audit_only=True,training_teacher_proposal_eligible=False,scope=p['role']))


def work(host,gpu):
    p=read(ROOT/'plan.json');j=next(j for j in p['jobs'] if j['host']==host and j['gpu']==gpu);source=Path(p['source'])
    try:command([PYTHON,str(Path(__file__).resolve()),'run','--name',j['name']],Path(j['output']).with_suffix('.log'),environment(source,gpu),source,p['deadline'],cap=1800)
    except BaseException:
        save(ROOT/'errors'/(j['name']+'.json'),dict(utc=now(),traceback=traceback.format_exc()));raise


def launch(host):
    p=read(ROOT/'plan.json');source=Path(p['source']);workers=[]
    for gpu in range(8):
        with (ROOT/f'worker-{host}-{gpu}.log').open('x') as f:
            c=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'work','--host',host,'--gpu',str(gpu)],cwd=source,
                env=environment(source,gpu)|{'CUDA_VISIBLE_DEVICES':''},stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(gpu=gpu,pid=c.pid))
    save(ROOT/f'launch-{host}.json',dict(utc=now(),host=host,workers=workers));print(host,workers,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','work','launch']);p.add_argument('--source',type=Path);p.add_argument('--name');p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.source)
    elif a.mode=='run':run(a.name)
    elif a.mode=='work':work(a.host,a.gpu)
    else:launch(a.host)
