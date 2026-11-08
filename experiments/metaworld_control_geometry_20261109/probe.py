"""TRAIN-only scalar uncertainty, then four frozen eight-case planning probes."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
from torch.utils.data import default_collate
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,tensor_sha,now,environment
from experiments.metaworld_transition_only_20261109.data import FactualWindows
from experiments.metaworld_transition_only_20261109.campaign import command,HARD
from experiments.metaworld_horizon_match_20261109.model import build
from .metric import cpu_audit

ROOT=BASE/'runs/metaworld_control_geometry_20261109'
MATCHED=BASE/'runs/metaworld_horizon_match_20261109'


def prepare(source):
    audit=cpu_audit();ROOT.mkdir(exist_ok=False);save(ROOT/'CPU_AUDIT.json',dict(audit,utc=now()))
    old=read(MATCHED/'extended-controller/plan.json');jobs=[]
    for j,(host,gpu) in zip(old['jobs'],[('dev6',0),('dev6',2),('dev13',1),('dev15',1)]):
        reference=Path(j['output'])/'metrics.json';assert read(reference)['passed']
        jobs.append(dict(name=j['name'],host=host,gpu=gpu,config=j['config'],reference=str(reference)))
    save(ROOT/'plan.json',dict(utc=now(),source=str(source),jobs=jobs,deadline=time.time()+7200,
        manifest=str(BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json'),
        technical=str(BASE/'runs/metaworld_transition_only_20261109/manifests/technical.json'),
        source_hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},
        formal_episodes=32,model_updates=0,automatic_expansion=False,automatic_long_promotion=False,
        design='4 frozen H15/H60 factual12000 participants,2 original parents,8 same inspected physical cases/model. '
            'CEM64x4/H60/held5/execute15/500native and all learner parameters unchanged. Only terminal metric changes. '
            'Covariance from first64 own proposal forecasts per decision,NO goal/statereward/expert features in covariance. '
            'Freeze this covariance across all4 CEM iterations;no extra proposal rollouts. '
            'Isotropic floor from64 identical TRAIN-only validH60 windows/model;not calibrated on evaluation/heldout cases. '
            'Same Gaussian first-bank/action parity verified before changed metric. Reused completed white physical baselines.',
        evidence='Exact Push action audit: better native object progress can have worse true RGB-goal MSE than an unmoved object. '
            'Perfect-physics RGB CEM still fails Door/Push, while Drawer succeeds. Predictive accuracy alone is insufficient. '
            'This tests control geometry rather than fixed spatial smoothing;not a proven nonlinear metric or novelty/SOTA claim.',
        primary_prior='https://pmc.ncbi.nlm.nih.gov/articles/PMC4795843/'))


def calibrate(job,out):
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.30)
    cfg=read(job['config']);data=FactualWindows(cfg['data'],'train',maximum_horizon=60)
    eligible=[];offset=0
    for record in data.records:
        eligible.extend(range(offset,offset+max(0,record['length']-60)))
        offset+=record['length']-1
    assert offset==len(data) and len(eligible)>=64
    indices=np.random.default_rng(20261109).choice(eligible,size=64,replace=False)
    model=build(cfg).cuda().eval().requires_grad_(False);weights=Path(cfg['output'])/'model-step0012000/weights.pt'
    assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256']
    model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True));before=tensor_sha(model)
    errors=[];identities=[]
    with torch.inference_mode():
        for start in range(0,64,8):
            batch=default_collate([data[int(i)] for i in indices[start:start+8]])
            assert batch['future_valid'][:,59].all()
            identities.extend(zip(batch['episode'].tolist(),batch['start'].tolist()))
            batch={k:v.cuda() for k,v in batch.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                hidden,image=model.prepare(batch)
                for t in range(60):hidden,image=model.predict(hidden,image,batch['actions'][:,t])
            target=model.pixels(batch['future_pixels'][:,59])
            errors.extend((image-target).square().mean((1,2,3)).cpu().tolist())
    assert tensor_sha(model)==before
    import hashlib,json
    variance=max(float(np.mean(errors)),1/(12*255**2))
    save(out/'CALIBRATION.json',dict(passed=True,utc=now(),variance=variance,train_residual_mse=errors,
        identities=identities,manifest_sha256=hashlib.sha256(json.dumps(identities).encode()).hexdigest(),
        split='train',windows=64,each_target60actually_observed=True,model_unchanged=True,model_state_sha256=before,
        goal_or_evaluation_data_used=False,expert_action_policy_or_proposal_statistics=False,model_updates=0))


def work(name):
    plan=read(ROOT/'plan.json');j=next(j for j in plan['jobs'] if j['name']==name);source=Path(plan['source'])
    out=ROOT/name;out.mkdir(exist_ok=False);env=environment(source,j['gpu'])
    try:
        command([PYTHON,'-m','experiments.metaworld_control_geometry_20261109.probe','calibrate','--name',name],
            out/'calibrate.log',env,source,plan['deadline'],cap=1800)
        reference=read(j['reference']);bycase={(r['task'],r['seed']):r for r in reference['episodes']}
        def call(kind,manifest,dest,budget):
            command([PYTHON,'-m','experiments.metaworld_control_geometry_20261109.evaluate','--config',j['config'],
                '--manifest',manifest,'--output',str(dest),'--calibration',str(out/'CALIBRATION.json'),
                '--kind',kind,'--budget',str(budget)],out/(kind+'.log'),env,source,plan['deadline'],cap=7200)
        tech=out/'technical';call('white',plan['technical'],tech,20)
        for row in read(tech/'metrics.json')['episodes']:
            old=bycase[(row['task'],row['seed'])];assert sha(old['trace'])==old['trace_sha256']
            with np.load(old['trace']) as a,np.load(row['trace']) as b:
                assert np.array_equal(a['actions'][:len(b['actions'])],b['actions']) and np.array_equal(a['successes'][:len(b['successes'])],b['successes'])
            assert row['planning'][0]['first_bank_prefix_sha256']==old['planning'][0]['first_bank_prefix_sha256']
        save(out/'PREFLIGHT.json',dict(passed=True,utc=now(),standard_actual_actions_success_bank_exact=True,model_updates=0))
        dest=out/'intrinsic';call('intrinsic',plan['manifest'],dest,500)
        result=read(dest/'metrics.json');assert result['passed']
        assert result['model_state_before']==result['model_state_after']==reference['model_state_before']
        pairs=[]
        for row in result['episodes']:
            old=bycase[(row['task'],row['seed'])];assert row['case_sha256']==old['case_sha256']
            assert sha(row['trace'])==row['trace_sha256']
            with np.load(row['trace']) as z:
                assert z['actions'].shape==(row['steps'],4) and bool(z['successes'].any())==row['success']
                assert np.isfinite(z['actions']).all() and np.abs(z['actions']).max()<=1
            assert row['planning'][0]['first_bank_prefix_sha256']==old['planning'][0]['first_bank_prefix_sha256']
            pairs.append(dict(task=row['task'],seed=row['seed'],case_sha256=row['case_sha256'],pixel_success=old['success'],intrinsic_success=row['success']))
        hard=[r for r in pairs if r['task'] in HARD]
        save(out/'DONE.json',dict(passed=True,utc=now(),pairs=pairs,hard=dict(episodes=len(hard),
            pixel_successes=sum(r['pixel_success'] for r in hard),intrinsic_successes=sum(r['intrinsic_success'] for r in hard)),
            model_updates=0,all_actual_traces_verified=True,learned_parameters_unchanged=True))
    except BaseException:
        save(out/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise


def launch(host):
    p=read(ROOT/'plan.json');s=Path(p['source']);workers=[]
    for j in p['jobs']:
        if j['host']!=host:continue
        with (ROOT/(j['name']+'-worker.log')).open('x') as f:
            child=subprocess.Popen([PYTHON,'-m','experiments.metaworld_control_geometry_20261109.probe','work','--name',j['name']],cwd=s,
                env=dict(environment(s,j['gpu']),CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(j,pid=child.pid))
    save(ROOT/('launch-'+host+'.json'),dict(utc=now(),host=host,workers=workers));print(workers,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch','work','calibrate']);p.add_argument('--source',type=Path)
    p.add_argument('--host');p.add_argument('--name');a=p.parse_args()
    if a.mode=='prepare':prepare(a.source)
    elif a.mode=='launch':launch(a.host)
    elif a.mode=='work':work(a.name)
    else:
        job=next(j for j in read(ROOT/'plan.json')['jobs'] if j['name']==a.name);calibrate(job,ROOT/a.name)
