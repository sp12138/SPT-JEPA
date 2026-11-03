"""Approve concrete technical proofs, freeze a finite queue, or launch one host."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from experiments.metaworld_eval_20261103.worker import digest, read, save, free_gpu


def approve(root, helper, source, base, plan):
    import torch
    recipes=read(plan)['recipes']; pairs=[]; configs={};proofs={}
    by_arm_seed={(r['arm'],r['seed']):Path(r['run']) for r in recipes}
    for recipe in recipes:
        run=Path(recipe['run']);cfg=read(run/'config.json');audit=read(run/'audit-step2.json')
        assert audit['passed'];assert audit['config_sha256']==recipe['config_sha256']==digest(run/'config.json')
        assert audit['checkpoint_sha256']==digest(run/'last.ckpt')
        runtime=read(run/'stage-runtime-source-0000002.json')
        assert runtime['target_step']==2 and runtime['CUDA_VISIBLE_DEVICES']==str(recipe['gpu'])
        assert str(source) in runtime['paths']['model']
        configs[run.name]=recipe['config_sha256'];proofs[str(run/'audit-step2.json')]=digest(run/'audit-step2.json')
    for seed in [1,2]:
        reference=torch.load(by_arm_seed['temporal-forecast',seed]/'initial-model/weights.pt',map_location='cpu',weights_only=True)
        for arm in ['as-jepa','forecast-render']:
            candidate=torch.load(by_arm_seed[arm,seed]/'initial-model/weights.pt',map_location='cpu',weights_only=True)
            assert candidate.keys()==reference.keys()
            different=[k for k in reference if not torch.equal(reference[k],candidate[k])]
            assert not different,(seed,arm,different)
            pairs.append(dict(seed=seed,reference='temporal-forecast',arm=arm,compared_tensors=len(reference),equal=True))
    names=['planning-preflight-paired-v2-tf-reach-v3','planning-preflight-paired-v2-tf-push-v3','planning-preflight-native-push-v3']
    smoke=[read(root/name/'metrics.json') for name in names]
    assert all(p['passed'] and p['model_state_before']==p['model_state_after'] for p in smoke)
    assert all(p['episodes'][0]['plan_calls']>=2 for p in smoke)
    assert len({p['episodes'][0]['first_bank_sha256'] for p in smoke})==1
    for name in names:proofs[str(root/name/'metrics.json')]=digest(root/name/'metrics.json')
    cases=read(root/'cases-v2/manifest.json');assert cases['passed'] and len(cases['rows'])==160
    data=base/'data/metaworld-lerobot-b46a1c18-packed-v1';assert read(data/'complete.json')['passed']
    deadline=time.time()+72*3600
    dispatch=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),deadline_unix=deadline,
                  deadline_utc=datetime.datetime.fromtimestamp(deadline,datetime.timezone.utc).isoformat(),
                  data=str(data),case_manifest=str(root/'cases-v2/manifest.json'),
                  helper=str(helper),source=str(source),base=str(base),plan_path=str(plan),
                  scope='Eight fixed trainings,8240-update cap;192 shared MPC endpoints;no retries,extensions,next research round or recurring LLM monitor.')
    jobs=[]
    for step in [2472,8240]:
        for task in sorted({c['task'] for c in cases['rows']}):
            for bank in (['primary'] if step==2472 else ['primary','fresh']):
                for recipe in recipes:
                    work=root/'evaluation'/f"step{step:07d}-{recipe['name']}-{task}-{bank}"
                    jobs.append(dict(name=work.name,run=recipe['run'],model=recipe['name'],arm=recipe['arm'],seed=recipe['seed'],
                                     step=step,checkpoint=str(Path(recipe['run'])/f'model-step{step:07d}'),
                                     task=task,bank=bank,limit=5 if step==2472 else 10,work=str(work)))
    assert len(jobs)==192
    save(root/'dispatch.json',dispatch)
    save(root/'evaluation-plan.json',dict(jobs=jobs,budget_native_steps=500,success_mode='any',
         episodes=1600,scope='Prespecified8tasks;all cases and failed goal constructions retained;not canonicalMT50 or publishedJEPA-WMs.'))
    save(root/'TRAINING_APPROVED.json',dict(passed=True,approved_utc=dispatch['created_utc'],config_sha256=configs,
         proofs=proofs,paired_initializations=pairs,planning_context2=True,planning_initial_banks_equal=True,
         case_manifest_sha256=digest(root/'cases-v2/manifest.json'),plan_sha256=digest(plan),
         scope='All8 actual native two-update/CPUrestore/GPUbackward checks and3 actual20-native-step planning checks pass. Technical only;no task performance claim.'))
    print(json.dumps(dict(passed=True,paired_initializations=pairs,jobs=len(jobs),deadline=dispatch['deadline_utc'])),flush=True)


def launch(root, host):
    dispatch=read(root/'dispatch.json');helper=Path(dispatch['helper']);source=Path(dispatch['source']);base=Path(dispatch['base'])
    assert read(root/'TRAINING_APPROVED.json')['passed'];assert host in ['dev2','dev13','dev15']
    specs=[]
    for recipe in read(dispatch['plan_path'])['recipes']:
        if recipe['host']==host:specs.append(dict(mode='train',gpu=recipe['gpu'],run=recipe['run'],name=recipe['name']))
    for gpu in (range(8) if host=='dev13' else range(4,8)):
        specs.append(dict(mode='evaluate',gpu=gpu,run=None,name=f'evaluation-{host}-g{gpu}'))
    assert len(specs)==8
    gpu_snapshot={g:free_gpu(g) for g in range(8)};jobs=[]
    for spec in specs:
        cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=60',str(max(1,int(dispatch['deadline_unix']-time.time()))),
             '/efs/prescottsun/project_jepa/envs/lewm/bin/python','-B','-u',str(helper/'experiments/metaworld_eval_20261103/worker.py'),
             '--root',str(root),'--mode',spec['mode'],'--source',str(source),'--helper',str(helper),'--base',str(base),
             '--gpu',str(spec['gpu']),'--host',host]
        if spec['run']:cmd.extend(['--run',spec['run']])
        logs=root/'workers';logs.mkdir(exist_ok=True)
        with (logs/(spec['name']+'.log')).open('x') as log:
            process=subprocess.Popen(cmd,cwd=helper,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append(dict(**spec,pid=process.pid,command=cmd))
    save(root/f'launch-{host}.json',dict(host=host,launched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_snapshot=gpu_snapshot,jobs=jobs))
    print(json.dumps(dict(host=host,jobs=[dict(name=j['name'],gpu=j['gpu'],pid=j['pid']) for j in jobs])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--helper');parser.add_argument('--source');parser.add_argument('--base',default='/efs/prescottsun/SPT-JEPA')
    parser.add_argument('--plan')
    parser.add_argument('--launch-host',choices=['dev2','dev13','dev15'])
    args=parser.parse_args()
    if args.launch_host:launch(Path(args.root),args.launch_host)
    else:approve(Path(args.root),Path(args.helper),Path(args.source),Path(args.base),Path(args.plan) if args.plan else Path(args.root)/'plan.json')
