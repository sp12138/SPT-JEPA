"""Four bounded, deliberately co-located continuations; old jobs are untouched."""
import argparse
import collections
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
from torch.utils.data import default_collate
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,tensor_sha,now,environment
from experiments.metaworld_transition_only_20261109.data import FactualWindows,WindowSampler
from experiments.metaworld_transition_only_20261109.train import build as oldbuild
from experiments.metaworld_transition_only_20261109.campaign import command,TASKS,HARD,HARD4
from .model import build

ROOT=BASE/'runs/metaworld_horizon_match_20261109'
OLD=BASE/'runs/metaworld_transition_only_20261109'


def contract(output):
    assert not torch.cuda.is_available();torch.set_num_threads(2)
    cfg=read(OLD/'training/h15-inverse1-seed1/config.json');state=torch.load(OLD/'training/h15-inverse1-seed1/model-step0010000/weights.pt',map_location='cpu',weights_only=True)
    old=oldbuild(cfg).eval();same=build(cfg).eval();long=build(dict(cfg,horizon=60)).eval()
    for m in [old,same,long]:m.load_state_dict(state,strict=True)
    assert len({tensor_sha(m) for m in [old,same,long]})==1
    a=FactualWindows(cfg['data'],'test');b=FactualWindows(cfg['data'],'test',maximum_horizon=60)
    for i in [0,17,481,1023]:
        x,y=a[i],b[i]
        for k in ['history_pixels','history_actions','history_valid','episode','start']:
            assert torch.equal(torch.as_tensor(x[k]),torch.as_tensor(y[k])),k
        for k in ['future_pixels','actions','future_valid']:assert torch.equal(x[k],y[k][:15]),k
    x=default_collate([a[i] for i in [0,17]]);y=default_collate([b[i] for i in [0,17]])
    l0,_=old.loss(x);l1,_=same.loss(y);assert torch.equal(l0,l1)
    l0.backward();l1.backward()
    for (n,p),(m,q) in zip(old.named_parameters(),same.named_parameters()):
        assert n==m and torch.equal(p.grad,q.grad),n
    with torch.no_grad():
        h0,i0=old.prepare(x);h1,i1=long.prepare(y)
        for t in range(60):
            controls=y['actions'][:,t];h0,i0=old.predict(h0,i0,controls);h1,i1=long.predict(h1,i1,controls)
            assert torch.equal(h0,h1) and torch.equal(i0,i1)
    s0=list(WindowSampler(a,640000,1,64))[:128];s1=list(WindowSampler(b,640000,1,64))[:128];assert s0==s1
    save(output,dict(passed=True,utc=now(),same_parent_parameters=True,h15_loss_and_all_gradients_bitexact=True,
        h15_vs_h60_constructor_same60step_inference_bitexact=True,real_window15prefix_exact=True,
        cursor640000_sampler_same=True,optimizer_updates=0,gpu_contexts=0,physics_steps=0,
        interpretation='Only loss horizon differs;longer targets are observed factual frames conditioned on supplied executed inputs,no desired-goal/path-policy training.'))
    print('CPU contract PASS',flush=True)


def prepare(source):
    assert read(BASE/'reports/metaworld_horizon_match_20261109/contract-v1.json')['passed']
    assert len(list(OLD.glob('training/*/TRAINING_COMPLETE.json')))==24
    ROOT.mkdir(exist_ok=False);jobs=[]
    for seed,host,gpus in [(1,'dev15',[2,4]),(2,'dev6',[0,5])]:
        parent=OLD/f'training/h15-inverse1-seed{seed}/checkpoint-0010000.pt';oldcfg=read(parent.parent/'config.json')
        for horizon,gpu in zip([15,60],gpus):
            name=f'h{horizon}-seed{seed}';out=ROOT/'training'/name;cfg=dict(oldcfg)
            cfg.update(name=name,output=str(out),source=str(source),horizon=horizon,short_stop=12000,
                maximum_horizon=60,parent=str(parent),parent_sha256=sha(parent),cumulative_parent_updates=10000,
                additional_short_updates=2000,co_located_with_old_frozen_evaluation=True,new_process_memory_fraction=.30)
            cfg.pop('expected_initial_parameters_sha256',None);save(out/'config.json',cfg)
            jobs.append(dict(name=name,host=host,gpu=gpu,seed=seed,horizon=horizon,config=str(out/'config.json'),output=str(out)))
    save(ROOT/'plan.json',dict(utc=now(),source=str(source),source_hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},
        jobs=jobs,deadline=time.time()+4*3600,manifest=str(OLD/'manifests/screen.json'),technical=str(OLD/'manifests/technical.json'),
        design='SameH15inverseparent/seed1,2 ->2000moreupdateswithH15controlvsH60treatment. Sameactualnativeanchors/futurearrays/dataorder/optimizer/LRhorizon30000. '
            'Wholeancestryrandominit/factual-only/samplelevel. Horizon60costsmorecompute;notFLOPmatched. FinalRGBgoalsONLYinplanning,notinverse/policytargets.',
        allocation='Fouradditionalpurposefulprocesses co-located on4of24H200s,oldfrozenworkuntouched. Newprocessmemorycap30percent;no old metadata paused/killed/adopted. '
            '24physicalGPUs,up to28CUDAcontexts,not28GPUs;GPUmemory/throughput inspected. No oldSGD repeated.',
        evaluation='8samepreviouslyinspectedphysicalcases/model xH30/e15andH60/e2,CEM64x4/500nativeanysuccess;64formalepisodes+4technical20step. '
            'twolearnedparentsrepeatphysicalcases,notlargeuntouchedconfirmation;no publicSOTA/novelty claim.',automatic_long_promotion=False))
    print('Prepared4matchedcontinuations',flush=True)


def gate():
    plan=read(ROOT/'plan.json')
    if not all((Path(j['output'])/'PREFLIGHT.json').exists() for j in plan['jobs']):return
    try:(ROOT/'GATE_LOCK').mkdir()
    except FileExistsError:return
    try:
        for seed in [1,2]:
            pair=[Path(j['output']) for j in plan['jobs'] if j['seed']==seed]
            assert len({read(p/'ORIGIN.json')['state_sha256'] for p in pair})==1
            assert len({(p/'sample-trace.jsonl').read_text() for p in pair})==1
            traces=[]
            for p in pair:
                d=read(p/'technical/metrics.json');assert d['passed'] and d['model_updates']==0
                e=d['episodes'][0]
                with np.load(e['trace']) as z:traces.append((z['actions'].copy(),z['successes'].copy()))
            assert np.array_equal(traces[0][0],traces[1][0]) and np.array_equal(traces[0][1],traces[1][1])
        save(ROOT/'TRAINING_APPROVED.json',dict(passed=True,utc=now(),paired_parent_states=True,actual20step_actions_success_bitexact=True,
            paired_actual_first2samples=True,all4actual2updates_and_fullCPUrestore=True))
    except BaseException:
        save(ROOT/'ERROR-gate.json',dict(utc=now(),traceback=traceback.format_exc()));raise


def work(name):
    plan=read(ROOT/'plan.json');j=next(j for j in plan['jobs'] if j['name']==name)
    out=Path(j['output']);source=Path(plan['source']);env=environment(source,j['gpu']);cpu=dict(env,CUDA_VISIBLE_DEVICES='')
    deadline=plan['deadline'];module='experiments.metaworld_horizon_match_20261109'
    def call(args,log,environment=env,cap=7200):
        command([PYTHON,'-m',module+'.'+args[0],*args[1:]],out/log,environment,source,deadline,cap=cap)
    try:
        raw=subprocess.check_output(['nvidia-smi','-i',str(j['gpu']),'--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
        total,used=map(float,raw.strip().split(','));assert total-used>90*1024,raw
        save(out/'COLOCATION_ENTRY.json',dict(utc=now(),host=j['host'],gpu=j['gpu'],memory_total_mib=total,memory_used_mib=used,
            new_process_memory_fraction=.30,old_processes_untouched=True))
        call(['train','init','--config',j['config']],'initialize.log')
        call(['evaluate','--config',j['config'],'--step','10000','--manifest',plan['technical'],'--output',out/'technical',
              '--horizon','60','--execute','2','--budget','20'],'technical.log')
        call(['train','train','--config',j['config'],'--stop','10002','--resume',out/'checkpoint-0010000.pt'],'train-entry.log')
        call(['train','audit','--config',j['config'],'--step','10002'],'audit-10002.log',cpu)
        save(out/'PREFLIGHT.json',dict(passed=True,utc=now(),actual_new_updates=2,cpu_full_restore=True,physical_steps=20))
        while not (ROOT/'TRAINING_APPROVED.json').exists():
            assert time.time()<deadline
            if list(ROOT.rglob('ERROR*.json')):raise RuntimeError('Peer preflight failure')
            gate();time.sleep(2)
        call(['train','train','--config',j['config'],'--stop','12000','--resume',out/'checkpoint-0010002.pt'],'train-12000.log')
        call(['train','audit','--config',j['config'],'--step','12000'],'audit-12000.log',cpu)
        for horizon,execute in [(30,15),(60,2)]:
            dest=ROOT/'evaluation'/f'{name}-H{horizon}-e{execute}'
            call(['evaluate','--config',j['config'],'--step','12000','--manifest',plan['manifest'],'--output',dest,
                '--horizon',str(horizon),'--execute',str(execute)],f'evaluate-H{horizon}-e{execute}.log')
            assert read(dest/'metrics.json')['passed']
        save(out/'DONE.json',dict(passed=True,utc=now(),cumulative_updates=12000,new_updates=2000))
    except BaseException:
        save(out/f'ERROR-worker-{os.getpid()}.json',dict(utc=now(),traceback=traceback.format_exc()));raise


def launch(host):
    p=read(ROOT/'plan.json');s=Path(p['source']);workers=[]
    for j in p['jobs']:
        if j['host']!=host:continue
        with (ROOT/f'worker-{j["name"]}.log').open('x') as f:
            child=subprocess.Popen([PYTHON,'-m','experiments.metaworld_horizon_match_20261109.campaign','work','--name',j['name']],
                cwd=s,env=dict(environment(s,j['gpu']),CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(j,pid=child.pid))
    save(ROOT/f'launch-{host}.json',dict(utc=now(),workers=workers,host=host));print(workers,flush=True)


def collect():
    p=read(ROOT/'plan.json')
    while time.time()<p['deadline']:
        groups=collections.defaultdict(list);endpoints=0
        for j in p['jobs']:
            for horizon,execute in [(30,15),(60,2)]:
                dest=ROOT/'evaluation'/f'{j["name"]}-H{horizon}-e{execute}'
                if not (dest/'metrics.json').exists():continue
                d=read(dest/'metrics.json');assert d['passed'] and d['model_state_before']==d['model_state_after'] and d['model_updates']==0
                expected={(r['task'],r['seed']):r['sha256'] for r in read(p['manifest'])['rows']}
                assert {(r['task'],r['seed']):r['case_sha256'] for r in d['episodes']}==expected
                for row in d['episodes']:
                    assert row['initial_rgb_replay']['passed'] and sha(row['trace'])==row['trace_sha256']
                    with np.load(row['trace']) as z:
                        assert z['actions'].shape==(row['steps'],4) and np.isfinite(z['actions']).all() and np.max(np.abs(z['actions']))<=1
                        assert bool(z['successes'].any())==row['success']
                groups[f'trainH{j["horizon"]}-planH{horizon}-execute{execute}'].extend(d['episodes']);endpoints+=1
        report={}
        for name,rows in groups.items():
            hard=[r for r in rows if r['task'] in HARD]
            report[name]=dict(episodes=len(rows),hard_successes=sum(r['success'] for r in hard),hard_episodes=len(hard),
                task_counts={t:dict(successes=sum(r['success'] for r in rows if r['task']==t),episodes=sum(r['task']==t for r in rows)) for t in TASKS})
        errors={str(f):read(f) for f in ROOT.rglob('ERROR*.json')};done=len(list(ROOT.glob('training/*/DONE.json')))
        save(ROOT/'results/summary.json',dict(utc=now(),done=done,models=4,endpoints=endpoints,expected_endpoints=8,groups=report,errors=errors,
            scope='Paireddevelopmental8physicalcases/2learnedparents,H15vsH60loss only;2000additionalupdates,bothcount10000parentupdates;noSOTA/equalFLOPsclaim.'))
        if done==4 and endpoints==8:
            save(ROOT/'CLOSED.json',dict(utc=now(),passed=not errors,automatic_long_promotion=False));return
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['contract','prepare','launch','work','collect']);p.add_argument('--source',type=Path)
    p.add_argument('--host');p.add_argument('--name');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.mode=='contract':contract(a.output)
    elif a.mode=='prepare':prepare(a.source)
    elif a.mode=='launch':launch(a.host)
    elif a.mode=='work':work(a.name)
    else:collect()
