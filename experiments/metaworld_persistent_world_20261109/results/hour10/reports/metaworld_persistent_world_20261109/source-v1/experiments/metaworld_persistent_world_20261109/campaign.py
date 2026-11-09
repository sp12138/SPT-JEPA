"""Finite three-arm paired world study; alternate exclusive training/evaluation."""
import argparse
import collections
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,now,environment
from experiments.metaworld_transition_only_20261109.campaign import command,HARD

ROOT=BASE/'runs/metaworld_persistent_world_20261109'
OLD=BASE/'runs/metaworld_endpoint_density_20261109'
STAGES=[2,500,2000]
MODULE='experiments.metaworld_persistent_world_20261109'


def prepare(source):
    ROOT.mkdir(exist_ok=False);jobs=[];evaluations=[]
    cases=read(BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json')['rows']
    for host in ['dev13','dev15','dev6']:assert read(BASE/f'reports/metaworld_boundary_world_20261109/INTERVENTION-CACHE-{host}.json')['passed']
    assert read(BASE/'runs/metaworld_interventions_20261109/CLOSED.json')['passed']
    for seed in [1,2]:
        parent_cfg=BASE/f'runs/metaworld_boundary_world_20261109/training/residual-seed{seed}/config.json';parent=read(parent_cfg)
        weights=Path(parent['output'])/'model-step0002000/weights.pt'
        assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256'] and read(Path(parent['output'])/'AUDIT-0002000.json')['passed']
        for pool in ['old','interventions']:
            for mode in ['gru','residual','reversible']:
                index=len(jobs);host=['dev13','dev15','dev6'][index//4];gpus=[2*(index%4),2*(index%4)+1]
                name=f'{pool}-{mode}-seed{seed}';out=ROOT/'training'/name;out.mkdir(parents=True)
                directories=[parent['data']] if pool=='old' else ['/tmp/prescottsun-random-interventions-native128-20261109-v1']
                cfg=dict(seed=seed,mode=mode,pool=pool,output=str(out),datasets=directories,parent_config=str(parent_cfg),
                    parent_weights=str(weights),parent_sha256=sha(weights),parent_updates=12000,mean_world_updates=2000,
                    transition_reset_intentional=True,new_updates_maximum=2000,training_gpus=2,global_batch=64,
                    supervision='ForwardOBSERVEDRGBtargets,knownexecutedcontrolsINPUTS;NOgoal/path/policy/actiontargets. '
                    'OwnscratchCNN/observer/control/RGBreadout14k ancestry;all3transitionmodulesNEWfromrandom. '
                    'ResidualparallelvsreversiblesequentialuseEXACTsameparameterVALUES;GRUhasdifferentparametrization/capacity. '
                    'Onlylastpairguaranteesalgebraicstateinverse. Allsamplelevel/GN/FP32.',
                    data_budget='Old-onlyordinaryLeRobotcontrolvsEXTRAnewordinaryrandom-interventionlogs;newdataarmNOTequal-dataSOTA. '
                    '400branchtrainepisodesfrom50freshcasecontexts+400testfrom50othercontexts;not400independentcases. '
                    'Scriptedcollectorcontextatpresampledtime,OURrandomactionsafterbranch. NOcase/successfiltering.',
                    optimization='FreshAdamW1e-4/warm100/cos6000/clip1/global64/twoGPUs;fixed2000maximum.')
                save(out/'config.json',cfg);jobs.append(dict(name=name,host=host,gpus=gpus,config=str(out/'config.json'),output=str(out),seed=seed,mode=mode,pool=pool))
    for case in cases:save(ROOT/'manifests'/(case['task']+'.json'),dict(rows=[case]))
    for step in STAGES:
        for job in jobs:
            for case in cases:
                name=f'{job["name"]}-s{step}-{case["task"]}'
                evaluations.append(dict(name=name,job=job['name'],step=step,config=job['config'],manifest=str(ROOT/'manifests'/(case['task']+'.json')),output=str(ROOT/'evaluation'/name)))
    save(ROOT/'plan.json',dict(utc=now(),source=str(source),deadline=time.time()+4*3600,jobs=jobs,evaluations=evaluations,
        source_hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},stages=STAGES,
        design='3controlledlatenttransitionsx2datapoolsx2OWNparents=12twoGPUlearners/24physicalcards. '
        'GRUfromrandomvsparallelresidualvssequentialreversiblecoupling;lastpairSAMEparams/FLOPs/differentdependence. '
        'AlluseSAMEpretrainedOWNCNN/observer/control/readout,14kcumulativepretraining;transitionRESETforall3intentionally. '
        'AllfollowordinaryfutureRGBMSE,NOgoals/expertpaths/actiontargets. WholepipelineGN/samplelevel. '
        'Separateold-dataarmfromnewadditionalrandom-interventiondataarm;notfairpublicSOTA/equaldata/equalcapacityversusGRU.',
        trigger='Forecastworldslosecontrol/contextoverlongunobservedrollouts;testpersistence,notlargersearch. '
        'Sequentialadditivecouplinghasexactconditionalinverse/unitJacobiandeterminant(mathematicalexactarithmetic),notallnumericalsingularvalues1. '
        'Freshinterventionaudit100cases/86withobjectposeproxybranchstd>1mm,75>1cm;stateONLYdiagnostic,NEVERlearnerinput/filter. '
        'Reversiblephysicalvisiblemotionneednotbeinvertible;extraunobservedlatentdimensionscanstorelostinformation. Mustvalidateactualcontrol.',
        automatic_long_promotion=False,automatic_expansion=False))


def cpu_audit(job,step,p):
    command([PYTHON,'-m',MODULE+'.train','audit','--config',job['config'],'--step',step],
        Path(job['output'])/f'cpu-audit-{step}.log',dict(environment(Path(p['source']),job['gpus'][0]),CUDA_VISIBLE_DEVICES=''),
        p['source'],p['deadline'],cap=1800)
    assert read(Path(job['output'])/f'AUDIT-{step:07d}.json')['passed']


def gate():
    p=read(ROOT/'plan.json')
    if not all((Path(j['output'])/'PREFLIGHT.json').exists() for j in p['jobs']):return
    try:(ROOT/'GLOBAL_GATE_LOCK').mkdir()
    except FileExistsError:return
    try:
        backbones=collections.defaultdict(set);streams=collections.defaultdict(set);couplings=collections.defaultdict(set);banks=[]
        for j in p['jobs']:
            out=Path(j['output']);assert read(out/'PREFLIGHT.json')['passed'];origin=read(out/'ORIGIN.json')
            backbones[j['seed']].add(origin['initial_backbone_sha256'])
            streams[(j['seed'],j['pool'])].add((out/'sample-trace.jsonl').read_text())
            if j['mode']!='gru':couplings[j['seed']].add(origin['initial_parameters_sha256'])
            banks.append(read(out/'technical/metrics.json')['episodes'][0]['planning'][0]['first_bank_prefix_sha256'])
        assert all(len(v)==1 for v in [*backbones.values(),*streams.values(),*couplings.values()]) and len(set(banks))==1
        save(ROOT/'TRAINING_APPROVED.json',dict(passed=True,utc=now(),twelve_native2_CPUrestore_FP32backward_physical20_pass=True,
            actual_twoGPU_global64_updates_each=True,identical_pretrained_backbones_withinseed=True,
            residual_reversible_exact_initial_parameter_values=True,paired_data_and_randomterms_withinseed_and_pool=True,
            common_initial_control_bank=True,technical_native_steps=240,not_GRU_parameter_matched=True))
    except BaseException:
        save(ROOT/'GLOBAL_GATE_ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise


def wait(path,p):
    while not path.exists():
        assert time.time()<p['deadline'] and not (ROOT/'STOP').exists()
        assert not (ROOT/'GLOBAL_GATE_ERROR.json').exists() and not list(ROOT.glob('errors/*.json'))
        time.sleep(3)


def train_work(name):
    p=read(ROOT/'plan.json');j=next(j for j in p['jobs'] if j['name']==name);s=Path(p['source']);out=Path(j['output'])
    env=dict(environment(s,j['gpus'][0]),CUDA_VISIBLE_DEVICES=','.join(map(str,j['gpus'])))
    try:
        previous=None
        for stop in STAGES:
            if previous is not None:wait(ROOT/f'COLLECTED-{previous:07d}.json',p)
            args=[PYTHON,'-m',MODULE+'.train','train','--config',j['config'],'--stop',stop]
            if previous is not None:args+=['--resume',out/f'checkpoint-{previous:07d}.pt']
            command(args,out/f'train-{stop}.log',env,s,p['deadline']);cpu_audit(j,stop,p)
            if stop==2:
                command([PYTHON,'-m',MODULE+'.evaluate','--config',j['config'],'--step',2,
                    '--manifest',BASE/'runs/metaworld_transition_only_20261109/manifests/technical.json','--output',out/'technical','--budget',20],
                    out/'technical.log',environment(s,j['gpus'][0]),s,p['deadline'])
                assert read(out/'technical/metrics.json')['passed']
                save(out/'PREFLIGHT.json',dict(passed=True,utc=now(),actual_new_updates=2,training_gpus=2,technical_steps=20))
                while not (ROOT/'TRAINING_APPROVED.json').exists():
                    gate();assert time.time()<p['deadline'] and not list(ROOT.glob('errors/*.json'));time.sleep(3)
            save(out/f'READY-{stop:07d}.json',dict(passed=True,utc=now(),step=stop));previous=stop
        save(out/'DONE.json',dict(passed=True,utc=now(),new_world_updates=2000))
    except BaseException:
        save(ROOT/'errors'/(name+'.json'),dict(utc=now(),traceback=traceback.format_exc()));raise


def evaluate_work(host,gpu):
    p=read(ROOT/'plan.json');s=Path(p['source'])
    try:
        for step in STAGES:
            for j in p['jobs']:wait(Path(j['output'])/f'READY-{step:07d}.json',p)
            for e in p['evaluations']:
                if e['step']!=step:continue
                claim=ROOT/'claims'/(e['name']+'.json');claim.parent.mkdir(exist_ok=True)
                try:
                    with claim.open('x') as f:json.dump(dict(host=host,gpu=gpu,pid=os.getpid(),utc=now()),f)
                except FileExistsError:continue
                command([PYTHON,'-m',MODULE+'.evaluate','--config',e['config'],'--step',step,
                    '--manifest',e['manifest'],'--output',e['output']],Path(e['output']).with_suffix('.log'),environment(s,gpu),s,p['deadline'])
            wait(ROOT/f'COLLECTED-{step:07d}.json',p)
    except BaseException:
        save(ROOT/'errors'/f'eval-{host}-{gpu}.json',dict(utc=now(),traceback=traceback.format_exc()));raise


def launch(host):
    p=read(ROOT/'plan.json');s=Path(p['source']);workers=[]
    for j in p['jobs']:
        if j['host']!=host:continue
        with (ROOT/(j['name']+'-worker.log')).open('x') as f:
            child=subprocess.Popen([PYTHON,'-m',MODULE+'.campaign','train','--name',j['name']],cwd=s,
                env=dict(environment(s,j['gpus'][0]),CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(kind='training',name=j['name'],gpus=j['gpus'],pid=child.pid))
    for gpu in range(8):
        with (ROOT/f'eval-{host}-{gpu}.log').open('x') as f:
            child=subprocess.Popen([PYTHON,'-m',MODULE+'.campaign','eval','--host',host,'--gpu',str(gpu)],cwd=s,
                env=dict(environment(s,gpu),CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(kind='frozen_eval_waits_exclusive_phase',gpu=gpu,pid=child.pid))
    save(ROOT/f'launch-{host}.json',dict(utc=now(),host=host,workers=workers));print(host,workers,flush=True)


def collect():
    p=read(ROOT/'plan.json');cache={}
    while time.time()<p['deadline']:
        for e in p['evaluations']:
            path=Path(e['output'])/'metrics.json'
            if e['name'] in cache or not path.exists():continue
            d=read(path);assert d['passed'] and d['state_before']==d['state_after']
            expected={(r['task'],r['seed']):r['sha256'] for r in read(e['manifest'])['rows']}
            assert {(r['task'],r['seed']):r['case_sha256'] for r in d['episodes']}==expected
            for row in d['episodes']:
                assert sha(row['trace'])==row['trace_sha256'] and row['initial_rgb_replay']['passed']
                with np.load(row['trace']) as z:
                    assert z['actions'].shape==(row['steps'],4) and bool(z['successes'].any())==row['success']
                    assert np.isfinite(z['actions']).all() and np.abs(z['actions']).max()<=1 and row['steps']<=500
            cache[e['name']]=dict(experiment=e,metrics=d)
        groups=collections.defaultdict(list)
        for d in cache.values():groups[f'{read(d["experiment"]["config"])["pool"]}-{d["metrics"]["mode"]}-step{d["metrics"]["step"]}'].extend(d['metrics']['episodes'])
        reports={k:dict(episodes=len(v),hard_successes=sum(r['success'] for r in v if r['task'] in HARD),hard_episodes=sum(r['task'] in HARD for r in v),
            by_task={t:dict(successes=sum(r['success'] for r in v if r['task']==t),episodes=sum(r['task']==t for r in v)) for t in sorted({r['task'] for r in v})}) for k,v in groups.items()}
        errors=[str(x) for x in ROOT.glob('errors/*.json')]
        save(ROOT/'results/summary.json',dict(utc=now(),endpoints=len(cache),planned_endpoints=288,groups=reports,errors=errors,
            trained=sum((Path(j['output'])/'DONE.json').exists() for j in p['jobs']),physical_cases=8,original_trained_parents=2))
        for step in STAGES:
            if sum(d['experiment']['step']==step for d in cache.values())==96 and not (ROOT/f'COLLECTED-{step:07d}.json').exists():
                assert not errors;save(ROOT/f'COLLECTED-{step:07d}.json',dict(passed=True,utc=now(),episodes=96,actual_trace_SHA_and_actions_and_successes_verified=True))
        if len(cache)==288:
            save(ROOT/'CLOSED.json',dict(passed=not errors,utc=now(),automatic_long_promotion=False));return
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch','train','eval','collect'])
    p.add_argument('--source',type=Path);p.add_argument('--host');p.add_argument('--name');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.source)
    elif a.mode=='launch':launch(a.host)
    elif a.mode=='train':train_work(a.name)
    elif a.mode=='eval':evaluate_work(a.host,a.gpu)
    else:collect()
