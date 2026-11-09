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

ROOT=BASE/'runs/metaworld_contact_geometry_20261109'
OLD=BASE/'runs/metaworld_endpoint_density_20261109'
STAGES=[2,500,2000]
MODULE='experiments.metaworld_contact_geometry_20261109'


def prepare(source):
    ROOT.mkdir(exist_ok=False);jobs=[];evaluations=[]
    cases=read(BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json')['rows']
    for seed in [1,2]:
        head_cfg=OLD/f'training/highnoise-seed{seed}/config.json';head=read(head_cfg)
        head_weights=Path(head['output'])/'head-step0002000/weights.pt'
        assert sha(head_weights)==read(head_weights.parent/'SEALED.json')['weights_sha256']
        assert read(Path(head['output'])/'AUDIT-0002000.json')['passed']
        origin=read(Path(head['output'])/'ORIGIN.json');assert origin['parent_goal_path_or_policy_ancestry'] is False
        initial=read(head['original_random_initial_evidence'])
        assert initial['all_parameters_random'] and not initial['external_weights_loaded'] and not initial['goal_path_supervised_ancestry']
        for mode in ['dense','sparse','transport']:
            index=len(jobs);host=['dev13','dev15','dev6'][index//2];gpus=list(range(0,4)) if index%2==0 else list(range(4,8))
            name=f'{mode}-seed{seed}';out=ROOT/'training'/name;out.mkdir(parents=True)
            cfg=dict(seed=seed,mode=mode,output=str(out),data=head['data'],head_config=str(head_cfg),
                head_weights=str(head_weights),head_sha256=sha(head_weights),parent_updates=12000,head_updates=2000,
                new_updates_maximum=2000,training_gpus=4,global_batch=64,
                supervision='FactualrecordedRGBandknownexecutedforwardcontrolsONLY. No desiredgoal/path/policy/actiontargets. '
                    'OwnparentinversepretrainingreconstructsSAMEcurrentinputa0only;inversefrozeninnewphase.',
                optimization='IntentionalfreshAdamW1e-4/warm100/cos6000/globalclip1,allarmsidentical. '
                    'Four-GPUDataParallel/global64;GNsamplelevel. Parent12000+head2000pretrainingdisclosed.')
            save(out/'config.json',cfg);jobs.append(dict(name=name,host=host,gpus=gpus,config=str(out/'config.json'),output=str(out),seed=seed,mode=mode))
    for case in cases:save(ROOT/'manifests'/(case['task']+'.json'),dict(rows=[case]))
    for step in STAGES:
        for job in jobs:
            for case in cases:
                name=f'{job["name"]}-s{step}-{case["task"]}'
                evaluations.append(dict(name=name,job=job['name'],step=step,config=job['config'],
                    manifest=str(ROOT/'manifests'/(case['task']+'.json')),output=str(ROOT/'evaluation'/name)))
    save(ROOT/'plan.json',dict(utc=now(),source=str(source),deadline=time.time()+4*3600,jobs=jobs,evaluations=evaluations,
        source_hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},stages=STAGES,
        design='Threeobservationalcodecsx2OWNfactualparents:denseAE,sparsekeypointAE,featuretransport. SameinitialCNN/keypoint/decoder/dynamicsparameterswithinseed;head2000weightsloadedbutUNUSED/frozen. '
            'SameinitialparameterVALUES/data/duration/randomterms withinseed. AdditionaluniformdrawchangesaugmentationRNGstreamrelativepreviousNLLcontrol;NOTexactpairedaugmentationsversusNLL. Sameglobal64/fourGPUs/budget. '
            'ActualfutureRGBreconstruction+futureconfigurationunit-featureprediction,allcoefficients1. GroupNorm/samplelevel. K16/sigmaonefeaturecellglobalarchitecturechoices,nokeypoint/state/actionlabels. '
            'ObservedactualendpointRGBonlytrainingtarget;desiredgoalONLYplanning. No posterior-future-image training branch. '
            'PlanningCEM64x4/H60/held5/execute15/native500,1configurationreadout/candidate,NORGBgeneratorqueries. '
            'Stages2/500/2000each48episodes,8developmentalphysicalcases/repeated2parents. NotpublicSOTA/equalbackboneFLOPs.',
        trigger='FrozenendpointheadsuseactionsandimproveRGByetweakmanipulation;generated-observationcompositionH120failsallReach/Wall. '
            'Protectedcontactaudit:evenACTUALterminalRGBdistancepreferfailurewhile3/64pathsfinishsuccessfully. Testconfigurationgeometryratherthanpixelarea. TransporterprinciplefromKulkarni2019,NOTnovelty/officialreplicationclaim. Nevertraindiagnosticbanks.',
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
        initials=collections.defaultdict(list);streams=collections.defaultdict(list);banks=[]
        for j in p['jobs']:
            out=Path(j['output']);assert read(out/'PREFLIGHT.json')['passed']
            initials[j['seed']].append(read(out/'ORIGIN.json')['initial_parameters_sha256'])
            streams[j['seed']].append((out/'sample-trace.jsonl').read_text())
            banks.append(read(out/'technical/metrics.json')['episodes'][0]['planning'][0]['first_bank_prefix_sha256'])
        assert all(len(set(x))==1 for x in initials.values()) and all(len(set(x))==1 for x in streams.values())
        assert len(set(banks))==1
        save(ROOT/'TRAINING_APPROVED.json',dict(passed=True,utc=now(),six_full_cpu_restores_and_positive_FP32_backward=True,
            actual4GPU_two_updates_each=True,technical_native_steps=120,exact_initial_parameter_values=True,
            exact_first2_data_and_allrandomterms=True,common_initial_control_bank=True,sample_level=True))
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
                save(out/'PREFLIGHT.json',dict(passed=True,utc=now(),actual_new_updates=2,training_gpus=4,technical_steps=20))
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
        for d in cache.values():groups[f'{d["metrics"]["mode"]}-step{d["metrics"]["step"]}'].extend(d['metrics']['episodes'])
        reports={k:dict(episodes=len(v),hard_successes=sum(r['success'] for r in v if r['task'] in HARD),hard_episodes=sum(r['task'] in HARD for r in v),
            by_task={t:dict(successes=sum(r['success'] for r in v if r['task']==t),episodes=sum(r['task']==t for r in v)) for t in sorted({r['task'] for r in v})}) for k,v in groups.items()}
        errors=[str(x) for x in ROOT.glob('errors/*.json')]
        save(ROOT/'results/summary.json',dict(utc=now(),endpoints=len(cache),planned_endpoints=144,groups=reports,errors=errors,
            trained=sum((Path(j['output'])/'DONE.json').exists() for j in p['jobs']),physical_cases=8,original_trained_parents=2))
        for step in STAGES:
            if sum(d['experiment']['step']==step for d in cache.values())==48 and not (ROOT/f'COLLECTED-{step:07d}.json').exists():
                assert not errors;save(ROOT/f'COLLECTED-{step:07d}.json',dict(passed=True,utc=now(),episodes=48,actual_trace_SHA_and_actions_and_successes_verified=True))
        if len(cache)==144:
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
