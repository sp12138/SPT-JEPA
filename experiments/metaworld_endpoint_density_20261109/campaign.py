"""Six paired factual endpoint heads, finite training and96physical episodes."""
import argparse
import collections
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,now,environment
from experiments.metaworld_transition_only_20261109.campaign import command,HARD

ROOT=BASE/'runs/metaworld_endpoint_density_20261109'
OLD=BASE/'runs/metaworld_transition_only_20261109'


def prepare(source):
    ROOT.mkdir(exist_ok=False);jobs=[];evaluations=[];cases=read(OLD/'manifests/screen.json')['rows']
    for seed in [1,2]:
        parent_config=BASE/f'runs/metaworld_horizon_match_20261109/training/h60-seed{seed}/config.json'
        parent=read(parent_config);weights=Path(parent['output'])/'model-step0012000/weights.pt'
        assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256']
        assert read(Path(parent['output'])/'audit-0012000.json')['passed']
        origin=read(Path(parent['output'])/'ORIGIN.json')
        assert origin['ancestry'].startswith('Ownrandominit-factual-onlyH15inverseparent')
        assert sha(parent['parent'])==origin['parent_sha256']
        initial=read(Path(parent['parent']).parent/'initial.json')
        assert initial['all_parameters_random'] and not initial['external_weights_loaded'] and not initial['goal_path_supervised_ancestry']
        for mode in ['mean','uniform','highnoise']:
            index=len(jobs);host=['dev13','dev15','dev6'][index//2];gpu=[0,4][index%2]
            name=f'{mode}-seed{seed}';out=ROOT/'training'/name;out.mkdir(parents=True)
            cfg=dict(seed=seed,mode=mode,output=str(out),data=parent['data'],parent_config=str(parent_config),
                parent_weights=str(weights),parent_weights_sha256=sha(weights),parent_total_updates=12000,
                original_parent=str(parent['parent']),original_random_initial_evidence=str(Path(parent['parent']).parent/'initial.json'),
                scratch_own_ancestry=True,parent_frozen=True,head_maximum_updates=2000,
                supervision='RGB/knownexecutedforwardinputs only;no desiredgoals,expertpaths/policies,unknownnextactiontargets;optionalcurrentinputinversealreadyinownparent.')
            save(out/'config.json',cfg);jobs.append(dict(name=name,host=host,gpu=gpu,config=str(out/'config.json'),output=str(out),seed=seed,mode=mode))
    for case in cases:
        save(ROOT/'manifests'/(case['task']+'.json'),dict(rows=[case]))
    for step in [1000,2000]:
        for job in jobs:
            for case in cases:
                name=f'{job["name"]}-s{step}-{case["task"]}'
                evaluations.append(dict(name=name,job=job['name'],step=step,config=job['config'],
                    manifest=str(ROOT/'manifests'/(case['task']+'.json')),output=str(ROOT/'evaluation'/name)))
    save(ROOT/'plan.json',dict(utc=now(),source=str(source),deadline=time.time()+4*3600,jobs=jobs,evaluations=evaluations,
        source_hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},
        training_updates=6*2000,original_trained_parents=2,parent_updates_per_participant=12000,
        formal_episodes=96,technical_episodes=6,technical_native_steps=120,
        design='Same6randomheadinitials pairedwithin2seeds;mean(t=1only),uniformflow,highnoiseoddsx4flow. '
            'Identicalarchitecture/nativewindows/durations/noises/optimizer/dataorder/budget. ParentfactualH60inverse12000frozenandfullydisclosed. '
            'EndpointobservedfutureRGBpredictionfromcurrentRGBandhiddenadvancedwithknownexecutedcontrols;no desiredgoaltrainingoraction-policytargets. '
            'Inferencegoalonly:meanRGBdistance or64-stepreverseGaussianNELBOestimate,5rendererqueries/candidateallmodes. '
            'CEM64x4/H60/held5/execute15/native500anysuccess,8shareddevelopmentalcases,notpublicSOTA/untouchedconfirmation/equalpretrainingFLOPs.',
        trigger='Actualfrozenforecastvisualsshowmultiplearmghosts. LowerpixelMSEandbetterRGB-goaloptimizationdidnotyieldobjectsuccess. '
            'Testendpointsceneformationseparatelyfromconditionaldensity/highnoisetraining;no privileged-oracle or protected-evaluation data usedinSGD.',
        automatic_long_promotion=False,automatic_expansion=False))


def cpu_audit(job,step,plan):
    path=Path(job['output'])/f'AUDIT-{step:07d}.json'
    if path.exists():assert read(path)['passed'];return
    command([PYTHON,'-m','experiments.metaworld_endpoint_density_20261109.train','audit','--config',job['config'],'--step',step],
        Path(job['output'])/f'cpu-audit-{step}.log',dict(environment(Path(plan['source']),job['gpu']),CUDA_VISIBLE_DEVICES=''),
        plan['source'],plan['deadline'],cap=1800)
    assert read(path)['passed']


def global_gate():
    p=read(ROOT/'plan.json')
    if not all((Path(j['output'])/'PREFLIGHT.json').exists() for j in p['jobs']):return
    try:(ROOT/'GLOBAL_GATE_LOCK').mkdir()
    except FileExistsError:return
    try:
        hashes=collections.defaultdict(list);samples=collections.defaultdict(list);parents=collections.defaultdict(list);banks=[]
        for j in p['jobs']:
            out=Path(j['output']);assert read(out/'PREFLIGHT.json')['passed']
            origin=read(out/'ORIGIN.json');hashes[j['seed']].append(origin['head_initial_parameters_sha256'])
            parents[j['seed']].append(origin['parent_state_sha256']);samples[j['seed']].append((out/'sample-trace.jsonl').read_text())
            banks.append(read(out/'technical/metrics.json')['episodes'][0]['planning'][0]['first_bank_prefix_sha256'])
        assert all(len(set(x))==1 for x in hashes.values()) and all(len(set(x))==1 for x in samples.values())
        assert all(len(set(x))==1 for x in parents.values()) and len(set(banks))==1
        save(ROOT/'TRAINING_APPROVED.json',dict(passed=True,utc=now(),all6_actual_cpu_restore_and_20native_interface_pass=True,
            exact_paired_head_initial_parameters=True,exact_first2_data_durations_noise_uniforms=True,
            exact_frozen_parents_withinseed=True,common_initial_action_bank=True,head_supervision_sample_level=True))
    except BaseException:
        save(ROOT/'GLOBAL_GATE_ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise


def train_work(name):
    p=read(ROOT/'plan.json');j=next(j for j in p['jobs'] if j['name']==name);s=Path(p['source']);env=environment(s,j['gpu']);out=Path(j['output'])
    try:
        command([PYTHON,'-m','experiments.metaworld_endpoint_density_20261109.train','train','--config',j['config'],'--stop',2],
            out/'train-entry.log',env,s,p['deadline'])
        cpu_audit(j,2,p)
        command([PYTHON,'-m','experiments.metaworld_endpoint_density_20261109.evaluate','--config',j['config'],'--step',2,
            '--manifest',OLD/'manifests/technical.json','--output',out/'technical','--budget',20],out/'technical.log',env,s,p['deadline'])
        assert read(out/'technical/metrics.json')['passed']
        save(out/'PREFLIGHT.json',dict(passed=True,utc=now(),actual_new_head_updates=2,cpu_full_restore=True,physical_native_steps=20,parent_unchanged=True))
        while not (ROOT/'TRAINING_APPROVED.json').exists():
            assert time.time()<p['deadline'] and not (ROOT/'GLOBAL_GATE_ERROR.json').exists()
            assert not list(ROOT.glob('training/*/ERROR-*.json'))
            global_gate();time.sleep(2)
        previous=2
        for stop in [1000,2000]:
            command([PYTHON,'-m','experiments.metaworld_endpoint_density_20261109.train','train','--config',j['config'],'--stop',stop,
                '--resume',out/f'checkpoint-{previous:07d}.pt'],out/f'train-{stop}.log',env,s,p['deadline'])
            cpu_audit(j,stop,p);previous=stop
        save(out/'DONE.json',dict(passed=True,utc=now(),head_updates=2000,parent_updates=0))
    except BaseException:
        save(out/f'ERROR-worker-{os.getpid()}.json',dict(utc=now(),traceback=traceback.format_exc()));raise


def evaluation_work(host,gpu):
    p=read(ROOT/'plan.json');s=Path(p['source']);jobs={j['name']:j for j in p['jobs']}
    while time.time()<p['deadline']:
        if (ROOT/'STOP').exists():return
        selected=None
        for e in p['evaluations']:
            if not (Path(jobs[e['job']]['output'])/f'AUDIT-{e["step"]:07d}.json').exists():continue
            claim=ROOT/'evaluation-claims'/(e['name']+'.json');claim.parent.mkdir(exist_ok=True)
            try:
                with claim.open('x') as f:__import__('json').dump(dict(utc=now(),host=host,gpu=gpu,pid=os.getpid()),f)
            except FileExistsError:continue
            selected=e;break
        if selected is None:
            if all((Path(e['output'])/'metrics.json').exists() or (ROOT/'evaluation-errors'/(e['name']+'.json')).exists() for e in p['evaluations']):return
            time.sleep(10);continue
        e=selected
        try:
            command([PYTHON,'-m','experiments.metaworld_endpoint_density_20261109.evaluate','--config',e['config'],'--step',e['step'],
                '--manifest',e['manifest'],'--output',e['output']],Path(e['output']).with_suffix('.log'),environment(s,gpu),s,p['deadline'])
            assert read(Path(e['output'])/'metrics.json')['passed']
        except BaseException:
            save(ROOT/'evaluation-errors'/(e['name']+'.json'),dict(utc=now(),traceback=traceback.format_exc()));raise


def launch(host):
    p=read(ROOT/'plan.json');s=Path(p['source']);workers=[]
    for j in p['jobs']:
        if j['host']!=host:continue
        with (ROOT/(j['name']+'-worker.log')).open('x') as f:
            child=subprocess.Popen([PYTHON,'-m','experiments.metaworld_endpoint_density_20261109.campaign','train','--name',j['name']],cwd=s,
                env=dict(environment(s,j['gpu']),CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(kind='head_training',name=j['name'],gpu=j['gpu'],pid=child.pid))
    for gpu in [1,2,3,5,6,7]:
        with (ROOT/f'eval-worker-{host}-{gpu}.log').open('x') as f:
            child=subprocess.Popen([PYTHON,'-m','experiments.metaworld_endpoint_density_20261109.campaign','eval','--host',host,'--gpu',str(gpu)],cwd=s,
                env=dict(environment(s,gpu),CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(kind='frozen_eval',gpu=gpu,pid=child.pid))
    save(ROOT/f'launch-{host}.json',dict(utc=now(),host=host,workers=workers));print(host,workers,flush=True)


def collect():
    p=read(ROOT/'plan.json');cache={}
    while time.time()<p['deadline']:
        for e in p['evaluations']:
            path=Path(e['output'])/'metrics.json'
            if not path.exists() or e['name'] in cache:continue
            d=read(path);assert d['passed'] and d['parent_state_before']==d['parent_state_after'] and d['head_state_before']==d['head_state_after']
            for row in d['episodes']:
                assert sha(row['trace'])==row['trace_sha256'] and row['initial_rgb_replay']['passed']
                with np.load(row['trace']) as z:
                    assert z['actions'].shape==(row['steps'],4) and bool(z['successes'].any())==row['success']
                    assert np.isfinite(z['actions']).all() and np.abs(z['actions']).max()<=1 and row['steps']<=500
            cache[e['name']]=dict(experiment=e,metrics=d)
        groups=collections.defaultdict(list)
        for d in cache.values():groups[f'{d["metrics"]["mode"]}-step{d["metrics"]["step"]}'].extend(d['metrics']['episodes'])
        reports={k:dict(episodes=len(rows),by_task={task:dict(successes=sum(r['success'] for r in rows if r['task']==task),
            episodes=sum(r['task']==task for r in rows)) for task in sorted({r['task'] for r in rows})},
            hard_successes=sum(r['success'] for r in rows if r['task'] in HARD),hard_episodes=sum(r['task'] in HARD for r in rows)) for k,rows in groups.items()}
        errors=[str(x) for x in ROOT.glob('training/*/ERROR-*.json')]+[str(x) for x in ROOT.glob('evaluation-errors/*.json')]
        trained=sum((Path(j['output'])/'DONE.json').exists() for j in p['jobs'])
        save(ROOT/'results/summary.json',dict(utc=now(),trainings_complete=trained,trainings=6,evaluation_endpoints_complete=len(cache),
            evaluation_endpoints=96,formal_episodes=sum(len(x['metrics']['episodes']) for x in cache.values()),groups=reports,errors=errors,
            scope='Twooriginaltrainedparents,repeated8physicalcases;partialgroupsnotbalancedconfirmation. No highperformance/SOTAclaim.'))
        if trained==6 and len(cache)==96:
            save(ROOT/'CLOSED.json',dict(passed=not errors,utc=now(),automatic_long_promotion=False));return
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch','train','eval','collect'])
    p.add_argument('--source',type=Path);p.add_argument('--host');p.add_argument('--name');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.source)
    elif a.mode=='launch':launch(a.host)
    elif a.mode=='train':train_work(a.name)
    elif a.mode=='eval':evaluation_work(a.host,a.gpu)
    else:collect()
