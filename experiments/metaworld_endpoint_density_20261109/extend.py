"""More physical cases for the frozen2000 heads; inherit48 completed episodes."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,now,environment
from experiments.metaworld_transition_only_20261109.campaign import command,HARD

ORIGINAL=BASE/'runs/metaworld_endpoint_density_20261109'
ROOT=ORIGINAL/'expanded-v1'


def verify(d):
    assert d['passed'] and d['parent_state_before']==d['parent_state_after'] and d['head_state_before']==d['head_state_after']
    for r in d['episodes']:
        assert sha(r['trace'])==r['trace_sha256'] and r['initial_rgb_replay']['passed']
        with np.load(r['trace']) as z:
            assert z['actions'].shape==(r['steps'],4) and bool(z['successes'].any())==r['success']
            assert np.isfinite(z['actions']).all() and np.abs(z['actions']).max()<=1 and r['steps']<=500


def prepare():
    original=read(ORIGINAL/'plan.json');assert read(ORIGINAL/'CLOSED.json')['passed']
    ROOT.mkdir(exist_ok=False);old=read(BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json')['rows']
    final=read(BASE/'runs/metaworld_transition_only_20261109/manifests/final.json')['rows']
    old_ids={(r['task'],r['seed'],r['sha256']) for r in old}
    extra=[r for r in final if (r['task'],r['seed'],r['sha256']) not in old_ids]
    assert len(old)==8 and len(final)==32 and len(extra)==24
    inherited={};evaluations=[]
    for j in original['jobs']:
        inherited[j['name']]=[]
        for e in original['evaluations']:
            if e['job']!=j['name'] or e['step']!=2000:continue
            path=Path(e['output'])/'metrics.json';d=read(path);verify(d)
            assert d['step']==2000 and d['mode']==j['mode']
            expected={(c['task'],c['seed']):c['sha256'] for c in read(e['manifest'])['rows']}
            assert {(c['task'],c['seed']):c['case_sha256'] for c in d['episodes']}==expected
            inherited[j['name']].append(dict(path=str(path),sha256=sha(path)))
        assert len(inherited[j['name']])==8
        for case in extra:
            manifest=ROOT/'manifests'/f'{case["task"]}-{case["seed"]}.json'
            if not manifest.exists():save(manifest,dict(rows=[case]))
            name=f'{j["name"]}-{case["task"]}-{case["seed"]}'
            evaluations.append(dict(name=name,job=j['name'],config=j['config'],step=2000,
                manifest=str(manifest),output=str(ROOT/'evaluation'/name)))
    save(ROOT/'plan.json',dict(utc=now(),source=original['source'],source_hashes=original['source_hashes'],
        deadline=min(original['deadline'],time.time()+7200),jobs=original['jobs'],evaluations=evaluations,inherited=inherited,
        inherited_episodes=48,new_episodes=144,combined_episodes=192,combined_physical_cases=32,
        parent_trainings=2,head_models=6,model_updates=0,
        trigger='Uniform/highnoise2000bothDrawer6042successseed2(152/138steps);meanseed2fails500. Seed1doesnotsucceed,soNOTrobustcrossparentgain.',
        scope='ExpandcasesforALL3methods/bothparents,frozenweights/source/controller/budget. Reuse48existingactualepisodeswithoutrepeatingthem. '
            'Previouslyinspecteddevelopmentalcases,notuntouchedconfirmation/publicSOTA. No longtrainingfromsingleparentpositive.',
        automatic_long_promotion=False,automatic_expansion=False))


def work(host,gpu):
    p=read(ROOT/'plan.json');s=Path(p['source'])
    assert {str(f.relative_to(s)):sha(f) for f in s.rglob('*.py')}==p['source_hashes']
    while time.time()<p['deadline']:
        if (ROOT/'STOP').exists():return
        selected=None
        for e in p['evaluations']:
            claim=ROOT/'claims'/(e['name']+'.json');claim.parent.mkdir(exist_ok=True)
            try:
                with claim.open('x') as f:__import__('json').dump(dict(host=host,gpu=gpu,pid=os.getpid(),utc=now()),f)
            except FileExistsError:continue
            selected=e;break
        if selected is None:return
        e=selected
        try:
            command([PYTHON,'-m','experiments.metaworld_endpoint_density_20261109.evaluate','--config',e['config'],'--step',2000,
                '--manifest',e['manifest'],'--output',e['output']],Path(e['output']).with_suffix('.log'),environment(s,gpu),s,p['deadline'])
            result=read(Path(e['output'])/'metrics.json');verify(result)
            expected={(c['task'],c['seed']):c['sha256'] for c in read(e['manifest'])['rows']}
            assert {(c['task'],c['seed']):c['case_sha256'] for c in result['episodes']}==expected
        except BaseException:
            save(ROOT/'errors'/(e['name']+'.json'),dict(utc=now(),traceback=traceback.format_exc()));raise


def launch(host):
    p=read(ROOT/'plan.json');s=Path(p['source']);workers=[]
    for gpu in range(8):
        with (ROOT/f'worker-{host}-{gpu}.log').open('x') as f:
            child=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'work','--host',host,'--gpu',str(gpu)],cwd=s,
                env=dict(environment(s,gpu),CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(gpu=gpu,pid=child.pid))
    save(ROOT/f'launch-{host}.json',dict(host=host,utc=now(),workers=workers));print(host,workers,flush=True)


def collect():
    p=read(ROOT/'plan.json');cache={}
    for name,paths in p['inherited'].items():
        for item in paths:
            assert sha(item['path'])==item['sha256'];d=read(item['path']);verify(d)
            cache['inherited-'+Path(item['path']).parent.name]=dict(job=name,metrics=d)
    assert len(cache)==48
    while time.time()<p['deadline']:
        for e in p['evaluations']:
            path=Path(e['output'])/'metrics.json'
            if e['name'] not in cache and path.exists():
                d=read(path);verify(d);cache[e['name']]=dict(job=e['job'],metrics=d)
        groups=__import__('collections').defaultdict(list)
        for item in cache.values():groups[item['metrics']['mode']].extend(item['metrics']['episodes'])
        summary={}
        for mode,rows in groups.items():
            summary[mode]=dict(episodes=len(rows),by_task={t:dict(successes=sum(r['success'] for r in rows if r['task']==t),
                episodes=sum(r['task']==t for r in rows)) for t in sorted({r['task'] for r in rows})},
                hard_successes=sum(r['success'] for r in rows if r['task'] in HARD),hard_episodes=sum(r['task'] in HARD for r in rows))
        errors=[str(x) for x in ROOT.glob('errors/*.json')]
        save(ROOT/'results/summary.json',dict(utc=now(),combined_endpoints=len(cache),combined_endpoints_planned=192,
            inherited_episodes=48,new_endpoints=len(cache)-48,new_endpoints_planned=144,groups=summary,errors=errors,
            model_updates=0,scope='2originalparents,32sharedphysicalcases,repeatedmethods. Partialgroupsnotbalanced;developmentalexpansion,notpublicSOTA.'))
        if len(cache)==192:
            save(ROOT/'CLOSED.json',dict(passed=not errors,utc=now(),model_updates=0,automatic_long_promotion=False));return
        time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch','work','collect']);p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='launch':launch(a.host)
    elif a.mode=='work':work(a.host,a.gpu)
    else:collect()
