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

ORIGINAL=BASE/'runs/metaworld_belief_state_20261109'
ROOT=ORIGINAL/'expanded-joint-v1'


def verify(d):
    assert d['passed'] and d['state_before']==d['state_after']
    for r in d['episodes']:
        assert sha(r['trace'])==r['trace_sha256'] and r['initial_rgb_replay']['passed']
        with np.load(r['trace']) as z:
            assert z['actions'].shape==(r['steps'],4) and bool(z['successes'].any())==r['success']
            assert np.isfinite(z['actions']).all() and np.abs(z['actions']).max()<=1 and r['steps']<=500


def prepare():
    original=read(ORIGINAL/'plan.json');ROOT.mkdir(exist_ok=False)
    old=read(BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json')['rows']
    final=read(BASE/'runs/metaworld_transition_only_20261109/manifests/final.json')['rows']
    old_ids={(r['task'],r['seed'],r['sha256']) for r in old}
    extra=[r for r in final if (r['task'],r['seed'],r['sha256']) not in old_ids]
    assert len(old)==8 and len(final)==32 and len(extra)==24
    jobs=[j for j in original['jobs'] if j['mode']=='joint_anchor'];assert len(jobs)==2
    inherited=[];evaluations=[]
    for step in [2,500,2000]:
        for j in jobs:
            inherited.extend(e for e in original['evaluations'] if e['job']==j['name'] and e['step']==step)
            for case in extra:
                manifest=ROOT/'manifests'/f'{case["task"]}-{case["seed"]}.json'
                if not manifest.exists():save(manifest,dict(rows=[case]))
                name=f'{j["name"]}-s{step}-{case["task"]}-{case["seed"]}'
                evaluations.append(dict(name=name,job=j['name'],config=j['config'],step=step,
                    manifest=str(manifest),output=str(ROOT/'evaluation'/name)))
    assert len(inherited)==48 and len(evaluations)==144
    save(ROOT/'plan.json',dict(utc=now(),source=original['source'],source_hashes=original['source_hashes'],
        deadline=min(original['deadline'],time.time()+7200),jobs=jobs,evaluations=evaluations,inherited=inherited,
        inherited_episodes=48,new_episodes=144,combined_episodes=192,combined_physical_cases=32,
        original_trained_parents=2,model_updates=0,
        trigger='Joint500pilotPush+Drawer2/12hard,onlyseed2;NOTcross-parentrobust. Compare2/500/2000onSAME32physicalcases/bothparentsbeforeanylongpromotion.',
        scope='Frozenworld/source/controller/budget;all3checkpoints10rendererqueries/candidate. No originalcompletedepisodesrepeated. '
            '24newfrozenworkerscolocatedwith6fourGPUtrainerswhereuseful:24PHYSICALGPUs,not48. '
            'Previouslyinspecteddevelopmentalcases,notuntouchedconfirmation/publicSOTA;no SGDfromprotecteddata.',
        automatic_long_promotion=False,automatic_expansion=False))


def work(host,gpu):
    p=read(ROOT/'plan.json');s=Path(p['source'])
    assert {str(f.relative_to(s)):sha(f) for f in s.rglob('*.py')}==p['source_hashes']
    while time.time()<p['deadline']:
        if (ROOT/'STOP').exists():return
        selected=None
        for e in p['evaluations']:
            if not (Path(read(e['config'])['output'])/f'AUDIT-{e["step"]:07d}.json').exists():continue
            claim=ROOT/'claims'/(e['name']+'.json');claim.parent.mkdir(exist_ok=True)
            try:
                with claim.open('x') as f:__import__('json').dump(dict(host=host,gpu=gpu,pid=os.getpid(),utc=now()),f)
            except FileExistsError:continue
            selected=e;break
        if selected is None:
            if all((ROOT/'claims'/(e['name']+'.json')).exists() for e in p['evaluations']):return
            time.sleep(10);continue
        e=selected
        try:
            command([PYTHON,'-m','experiments.metaworld_belief_state_20261109.evaluate','--config',e['config'],'--step',e['step'],
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
    while time.time()<p['deadline']:
        for inherited,items in [(True,p['inherited']),(False,p['evaluations'])]:
            for e in items:
                key=('inherited-' if inherited else '')+e['name'];path=Path(e['output'])/'metrics.json'
                if key in cache or not path.exists():continue
                d=read(path);verify(d)
                expected={(c['task'],c['seed']):c['sha256'] for c in read(e['manifest'])['rows']}
                assert {(c['task'],c['seed']):c['case_sha256'] for c in d['episodes']}==expected
                cache[key]=dict(inherited=inherited,experiment=e,metrics=d,metadata_sha256=sha(path))
        groups=__import__('collections').defaultdict(list)
        for item in cache.values():groups[str(item['experiment']['step'])].extend(item['metrics']['episodes'])
        summary={}
        for mode,rows in groups.items():
            summary[mode]=dict(episodes=len(rows),by_task={t:dict(successes=sum(r['success'] for r in rows if r['task']==t),
                episodes=sum(r['task']==t for r in rows)) for t in sorted({r['task'] for r in rows})},
                hard_successes=sum(r['success'] for r in rows if r['task'] in HARD),hard_episodes=sum(r['task'] in HARD for r in rows))
        errors=[str(x) for x in ROOT.glob('errors/*.json')]
        n_inherited=sum(x['inherited'] for x in cache.values())
        save(ROOT/'results/summary.json',dict(utc=now(),combined_endpoints=len(cache),combined_endpoints_planned=192,
            inherited_episodes=n_inherited,inherited_episodes_planned=48,new_endpoints=len(cache)-n_inherited,new_endpoints_planned=144,
            groups=summary,errors=errors,model_updates=0,
            scope='2originalparents,32sharedphysicalcases,repeatedcheckpoints. Partialgroupsnotbalanced;developmentalexpansion,notpublicSOTA.'))
        if len(cache)==192:
            save(ROOT/'EPISODE_INDEX.json',{k:dict(experiment=x['experiment'],metadata_sha256=x['metadata_sha256'],inherited=x['inherited']) for k,x in cache.items()})
            save(ROOT/'CLOSED.json',dict(passed=not errors,utc=now(),model_updates=0,all192actualtraces_SHA_actions_success_checked=True,automatic_long_promotion=False));return
        time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch','work','collect']);p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='launch':launch(a.host)
    elif a.mode=='work':work(a.host,a.gpu)
    else:collect()
