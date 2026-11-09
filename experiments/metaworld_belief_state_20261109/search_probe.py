"""Frozen solver-budget test, selected before the2000-update result exists."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,now,environment
from experiments.metaworld_transition_only_20261109.campaign import command,HARD

PARENT=BASE/'runs/metaworld_belief_state_20261109'
ROOT=PARENT/'search-v1'
PARTICLES=64
ITERATIONS=4


def plan(world,hidden,image,goal,generator,warm):
    import torch
    from experiments.metaworld_endpoint_density_20261109.model import advance
    from experiments.metaworld_belief_state_20261109.evaluate import cost
    mean=image.new_zeros(12,4) if warm is None else warm.reshape(12,5,4).mean(1)
    std=torch.ones_like(mean);best=None;best_score=float('inf');first=None
    energy_seed=int(hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()[:15],16)
    for iteration in range(ITERATIONS):
        raw=torch.randn(PARTICLES,12,4,device=image.device,generator=generator)
        candidates=(mean[None]+std[None]*raw).clamp(-1,1);candidates[0]=mean
        if best is not None:candidates[1]=best
        if iteration==0:first=hashlib.sha256(candidates[:,:6].cpu().numpy().tobytes()).hexdigest()
        state=hidden.expand(PARTICLES,-1,-1,-1).clone()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for t in range(60):state=advance(world.parent,state,candidates[:,t//5])
        scores=cost(world,state,(image*2-1).expand(PARTICLES,-1,-1,-1),goal*2-1,energy_seed)
        values,indices=scores.topk(10,largest=False);elite=candidates[indices]
        if float(values[0])<best_score:best_score=float(values[0]);best=elite[0].clone()
        mean=elite.mean(0);std=elite.std(0,unbiased=False).clamp_min(1e-6)
    return best[:,None].expand(-1,5,-1).reshape(60,4),dict(planner_score=best_score,
        score_kind='latent_KL_plus_expected_finite_image_NELBO',first_bank_prefix_sha256=first,
        renderer_queries_per_candidate=10,latent_samples=2,actual_particles=PARTICLES,actual_iterations=ITERATIONS)


def evaluate(config,manifest,output,particles,iterations,budget):
    global PARTICLES,ITERATIONS
    PARTICLES=particles;ITERATIONS=iterations
    from experiments.metaworld_belief_state_20261109 import evaluate as original
    original.plan=__import__('torch').no_grad()(plan);old_save=original.save
    def metadata(path,value):
        if Path(path).name=='metrics.json':
            value.update(particles=PARTICLES,iterations=ITERATIONS,
                controller_override='frozen_CEM_population_iteration_budget',
                scope='Samefrozenjoint500world/goal/cases/H60/e15/10imagequeriespercandidate. '
                    'OnlyCEMparticles/iterationschange;17.578125xor32xtrajectoryforecastsversus64x4. '
                    'Candidatenoisebanksdifferentdimensions,notclaimedidenticalpathdraws. NoSGD/goalpath/actionlabels. NotSOTA/equalcompute.')
        old_save(path,value)
    original.save=metadata;original.run(config,500,manifest,output,budget)


def verify(d):
    assert d['passed'] and d['state_before']==d['state_after']
    for r in d['episodes']:
        assert sha(r['trace'])==r['trace_sha256'] and r['initial_rgb_replay']['passed']
        with np.load(r['trace']) as z:
            assert z['actions'].shape==(r['steps'],4) and bool(z['successes'].any())==r['success']
            assert np.isfinite(z['actions']).all() and np.abs(z['actions']).max()<=1 and r['steps']<=500


def prepare():
    p=read(PARENT/'plan.json');ROOT.mkdir(exist_ok=False);jobs=[j for j in p['jobs'] if j['mode']=='joint_anchor']
    cases=read(BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json')['rows'];evaluations=[]
    for particles,iterations in [(300,15),(1024,8)]:
        for j in jobs:
            assert read(Path(j['output'])/'AUDIT-0000500.json')['passed']
            for c in cases:
                manifest=ROOT/'manifests'/(c['task']+'.json')
                if not manifest.exists():save(manifest,dict(rows=[c]))
                name=f'{j["name"]}-p{particles}-i{iterations}-{c["task"]}'
                evaluations.append(dict(name=name,job=j['name'],config=j['config'],particles=particles,iterations=iterations,
                    manifest=str(manifest),output=str(ROOT/'evaluation'/name)))
    inherited=[e for e in p['evaluations'] if e['job'] in {j['name'] for j in jobs} and e['step']==500]
    assert len(inherited)==16 and len(evaluations)==32
    save(ROOT/'plan.json',dict(utc=now(),source=p['source'],source_hashes=p['source_hashes'],helper_sha256=sha(__file__),
        jobs=jobs,evaluations=evaluations,inherited=inherited,deadline=min(p['deadline'],time.time()+7200),
        design='Freezejoint500selectedbefore2000result. Same2parents/8cases:64x4inheritedvs300x15and1024x8. '
            'Onlysolverbudgetchanges;top10/H60/held5/e15/10imagequeriespercandidateunchanged. '
            'Twoactual20stepdefault64x4bitexactcontrols/successes/firstbankgatesbeforeformal.',
        model_updates=0,automatic_expansion=False,automatic_long_promotion=False))


def preflight(name,gpu):
    p=read(ROOT/'plan.json');j=next(x for x in p['jobs'] if x['name']==name);s=Path(p['source']);out=ROOT/'technical'/name
    command([PYTHON,str(Path(__file__).resolve()),'evaluate','--config',j['config'],'--manifest',ROOT/'manifests/push-v3.json',
        '--output',out,'--particles',64,'--iterations',4,'--budget',20],out.with_suffix('.log'),environment(s,gpu),s,p['deadline'])
    d=read(out/'metrics.json');verify(d);reference=read(PARENT/'evaluation'/f'{name}-s500-push-v3'/'metrics.json')
    a,b=d['episodes'][0],reference['episodes'][0]
    with np.load(a['trace']) as x,np.load(b['trace']) as y:
        assert np.array_equal(x['actions'],y['actions'][:20]) and np.array_equal(x['successes'],y['successes'][:20])
    assert a['planning'][0]['first_bank_prefix_sha256']==b['planning'][0]['first_bank_prefix_sha256']
    save(out/'PREFLIGHT.json',dict(passed=True,utc=now(),actual20_native_actions_success_firstbank_bitexact=True,model_updates=0))


def work(host,gpu):
    p=read(ROOT/'plan.json');s=Path(p['source'])
    assert sha(__file__)==p['helper_sha256'] and {str(f.relative_to(s)):sha(f) for f in s.rglob('*.py')}==p['source_hashes']
    assert all(read(ROOT/'technical'/j['name']/'PREFLIGHT.json')['passed'] for j in p['jobs'])
    for e in p['evaluations']:
        if (ROOT/'STOP').exists():return
        claim=ROOT/'claims'/(e['name']+'.json');claim.parent.mkdir(exist_ok=True)
        try:
            with claim.open('x') as f:json.dump(dict(host=host,gpu=gpu,pid=os.getpid(),utc=now()),f)
        except FileExistsError:continue
        try:
            command([PYTHON,str(Path(__file__).resolve()),'evaluate','--config',e['config'],'--manifest',e['manifest'],
                '--output',e['output'],'--particles',e['particles'],'--iterations',e['iterations']],Path(e['output']).with_suffix('.log'),environment(s,gpu),s,p['deadline'])
            verify(read(Path(e['output'])/'metrics.json'))
        except BaseException:
            save(ROOT/'errors'/(e['name']+'.json'),dict(utc=now(),traceback=traceback.format_exc()));raise


def launch(host):
    p=read(ROOT/'plan.json');s=Path(p['source']);workers=[]
    for gpu in range(8):
        with (ROOT/f'worker-{host}-{gpu}.log').open('x') as f:
            c=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'work','--host',host,'--gpu',str(gpu)],cwd=s,
                env=dict(environment(s,gpu),CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(gpu=gpu,pid=c.pid))
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
                cache[key]=d
        groups=__import__('collections').defaultdict(list)
        for d in cache.values():groups[f'{d["particles"]}x{d["iterations"]}'].extend(d['episodes'])
        groups={k:dict(episodes=len(v),hard_successes=sum(r['success'] for r in v if r['task'] in HARD),hard_episodes=sum(r['task'] in HARD for r in v),
            by_task={t:dict(successes=sum(r['success'] for r in v if r['task']==t),episodes=sum(r['task']==t for r in v)) for t in sorted({r['task'] for r in v})}) for k,v in groups.items()}
        errors=[str(x) for x in ROOT.glob('errors/*.json')]
        save(ROOT/'results/summary.json',dict(utc=now(),endpoints=len(cache),planned_endpoints=48,groups=groups,errors=errors,model_updates=0))
        if len(cache)==48:
            save(ROOT/'CLOSED.json',dict(passed=not errors,utc=now(),actual48trace_SHA_actions_success_checked=True,automatic_expansion=False));return
        time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','preflight','launch','work','evaluate','collect'])
    p.add_argument('--host');p.add_argument('--gpu',type=int);p.add_argument('--name')
    for key in ['config','manifest','output']:p.add_argument('--'+key,type=Path)
    p.add_argument('--particles',type=int,default=64);p.add_argument('--iterations',type=int,default=4);p.add_argument('--budget',type=int,default=500);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='preflight':preflight(a.name,a.gpu)
    elif a.mode=='launch':launch(a.host)
    elif a.mode=='work':work(a.host,a.gpu)
    elif a.mode=='evaluate':evaluate(a.config,a.manifest,a.output,a.particles,a.iterations,a.budget)
    else:collect()
