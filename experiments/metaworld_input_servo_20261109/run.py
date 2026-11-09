"""Frozen SAME-input inverse servo; goal enters only at evaluation."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,tensor_sha,now,environment
from experiments.metaworld_transition_only_20261109.campaign import command,HARD
from experiments.metaworld_horizon_match_20261109.model import build
from experiments.metaworld_transition_only_20261109.evaluate import rgb_tensor
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment

ROOT=BASE/'runs/metaworld_input_servo_20261109'


def prepare(source):
    ROOT.mkdir(exist_ok=False);cases=read(BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json')['rows'];jobs=[]
    for seed in [1,2]:
        cfg=BASE/f'runs/metaworld_horizon_match_20261109/training/h60-seed{seed}/config.json'
        weights=Path(read(cfg)['output'])/'model-step0012000/weights.pt'
        assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256']
        for cadence in [1,5,15]:
            for case in cases:
                manifest=ROOT/'manifests'/(case['task']+'.json')
                if not manifest.exists():save(manifest,dict(rows=[case]))
                name=f'seed{seed}-execute{cadence}-{case["task"]}'
                jobs.append(dict(name=name,seed=seed,cadence=cadence,config=str(cfg),manifest=str(manifest),output=str(ROOT/'evaluation'/name)))
    save(ROOT/'plan.json',dict(utc=now(),source=str(source),source_hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},
        helper_sha256=sha(__file__),jobs=jobs,deadline=time.time()+7200,model_updates=0,formal_episodes=48,physical_cases=8,
        setting='OwnscratchH60factual12000parents/currentnativea0inverse. NOa1/unknownoptimal-actionlabels,goal/pathpolicytraining. '
            'Goalonlyinference. OriginalinverseheadNOTupdatedafterownfactualparent. RGB-only/FP32. '
            'Cadence15matchespreviousobservationinterval;cadence1/5haveMOREfeedback,not same-evaluationbudget/SOTAcomparisons.',
        automatic_long_promotion=False,automatic_expansion=False))


@torch.no_grad()
def evaluate(config,manifest,output,cadence):
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    cfg=read(config);model=build(cfg).cuda().eval().requires_grad_(False)
    weights=Path(cfg['output'])/'model-step0012000/weights.pt';assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256']
    model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True),strict=True);before=tensor_sha(model)
    output.mkdir(exist_ok=False);rows=[]
    for case in read(manifest)['rows']:
        assert sha(case['path'])==case['sha256'];cached=np.load(case['path']);env,state,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
        assert np.array_equal(state,cached['initial_state'])
        assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
        gate=rgb_replay_check(cached['initial'],pixels(env));assert gate['passed']
        image=rgb_tensor(cached['initial'],model);goal=rgb_tensor(cached['goal'],model);gf=model.vision(goal)
        actions=[];flags=[];norms=[]
        try:
            while len(actions)<500:
                f=model.vision(image);raw=model.inverse(torch.cat([f,gf,gf-f],1)).float()[0]
                assert torch.isfinite(raw).all();a=raw.clamp(-1,1).cpu().numpy();norms.append(float(raw.norm()))
                for _ in range(min(cadence,500-len(actions))):
                    _,_,term,trunc,info=env.step(a);actions.append(a.copy());flags.append(bool(info['success']))
                    if flags[-1] or term or trunc:break
                if flags[-1] or term or trunc:break
                image=rgb_tensor(pixels(env),model)
        finally:close_environment(env)
        trace=output/f'{case["task"]}-seed{case["seed"]}.npz'
        np.savez_compressed(trace,actions=np.asarray(actions,np.float32),successes=np.asarray(flags,bool))
        rows.append(dict(task=case['task'],seed=case['seed'],case_sha256=case['sha256'],success=bool(any(flags)),steps=len(actions),
            initial_rgb_replay=gate,trace=str(trace),trace_sha256=sha(trace),raw_inverse_norms=norms))
    assert before==tensor_sha(model)
    save(output/'metrics.json',dict(passed=True,utc=now(),episodes=rows,state_before=before,state_after=before,
        weights_sha256=sha(weights),cadence=cadence,training_parent_seed=cfg['seed'],budget_native=500,
        target_action_role='SAMEcurrentforwardinputnativea0ONLY,nevera1/unknownoptimal.',goal_only_in_planner=True,
        expert_paths_or_optimal_action_labels=False,model_updates=0))


def work(host,gpu):
    p=read(ROOT/'plan.json');s=Path(p['source']);assert sha(__file__)==p['helper_sha256']
    for j in p['jobs']:
        claim=ROOT/'claims'/(j['name']+'.json');claim.parent.mkdir(exist_ok=True)
        try:
            with claim.open('x') as f:__import__('json').dump(dict(host=host,gpu=gpu,pid=os.getpid(),utc=now()),f)
        except FileExistsError:continue
        try:
            command([PYTHON,str(Path(__file__).resolve()),'evaluate','--config',j['config'],'--manifest',j['manifest'],
                '--output',j['output'],'--cadence',j['cadence']],Path(j['output']).with_suffix('.log'),environment(s,gpu),s,p['deadline'])
        except BaseException:
            save(ROOT/'errors'/(j['name']+'.json'),dict(utc=now(),traceback=traceback.format_exc()));raise


def launch(host):
    p=read(ROOT/'plan.json');s=Path(p['source']);workers=[]
    for gpu in range(8):
        with (ROOT/f'worker-{host}-{gpu}.log').open('x') as f:
            c=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'work','--host',host,'--gpu',str(gpu)],cwd=s,
                env=environment(s,gpu),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(gpu=gpu,pid=c.pid))
    save(ROOT/f'launch-{host}.json',dict(host=host,utc=now(),workers=workers));print(host,workers,flush=True)


def collect():
    p=read(ROOT/'plan.json');cache={}
    while time.time()<p['deadline']:
        for j in p['jobs']:
            path=Path(j['output'])/'metrics.json'
            if j['name'] in cache or not path.exists():continue
            d=read(path);assert d['passed'] and d['state_before']==d['state_after']
            for r in d['episodes']:
                assert sha(r['trace'])==r['trace_sha256'] and r['initial_rgb_replay']['passed']
                with np.load(r['trace']) as z:
                    assert z['actions'].shape==(r['steps'],4) and bool(z['successes'].any())==r['success']
                    assert np.isfinite(z['actions']).all() and np.abs(z['actions']).max()<=1
            cache[j['name']]=d
        groups=__import__('collections').defaultdict(list)
        for d in cache.values():groups[str(d['cadence'])].extend(d['episodes'])
        groups={k:dict(episodes=len(v),hard_successes=sum(r['success'] for r in v if r['task'] in HARD),hard_episodes=sum(r['task'] in HARD for r in v),
            by_task={t:dict(successes=sum(r['success'] for r in v if r['task']==t),episodes=sum(r['task']==t for r in v)) for t in sorted({r['task'] for r in v})}) for k,v in groups.items()}
        errors=[str(p) for p in ROOT.glob('errors/*.json')]
        save(ROOT/'results/summary.json',dict(utc=now(),endpoints=len(cache),planned_endpoints=48,groups=groups,errors=errors,model_updates=0))
        if len(cache)==48:
            save(ROOT/'CLOSED.json',dict(passed=not errors,utc=now(),actual48trace_SHA_and_actions_success_checked=True,automatic_long_promotion=False));return
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch','work','evaluate','collect']);p.add_argument('--source',type=Path);p.add_argument('--host');p.add_argument('--gpu',type=int)
    for key in ['config','manifest','output']:p.add_argument('--'+key,type=Path)
    p.add_argument('--cadence',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.source)
    elif a.mode=='launch':launch(a.host)
    elif a.mode=='work':work(a.host,a.gpu)
    elif a.mode=='evaluate':evaluate(a.config,a.manifest,a.output,a.cadence)
    else:collect()
