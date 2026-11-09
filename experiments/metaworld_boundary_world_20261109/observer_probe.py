"""Frozen native-camera/filtering diagnosis; MORE RGB than sparse baseline.

Training histories contain eight native-step images. The existing controller
assimilates only every15 native steps. Compare dense assimilation and rebuilding
the exact eight-frame training context. All images are actually observed after
execution; no future GT, action targets, SGD or expert path is introduced.
"""
import argparse
import collections
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,tensor_sha,now,environment
from experiments.metaworld_transition_only_20261109.campaign import command,HARD
from experiments.metaworld_transition_only_20261109.evaluate import rgb_tensor
from experiments.metaworld_boundary_world_20261109.train import setup
from experiments.metaworld_boundary_world_20261109.evaluate import plan
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment

ROOT=BASE/'runs/metaworld_observer_probe_20261109'
PARENT=BASE/'runs/metaworld_boundary_world_20261109'


def prepare(source):
    ROOT.mkdir(exist_ok=False);cases=read(BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json')['rows'];jobs=[];inherited=[]
    for seed in [1,2]:
        cfg=PARENT/f'training/residual-seed{seed}/config.json'
        for case in cases:
            manifest=ROOT/'manifests'/(case['task']+'.json')
            if not manifest.exists():save(manifest,dict(rows=[case]))
            inherited.append(dict(name=f'sparse-seed{seed}-{case["task"]}',output=str(PARENT/f'evaluation/residual-seed{seed}-s2000-{case["task"]}')))
            for observer in ['dense','rebuild8']:
                name=f'{observer}-seed{seed}-{case["task"]}'
                jobs.append(dict(name=name,seed=seed,observer=observer,config=str(cfg),manifest=str(manifest),output=str(ROOT/'evaluation'/name)))
    save(ROOT/'plan.json',dict(utc=now(),source=str(source),source_hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},
        helper_sha256=sha(__file__),jobs=jobs,inherited=inherited,deadline=time.time()+7200,model_updates=0,
        inherited_episodes=16,new_episodes=32,combined_episodes=48,physical_cases=8,
        scope='FROZENownmeanworld14k. SameCEM64x4/H60/e15/held5/500anysuccess;denseandrebuild8captureRGBEVERYnative1vsbaselineevery15. '
        'MOREsensorinformationNOTsame-observation-budget/SOTA. BothnewmodesSAMEcameraobservations;onlystatefilterdiffers. '
        'No extraSGD/teacher/unknownfutureactions/goalpathtraining. ActualonlineRGBonly.',automatic_long_promotion=False))


@torch.no_grad()
def evaluate(config,manifest,output,observer,budget):
    cfg=read(config);device,world,_,_=setup(cfg);world.eval().requires_grad_(False);parent=world.parent
    weights=Path(cfg['output'])/'model-step0002000/weights.pt';assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256']
    world.load_state_dict(torch.load(weights,map_location=device,weights_only=True),strict=True);before=tensor_sha(world);output.mkdir(exist_ok=False);rows=[]
    for case in read(manifest)['rows']:
        assert sha(case['path'])==case['sha256'];cached=np.load(case['path']);env,state,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
        assert np.array_equal(state,cached['initial_state'])
        assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
        gate=rgb_replay_check(cached['initial'],pixels(env));assert gate['passed']
        image=rgb_tensor(cached['initial'],parent);goal=rgb_tensor(cached['goal'],parent);hidden=parent.observe(parent.initial(1),image)
        images=collections.deque([image],maxlen=8);history=collections.deque(maxlen=7);sensor_frames=1
        generator=torch.Generator(device='cuda').manual_seed(610000+case['seed']);actions=[];flags=[];planning=[];warm=None
        try:
            while len(actions)<budget:
                commands,detail=plan(world,hidden,image,goal,generator,warm);count=min(15,budget-len(actions))
                for a in commands[:count]:
                    _,_,term,trunc,info=env.step(a.cpu().numpy().astype(np.float32));actions.append(a.cpu().numpy().copy());flags.append(bool(info['success']))
                    hidden=world.advance(hidden,a[None],torch.zeros(1,device=device,dtype=torch.long))
                    if observer!='sparse':
                        actual=rgb_tensor(pixels(env),parent);sensor_frames+=1;images.append(actual);history.append(a.clone())
                        if observer=='dense':hidden=parent.observe(hidden,actual)
                    if flags[-1] or term or trunc:break
                if observer=='sparse':
                    image=rgb_tensor(pixels(env),parent);sensor_frames+=1;hidden=parent.observe(hidden,image)
                else:
                    image=images[-1]
                    if observer=='rebuild8':
                        n=len(images);frame_list=[images[0]]*(8-n)+list(images)
                        control_list=[torch.zeros(4,device=device)]*(7-len(history))+list(history)
                        batch=dict(history_pixels=torch.stack(frame_list,1).mul(255).round().byte(),
                            history_actions=torch.stack(control_list)[None],history_valid=torch.tensor([[False]*(8-n)+[True]*n],device=device))
                        hidden,reconstructed=world.prepare(batch,torch.zeros(1,device=device,dtype=torch.long));assert torch.equal(reconstructed,image)
                detail.update(native_steps=len(actions),sensor_frames=sensor_frames);planning.append(detail)
                if flags[-1] or term or trunc:break
                warm=torch.cat([commands[count:],commands[-1:].expand(count,-1)],0)
        finally:close_environment(env)
        trace=output/f'{case["task"]}-seed{case["seed"]}.npz';np.savez_compressed(trace,actions=np.asarray(actions,np.float32),successes=np.asarray(flags,bool))
        rows.append(dict(task=case['task'],seed=case['seed'],case_sha256=case['sha256'],success=bool(any(flags)),steps=len(actions),
            initial_rgb_replay=gate,trace=str(trace),trace_sha256=sha(trace),sensor_frames=sensor_frames,planning=planning))
    assert before==tensor_sha(world)
    save(output/'metrics.json',dict(passed=True,utc=now(),episodes=rows,state_before=before,state_after=before,weights_sha256=sha(weights),
        observer=observer,model_updates=0,goal_only_in_planner=True,executed_controls_INPUTS_ONLY=True,
        MORE_RGB_than_sparse_baseline=observer!='sparse',same_MPC_control_compute=True))


def preflight():
    p=read(ROOT/'plan.json');s=Path(p['source'])
    for seed in [1,2]:
        cfg=PARENT/f'training/residual-seed{seed}/config.json';out=ROOT/f'technical/seed{seed}'
        manifest=ROOT/'manifests/push-v3.json'
        command([PYTHON,str(Path(__file__).resolve()),'evaluate','--config',cfg,'--manifest',manifest,'--output',out,'--observer','sparse','--budget','20'],
            out.with_suffix('.log'),environment(s,seed-1),s,p['deadline'])
        d=read(out/'metrics.json');ref=read(PARENT/f'evaluation/residual-seed{seed}-s2000-push-v3/metrics.json')
        with np.load(d['episodes'][0]['trace']) as a,np.load(ref['episodes'][0]['trace']) as b:
            assert np.array_equal(a['actions'],b['actions'][:20]) and np.array_equal(a['successes'],b['successes'][:20])
        assert d['episodes'][0]['planning'][0]['first_bank_prefix_sha256']==ref['episodes'][0]['planning'][0]['first_bank_prefix_sha256']
    save(ROOT/'PREFLIGHT.json',dict(passed=True,utc=now(),two_actual20_native_sparse_controls_exact_to_original=True,model_updates=0))


def work(host,gpu):
    p=read(ROOT/'plan.json');s=Path(p['source']);assert sha(__file__)==p['helper_sha256']
    for j in p['jobs']:
        claim=ROOT/'claims'/(j['name']+'.json');claim.parent.mkdir(exist_ok=True)
        try:
            with claim.open('x') as f:__import__('json').dump(dict(host=host,gpu=gpu,pid=os.getpid(),utc=now()),f)
        except FileExistsError:continue
        try:
            command([PYTHON,str(Path(__file__).resolve()),'evaluate','--config',j['config'],'--manifest',j['manifest'],
                '--output',j['output'],'--observer',j['observer']],Path(j['output']).with_suffix('.log'),environment(s,gpu),s,p['deadline'])
        except BaseException:
            save(ROOT/'errors'/(j['name']+'.json'),dict(utc=now(),traceback=traceback.format_exc()));raise


def launch(host):
    p=read(ROOT/'plan.json');assert read(ROOT/'PREFLIGHT.json')['passed'];s=Path(p['source']);workers=[]
    for gpu in range(8):
        with (ROOT/f'worker-{host}-{gpu}.log').open('x') as f:
            c=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'work','--host',host,'--gpu',str(gpu)],cwd=s,
                env=environment(s,gpu)|{'CUDA_VISIBLE_DEVICES':''},stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(gpu=gpu,pid=c.pid))
    save(ROOT/f'launch-{host}.json',dict(host=host,utc=now(),workers=workers));print(host,workers,flush=True)


def collect():
    p=read(ROOT/'plan.json');cache={}
    while time.time()<p['deadline']:
        for inherited,jobs in [(True,p['inherited']),(False,p['jobs'])]:
            for j in jobs:
                path=Path(j['output'])/'metrics.json'
                if j['name'] in cache or not path.exists():continue
                d=read(path);assert d['passed'] and d['state_before']==d['state_after']
                for r in d['episodes']:
                    assert sha(r['trace'])==r['trace_sha256'] and r['initial_rgb_replay']['passed']
                    with np.load(r['trace']) as z:
                        assert z['actions'].shape==(r['steps'],4) and np.isfinite(z['actions']).all() and np.abs(z['actions']).max()<=1
                        assert bool(z['successes'].any())==r['success']
                cache[j['name']]=dict(observer='sparse' if inherited else j['observer'],metrics=d,inherited=inherited)
        groups=collections.defaultdict(list)
        for x in cache.values():groups[x['observer']].extend(x['metrics']['episodes'])
        summary={k:dict(episodes=len(v),hard_successes=sum(r['success'] for r in v if r['task'] in HARD),hard_episodes=sum(r['task'] in HARD for r in v),
            by_task={t:dict(successes=sum(r['success'] for r in v if r['task']==t),episodes=sum(r['task']==t for r in v)) for t in sorted({r['task'] for r in v})}) for k,v in groups.items()}
        errors=list(map(str,ROOT.glob('errors/*.json')));save(ROOT/'results/summary.json',dict(utc=now(),combined_episodes=len(cache),planned=48,groups=summary,errors=errors,model_updates=0))
        if len(cache)==48:
            save(ROOT/'CLOSED.json',dict(passed=not errors,utc=now(),all48_trace_SHA_action_success_verified=True,automatic_long_promotion=False));return
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','preflight','work','evaluate','launch','collect']);p.add_argument('--source',type=Path);p.add_argument('--host');p.add_argument('--gpu',type=int)
    for k in ['config','manifest','output']:p.add_argument('--'+k,type=Path)
    p.add_argument('--observer',choices=['sparse','dense','rebuild8']);p.add_argument('--budget',type=int,default=500);a=p.parse_args()
    if a.mode=='prepare':prepare(a.source)
    elif a.mode=='preflight':preflight()
    elif a.mode=='work':work(a.host,a.gpu)
    elif a.mode=='evaluate':evaluate(a.config,a.manifest,a.output,a.observer,a.budget)
    elif a.mode=='launch':launch(a.host)
    else:collect()
