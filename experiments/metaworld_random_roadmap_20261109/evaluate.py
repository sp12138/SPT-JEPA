"""Frozen random-state roadmap control; goals enter only graph inference.

Neither graph search nor this actor reads provenance, trajectory order, task
labels, expert paths, or logged future commands. Actions are freshly generated
by our SAME-current-input inverse; all sensor frames follow actual execution.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import time
import traceback
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,tensor_sha,now,environment
from experiments.metaworld_transition_only_20261109.campaign import command,HARD
from experiments.metaworld_transition_only_20261109.evaluate import rgb_tensor
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment

OPS=runpy.run_path(str(BASE/'reports/metaworld_random_roadmap_20261109/edges-v1.py'))
ROOT=OPS['ROOT'];EVAL=ROOT/'evaluation-v1';K=OPS['K']


def prepare():
    assert read(ROOT/'GRAPH_READY.json')['passed'];EVAL.mkdir(exist_ok=False);cases=read(BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json')['rows'];jobs=[];inherited=[]
    for seed in [1,2]:
        for case in cases:
            manifest=EVAL/'manifests'/(case['task']+'.json')
            if not manifest.exists():save(manifest,dict(rows=[case]))
            inherited.append(dict(name=f'direct-seed{seed}-{case["task"]}',output=str(BASE/f'runs/metaworld_input_servo_20261109/evaluation/seed{seed}-execute1-{case["task"]}')))
            for mode in ['appearance','world']:
                name=f'{mode}-seed{seed}-{case["task"]}'
                jobs.append(dict(name=name,seed=seed,mode=mode,manifest=str(manifest),output=str(EVAL/'episodes'/name)))
    p=read(ROOT/'plan.json');save(EVAL/'plan.json',dict(utc=now(),source=p['source'],source_hashes=p['source_hashes'],helper_sha256=sha(__file__),
        jobs=jobs,inherited=inherited,deadline=time.time()+7200,inherited_episodes=16,new_episodes=32,combined_episodes=48,physical_cases=8,
        models='FROZENownIDM12k+meanworld14k+diagcov2k;noSGD/action/goal/path/policytargetlearning.',
        scope='EXTRArandom-TRAIN-state-memory/10,800nodes/pastonly. NOfuture-command/expertpathretrieval. '
        '32root-neighborsteeringcandidatesx5native/historicalgoalvalueDijkstra,execute1/nativecamera/500anysuccess. '
        'DirectIDMcontrolsame12kweights/nativecamera16inherited. Notmatchedoriginaldata/plannerinformation/publicSOTA/equalFLOPs.',automatic_long_promotion=False))


def graph(seed,mode):
    p=read(ROOT/'GRAPH_READY.json');directory=ROOT/'graph'/f'seed{seed}';neighbors=np.load(directory/'neighbors.npy');cost=np.load(directory/(mode+'.npy'))
    assert sha(directory/'neighbors.npy')==p['files'][str(seed)]['neighbors.npy'] and sha(directory/(mode+'.npy'))==p['files'][str(seed)][mode+'.npy']
    n=p['nodes'];assert neighbors.shape==cost.shape==(n,K) and np.isfinite(cost).all() and cost.min()>0
    assert neighbors.min()>=0 and neighbors.max()<n
    return neighbors,cost


@torch.no_grad()
def goal_values(world,inverse,bank,goal,goal_feature,mode,neighbors,cost):
    n=len(bank['image']);terminal=OPS['neighbors'](bank,goal).flatten()
    world_cost,appearance_cost,_=OPS['steer'](world,inverse,bank['hidden'][terminal],bank['image'][terminal],bank['feature'][terminal],goal.expand(K,-1,-1,-1),goal_feature.expand(K,-1,-1,-1))
    edge=(world_cost if mode=='world' else appearance_cost).cpu().numpy()
    rows=np.concatenate([np.repeat(np.arange(n),K),terminal.cpu().numpy()]);columns=np.concatenate([neighbors.flatten(),np.full(K,n)])
    values=np.concatenate([cost.flatten(),edge]).astype(np.float64)
    matrix=csr_matrix((values,(rows,columns)),shape=(n+1,n+1));distance=dijkstra(matrix.T.tocsr(),directed=True,indices=n)
    return distance[:n],dict(goal_attachment_nodes=terminal.cpu().tolist(),goal_attachment_costs=edge.tolist(),
        reachable_bank_nodes=int(np.isfinite(distance[:n]).sum()),bank_nodes=n)


def observed_state(world,inverse,images,history):
    n=len(images);frames=[images[0]]*(8-n)+list(images);controls=[torch.zeros(4,device='cuda')]*(7-len(history))+list(history)
    batch=dict(history_pixels=torch.stack(frames,1).mul(255).round().byte(),history_actions=torch.stack(controls)[None],
        history_valid=torch.tensor([[False]*(8-n)+[True]*n],device='cuda'))
    hidden,current=world.parent.prepare(batch,torch.zeros(1,device='cuda',dtype=torch.long));feature=inverse.vision(current)
    assert torch.equal(current,images[-1]);return hidden,current,feature


@torch.no_grad()
def action(world,inverse,bank,hidden,current,feature,goal,goal_feature,value,mode):
    nearest=OPS['neighbors'](bank,current).flatten()
    w,a,commands=OPS['steer'](world,inverse,hidden.expand(K,-1,-1,-1).clone(),current.expand(K,-1,-1,-1),
        feature.expand(K,-1,-1,-1),bank['image'][nearest],bank['feature'][nearest])
    local=w if mode=='world' else a;future=torch.as_tensor(value[nearest.cpu().numpy()],device='cuda',dtype=local.dtype);total=local+future
    best=int(total.argmin());score=float(total[best]);chosen=int(nearest[best]);command=commands[best]
    goal_distance=float((current-goal).square().sum());radius=float((current-bank['image'][nearest[-1:]]).square().sum())
    direct_allowed=goal_distance<=radius;fallback=not np.isfinite(score)
    # Insert the goal by the SAME local-neighborhood rule; avoid a long, invalid
    # direct shortcut through contact barriers. Fall back only when no route exists.
    if direct_allowed or fallback:
        dw,da,dc=OPS['steer'](world,inverse,hidden,current,feature,goal,goal_feature);direct=float((dw if mode=='world' else da)[0])
        if fallback or direct<score:command=dc[0];score=direct;chosen=-1
    assert torch.isfinite(command).all() and command.abs().max()<=1
    return command,dict(selected_node=chosen,root_total_cost=score,direct_goal_local=direct_allowed,
        no_finite_roadmap_route_fallback=fallback,current_goal_RGB_MSE=goal_distance/current[0].numel())


@torch.no_grad()
def evaluate(seed,mode,manifest,output,budget):
    world,inverse,ancestry=OPS['NODE']['load_models'](seed);before=(tensor_sha(world),tensor_sha(inverse));bank=None if mode=='direct' else OPS['load_bank'](seed)
    neighbors,cost=(None,None) if mode=='direct' else graph(seed,mode);output.mkdir(parents=True,exist_ok=False);rows=[]
    for case in read(manifest)['rows']:
        assert sha(case['path'])==case['sha256'];cached=np.load(case['path']);env,state,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
        assert np.array_equal(state,cached['initial_state'])
        assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
        gate=rgb_replay_check(cached['initial'],pixels(env));assert gate['passed']
        image=rgb_tensor(cached['initial'],inverse);goal=rgb_tensor(cached['goal'],inverse);goal_feature=inverse.vision(goal)
        value,diagnostic=(None,{}) if mode=='direct' else goal_values(world,inverse,bank,goal,goal_feature,mode,neighbors,cost)
        images=collections.deque([image],maxlen=8);history=collections.deque(maxlen=7);actions=[];flags=[];planning=[];frames=[(image[0]*255).round().byte().cpu().numpy()]
        try:
            for step in range(budget):
                if mode=='direct':
                    f=inverse.vision(image);a=inverse.inverse(torch.cat([f,goal_feature,goal_feature-f],1)).float().clamp(-1,1)[0];detail=dict(selected_node=-1)
                else:
                    hidden,current,f=observed_state(world,inverse,images,history);a,detail=action(world,inverse,bank,hidden,current,f,goal,goal_feature,value,mode)
                _,_,term,trunc,info=env.step(a.cpu().numpy().astype(np.float32));actions.append(a.cpu().numpy().copy());flags.append(bool(info['success']))
                image=rgb_tensor(pixels(env),inverse);images.append(image);history.append(a.clone());planning.append(detail)
                if (step+1)%50==0 or flags[-1]:frames.append((image[0]*255).round().byte().cpu().numpy())
                if flags[-1] or term or trunc:break
        finally:close_environment(env)
        trace=output/f'{case["task"]}-seed{case["seed"]}.npz'
        np.savez_compressed(trace,actions=np.asarray(actions,np.float32),successes=np.asarray(flags,bool),sampled_actual_RGB=np.stack(frames))
        rows.append(dict(task=case['task'],seed=case['seed'],case_sha256=case['sha256'],success=bool(any(flags)),steps=len(actions),
            trace=str(trace),trace_sha256=sha(trace),initial_rgb_replay=gate,planning=planning,goal_graph_diagnostic=diagnostic,
            sensor_frames=len(actions)+1,expert_goal_success=case['expert_any_success']))
    assert before==(tensor_sha(world),tensor_sha(inverse))
    save(output/'metrics.json',dict(passed=True,utc=now(),episodes=rows,mode=mode,training_parent_seed=seed,state_before=before,state_after=before,
        ancestry=ancestry,graph_ready_sha256=None if mode=='direct' else sha(ROOT/'GRAPH_READY.json'),
        model_updates=0,budget_native=budget,camera_native_stride=1,execute_native=1,goal_only_in_planner=True,
        no_provenance_trajectory_order_or_logged_future_commands_read=True,no_state_reward_taskID_success_model_inputs=True,
        extra_TRAIN_memory_data_and_graph_compute=True,not_public_SOTA_or_equal_information_budget=True))


def preflight():
    p=read(EVAL/'plan.json');source=Path(p['source']);manifest=EVAL/'manifests/push-v3.json';total_native=0
    for seed in [1,2]:
        for mode in ['direct','world']:
            out=EVAL/f'technical/seed{seed}-{mode}'
            command([PYTHON,str(Path(__file__).resolve()),'evaluate','--seed',seed,'--cost',mode,'--manifest',manifest,'--output',out,'--budget','20'],
                out.with_suffix('.log'),environment(source,seed-1),source,p['deadline'])
            d=read(out/'metrics.json');assert d['passed'] and d['state_before']==d['state_after'];total_native+=sum(r['steps'] for r in d['episodes'])
            if mode=='direct':
                old=read(BASE/f'runs/metaworld_input_servo_20261109/evaluation/seed{seed}-execute1-push-v3/metrics.json')
                with np.load(d['episodes'][0]['trace']) as a,np.load(old['episodes'][0]['trace']) as b:
                    assert np.array_equal(a['actions'],b['actions'][:20]) and np.array_equal(a['successes'],b['successes'][:20])
    save(EVAL/'PREFLIGHT.json',dict(passed=True,utc=now(),actual_native_steps=total_native,native_step_budget=80,two_direct_controls_exact_to_original20=True,
        two_graph_controls_up_to20_frozen_and_bounded=True,model_updates=0))


def work(host,gpu):
    p=read(EVAL/'plan.json');source=Path(p['source']);assert sha(__file__)==p['helper_sha256']
    for j in p['jobs']:
        claim=EVAL/'claims'/(j['name']+'.json');claim.parent.mkdir(exist_ok=True)
        try:
            with claim.open('x') as f:json.dump(dict(host=host,gpu=gpu,pid=os.getpid(),utc=now()),f)
        except FileExistsError:continue
        try:command([PYTHON,str(Path(__file__).resolve()),'evaluate','--seed',j['seed'],'--cost',j['mode'],
            '--manifest',j['manifest'],'--output',j['output']],Path(j['output']).with_suffix('.log'),environment(source,gpu),source,p['deadline'],cap=3600)
        except BaseException:
            save(EVAL/'errors'/(j['name']+'.json'),dict(utc=now(),traceback=traceback.format_exc()));raise


def launch(host):
    p=read(EVAL/'plan.json');assert read(EVAL/'PREFLIGHT.json')['passed'];source=Path(p['source']);workers=[]
    for gpu in range(8):
        with (EVAL/f'worker-{host}-{gpu}.log').open('x') as f:
            child=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'work','--host',host,'--gpu',str(gpu)],cwd=source,
                env=environment(source,gpu)|{'CUDA_VISIBLE_DEVICES':''},stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(gpu=gpu,pid=child.pid))
    save(EVAL/f'launch-{host}.json',dict(utc=now(),host=host,workers=workers));print(host,workers,flush=True)


def collect():
    p=read(EVAL/'plan.json');cache={}
    while time.time()<p['deadline']:
        for inherited,jobs in [(True,p['inherited']),(False,p['jobs'])]:
            for j in jobs:
                path=Path(j['output'])/'metrics.json'
                if j['name'] in cache or not path.exists():continue
                d=read(path);assert d['passed'] and d['state_before']==d['state_after']
                for row in d['episodes']:
                    assert sha(row['trace'])==row['trace_sha256'] and row['initial_rgb_replay']['passed']
                    with np.load(row['trace']) as z:
                        assert z['actions'].shape==(row['steps'],4) and np.isfinite(z['actions']).all() and np.abs(z['actions']).max()<=1
                        assert bool(z['successes'].any())==row['success']
                cache[j['name']]=dict(mode='direct' if inherited else j['mode'],metrics=d,inherited=inherited)
        groups=collections.defaultdict(list)
        for x in cache.values():groups[x['mode']].extend(x['metrics']['episodes'])
        reports={k:dict(episodes=len(v),hard_successes=sum(r['success'] for r in v if r['task'] in HARD),hard_episodes=sum(r['task'] in HARD for r in v),
            by_task={t:dict(successes=sum(r['success'] for r in v if r['task']==t),episodes=sum(r['task']==t for r in v)) for t in sorted({r['task'] for r in v})}) for k,v in groups.items()}
        errors=list(map(str,EVAL.glob('errors/*.json')));save(EVAL/'results/summary.json',dict(utc=now(),combined_episodes=len(cache),planned=48,groups=reports,errors=errors,model_updates=0))
        if len(cache)==48:
            save(EVAL/'CLOSED.json',dict(passed=not errors,utc=now(),actual48_trace_SHA_action_success_verified=True,automatic_long_promotion=False));return
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','preflight','evaluate','work','launch','collect']);p.add_argument('--seed',type=int);p.add_argument('--cost',choices=['world','appearance','direct']);p.add_argument('--host');p.add_argument('--gpu',type=int)
    for k in ['manifest','output']:p.add_argument('--'+k,type=Path)
    p.add_argument('--budget',type=int,default=500);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='preflight':preflight()
    elif a.mode=='evaluate':evaluate(a.seed,a.cost,a.manifest,a.output,a.budget)
    elif a.mode=='work':work(a.host,a.gpu)
    elif a.mode=='launch':launch(a.host)
    else:collect()
