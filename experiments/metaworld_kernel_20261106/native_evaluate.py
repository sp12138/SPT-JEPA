"""Frozen forward-only control:15imagined native steps,then15real steps.

Displacement maps compose before rendering from the observed image,avoiding
repeated texture interpolation.Only own planned/executed controls enter history.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment
from experiments.metaworld_eval_20261103.evaluate import resize_tensor,model_sha


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def plan(model,current,goal,past,valid,native_steps=15,planner='tangent'):
    base=current;mapping=model.grid.permute(0,3,1,2).expand(len(current),-1,-1,-1).clone();commands=[]
    for _ in range(native_steps):
        if planner=='observation':
            from as_jepa_lewm.observation_projection import observation_projection
            action,flow=observation_projection(model,current,goal,valid,past)
        else:action,flow=model.choose_control(current,goal,valid,past)
        assert torch.isfinite(action).all()and action.abs().max()<=1
        commands.append(action[0].cpu().numpy());past=action;valid=valid.new_ones(len(valid))
        mapping=model.warp(mapping,flow)
        current=F.grid_sample(base,mapping.permute(0,2,3,1),mode='bilinear',padding_mode='border',align_corners=False)
    return commands


def run(a):
    from as_jepa_lewm.factory import build_model
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    a.output.mkdir(parents=True,exist_ok=False);cfg=json.loads((a.run/'config.json').read_text())
    model=build_model(cfg).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(a.run/f'model-step{a.step:07d}/weights.pt',map_location='cuda',weights_only=True));before=model_sha(model)
    cases=[r for r in json.loads(a.manifest.read_text())['rows']if r['task']in a.tasks and r['seed']in a.seeds]
    cases.sort(key=lambda r:(a.tasks.index(r['task']),a.seeds.index(r['seed'])));assert len(cases)==len(a.tasks)*len(a.seeds);rows=[]
    with torch.inference_mode():
        for case in cases:
            assert digest(Path(case['path']))==case['sha256'];cached=np.load(case['path'])
            env,state,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
            assert np.array_equal(state,cached['initial_state'])
            assert all(np.array_equal(v,cached['initial_physics_'+k])for k,v in physics_state(env).items())
            gate=rgb_replay_check(cached['initial'],pixels(env));assert gate['passed'],gate
            frames=[cached['initial']];actions=[];successes=[];calls=0;warm=None;planning=[]
            generator=torch.Generator(device='cuda').manual_seed(610000+case['seed'])
            goal=model.encoder(resize_tensor([cached['goal']]).cuda())
            try:
                while len(actions)<a.budget:
                    current=model.encoder(resize_tensor(frames[-1:]).cuda())
                    past=torch.as_tensor(actions[-1],device=current.device).reshape(1,4)if actions else current.new_zeros(1,4)
                    valid=current.new_tensor([bool(actions)])
                    if a.planner=='path':
                        if hasattr(model,'plan_controls'):
                            previous=model.encoder(resize_tensor(frames[-2:-1]if len(frames)>1 else frames[-1:]).cuda())
                            if a.path_metric=='rgb'and a.control_prior=='increment':
                                commands,word,detail=model.plan_controls(current,previous,goal,valid,past,generator,warm,particles=a.path_particles,iterations=a.path_iterations)
                            else:
                                from as_jepa_lewm.reachable_projection import project
                                commands,word,detail=project(model,current,previous,goal,valid,past,generator,warm,particles=a.path_particles,iterations=a.path_iterations,metric=a.path_metric,control_prior=a.control_prior)
                        else:
                            from as_jepa_lewm.path_projection import path_projection
                            commands,word,detail=path_projection(model,current,goal,valid,past,generator,warm,particles=a.path_particles,iterations=a.path_iterations)
                        commands=commands[:a.replan_native];shift=a.replan_native//5
                        warm=torch.cat([word[shift:],word[-1:].expand(shift,-1)],0);planning.append(detail)
                    else:commands=plan(model,current,goal,past,valid,a.replan_native,a.planner)
                    calls+=1
                    for j,command in enumerate(commands):
                        _,_,terminated,truncated,info=env.step(command.astype(np.float32))
                        actions.append(command.copy());successes.append(bool(info['success']))
                        if successes[-1]or terminated or truncated or len(actions)>=a.budget:break
                        if(j+1)%5==0:frames.append(pixels(env))
                    if successes[-1]or terminated or truncated:break
            finally:close_environment(env)
            trace=a.output/f'{case["task"]}-seed{case["seed"]}.npz'
            np.savez_compressed(trace,actions=np.asarray(actions),successes=np.asarray(successes))
            row=dict(task=case['task'],seed=case['seed'],success=bool(any(successes)),steps=len(actions),calls=calls,
                case_sha256=case['sha256'],expert_goal_success=case['expert_any_success'],initial_rgb=gate,
                trace=str(trace),trace_sha256=digest(trace),planning=planning)
            rows.append(row);save(a.output/'progress.json',dict(rows=rows));print(json.dumps(row),flush=True)
    assert model_sha(model)==before
    save(a.output/'summary.json',dict(passed=True,model_unchanged=True,model_sha256=before,rows=rows,step=a.step,
        no_inverse_action_objective_or_future_action_reference=True,from_scratch=True,planner=a.planner,replan_native=a.replan_native,path_metric=a.path_metric,control_prior=a.control_prior,
        scope=f'Same original LeRobot corpus/split,RGBgoals/physicalcases,500native any-success.Replan{a.replan_native}native,imagesat5native.'
              'Native training and ordered control word,different compute.No extra control labels/state/reward/teacher.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in['run','manifest','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--budget',type=int,default=500)
    p.add_argument('--planner',choices=['tangent','observation','path'],default='tangent');p.add_argument('--replan-native',type=int,choices=[5,15],default=15)
    p.add_argument('--path-particles',type=int,default=128);p.add_argument('--path-iterations',type=int,default=8)
    p.add_argument('--path-metric',choices=['rgb','reachable'],default='rgb');p.add_argument('--control-prior',choices=['increment','uniform'],default='increment')
    p.add_argument('--tasks',nargs='+',required=True);p.add_argument('--seeds',type=int,nargs='+',required=True);run(p.parse_args())
