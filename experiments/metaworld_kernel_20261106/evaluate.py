"""RGB-only predictive-kernel control; all commands inferred from a forward model."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment
from experiments.metaworld_eval_20261103.evaluate import resize_tensor,model_sha


def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def plan(model,previous,current,goal,valid,execute,seed,objective='projection',previous_control=None):
    commands=[]
    for j in range(execute):
        if hasattr(model,'command'):
            command,future=model.command(previous,current,goal,valid,objective,previous_control)
            assert torch.isfinite(command).all()and command.abs().max()<=1
            commands.append(command[0].cpu().numpy());previous,current=current,future;valid=valid.new_ones(1);previous_control=command
            continue
        zero=current.new_zeros(1,4);flag=current.new_zeros(1)
        if model.boundary:
            desired,_,target_var=model.fields(previous,current,goal,zero,flag+1,valid)
        else:desired,target_var=goal-current,torch.zeros_like(current)
        if model.affine:
            drift,effect,var=model.fields(previous,current,torch.zeros_like(goal),zero,flag,valid)
            command=model.box_projection(drift,effect,desired,var+target_var)
            future=current+drift+torch.einsum('na,nachw->nchw',command,effect)
        else:
            from stable_worldmodel.planning.solver import CEMSolver
            import gymnasium as gym
            class Cost:
                def get_cost(self,info,actions):
                    raw=actions[:,:,0].flatten(0,1).clamp(-1,1);n=len(raw)
                    drift,_,var=model.fields(previous.expand(n,-1,-1,-1),current.expand(n,-1,-1,-1),
                        torch.zeros_like(goal).expand(n,-1,-1,-1),raw,flag.expand(n),valid.expand(n))
                    variance=var+target_var;error=(drift-desired).square()
                    return (.5*(error/variance+variance.log())).flatten(1).mean(1).reshape(actions.shape[:2])
            solver=CEMSolver(Cost(),batch_size=1,num_samples=300,n_steps=15,topk=10,device='cuda',seed=seed+j)
            solver.configure(action_space=gym.spaces.Box(-1.,1.,shape=(1,4),dtype=np.float32),n_envs=1,
                config=SimpleNamespace(horizon=1,action_block=1))
            result=solver.solve(dict(current=current),init_action=torch.zeros(1,1,4))
            command=result['actions'][:,0].cuda().clamp(-1,1)
            drift,_,_=model.fields(previous,current,torch.zeros_like(goal),command,flag,valid);future=current+drift
        assert torch.isfinite(command).all()and command.abs().max()<=1
        commands.append(command[0].cpu().numpy());previous,current=current,future;valid=valid.new_ones(1)
    return commands


def run(a):
    from as_jepa_lewm.factory import build_model
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    a.output.mkdir(parents=True,exist_ok=False);cfg=json.loads((a.run/'config.json').read_text())
    model=build_model(cfg).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(a.run/f'model-step{a.step:07d}/weights.pt',map_location='cuda',weights_only=True))
    before=model_sha(model)
    cases=[r for r in json.loads(a.manifest.read_text())['rows']if r['task']in a.tasks and r['seed']in a.seeds]
    cases.sort(key=lambda r:(a.tasks.index(r['task']),a.seeds.index(r['seed'])))
    assert len(cases)==len(a.tasks)*len(a.seeds);rows=[]
    with torch.inference_mode():
        for case in cases:
            assert digest(case['path'])==case['sha256'];cached=np.load(case['path'])
            env,state,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
            assert np.array_equal(state,cached['initial_state'])
            assert all(np.array_equal(v,cached['initial_physics_'+k])for k,v in physics_state(env).items())
            gate=rgb_replay_check(cached['initial'],pixels(env));assert gate['passed'],gate
            frames=[cached['initial']];actions=[];successes=[];calls=0
            goal=model.encoder(resize_tensor([cached['goal']]).cuda())
            try:
                while len(actions)<a.budget:
                    current=model.encoder(resize_tensor(frames[-1:]).cuda())
                    previous=model.encoder(resize_tensor(frames[-2:-1]if len(frames)>1 else frames[-1:]).cuda())
                    commands=plan(model,previous,current,goal,current.new_tensor([len(frames)>1]),a.execute,
                        20261106+case['seed']*1009+len(actions)*31,a.objective,
                        torch.as_tensor(actions[-1],device=current.device).reshape(1,4)if actions else current.new_zeros(1,4))
                    calls+=1
                    for command in commands:
                        for _ in range(5):
                            _,_,terminated,truncated,info=env.step(command.astype(np.float32))
                            actions.append(command.copy());successes.append(bool(info['success']))
                            if successes[-1]or terminated or truncated or len(actions)>=a.budget:break
                        if successes[-1]or terminated or truncated or len(actions)>=a.budget:break
                        frames.append(pixels(env))
                    if successes[-1]or terminated or truncated:break
            finally:close_environment(env)
            trace=a.output/f'{case["task"]}-seed{case["seed"]}.npz'
            np.savez_compressed(trace,actions=np.asarray(actions),successes=np.asarray(successes))
            row=dict(task=case['task'],seed=case['seed'],success=bool(any(successes)),steps=len(actions),calls=calls,
                case_sha256=case['sha256'],expert_goal_success=case['expert_any_success'],initial_rgb=gate,
                trace=str(trace),trace_sha256=digest(trace))
            rows.append(row);save(a.output/'progress.json',dict(rows=rows));print(json.dumps(row),flush=True)
    assert model_sha(model)==before
    save(a.output/'summary.json',dict(passed=True,model_unchanged=True,model_sha256=before,rows=rows,step=a.step,objective=a.objective,
        no_reconstruction=True,no_action_targets_or_references=True,random_features_only=True,
        scope='IndependentLeRobot50task training corpus;sharedRGBgoals/cases,500native any-success.RGBobservations only. '
              'Commands solved from learned future-observation dynamics,not predicted from actionlabels.No oldweights/teacher.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in['run','manifest','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--budget',type=int,default=500);p.add_argument('--execute',type=int,default=3)
    p.add_argument('--objective',choices=['projection','ratio','marginal'],default='projection')
    p.add_argument('--tasks',nargs='+',required=True);p.add_argument('--seeds',type=int,nargs='+',required=True);run(p.parse_args())
