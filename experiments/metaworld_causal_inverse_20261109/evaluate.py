"""Local current-input inference, optionally ranked by its factual forward world."""
import argparse
import hashlib
from pathlib import Path
import traceback
import numpy as np
import torch
from experiments.metaworld_transition_only_20261109.common import read,save,sha,tensor_sha,now
from experiments.metaworld_transition_only_20261109.evaluate import rgb_tensor
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment
from .train import setup


@torch.no_grad()
def run(config,step,manifest,output,budget,selection='forward'):
    cfg=read(config);device,world,_,_=setup(cfg);world.eval().requires_grad_(False);body=world.body
    weights=Path(cfg['output'])/f'model-step{step:07d}/weights.pt'
    assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256']
    world.load_state_dict(torch.load(weights,map_location=device,weights_only=True),strict=True)
    output=Path(output);output.mkdir(exist_ok=False);before=tensor_sha(world);rows=[]
    for case in read(manifest)['rows']:
        assert sha(case['path'])==case['sha256'];cached=np.load(case['path'])
        env,state,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
        assert np.array_equal(state,cached['initial_state'])
        assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
        gate=rgb_replay_check(cached['initial'],pixels(env));assert gate['passed']
        goal=rgb_tensor(cached['goal'],body);history=[torch.from_numpy(cached['initial'].copy()).permute(2,0,1)]
        actions=[];flags=[];planning=[];generator=torch.Generator(device=device).manual_seed(1075000+case['seed'])
        try:
            while len(actions)<budget:
                n=len(history);times=list(range(n-8,n))
                batch=dict(history_pixels=torch.stack([history[max(0,t)] for t in times])[None].to(device),
                    history_actions=torch.tensor(np.array([actions[t] if t>=0 else np.zeros(4,np.float32) for t in times[:-1]]),device=device)[None],
                    history_valid=torch.tensor([[t>=0 for t in times]],device=device))
                hidden,image=body.prepare(batch)
                rng_sha=hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()
                proposals=world.proposals(hidden,image,goal,generator,16)
                assert torch.isfinite(proposals).all() and float(proposals.abs().max())<=1
                if selection=='forward':
                    h=hidden.expand(16,-1,-1,-1).clone();forecast=image.expand(16,-1,-1,-1)
                    for _ in range(5):h,forecast=body.predict(h,forecast,proposals)
                    scores=(forecast-goal).square().mean((1,2,3));index=int(scores.argmin());command=proposals[index]
                    score=float(scores[index])
                else:
                    command=proposals.mean(0) if world.mode.startswith('flow_') else proposals[0]
                    score=None;index=-1
                a=command.cpu().numpy().astype(np.float32)
                _,_,terminated,truncated,info=env.step(a);actions.append(a.copy());flags.append(bool(info['success']))
                actual=pixels(env);history.append(torch.from_numpy(actual.copy()).permute(2,0,1))
                planning.append(dict(first_proposal_RNG_sha256=rng_sha,selected_index=index,score=score,
                    actual_rgb_goal_mse=float((rgb_tensor(actual,body)-goal).square().mean())))
                if flags[-1] or terminated or truncated:break
        finally:close_environment(env)
        trace=output/f'{case["task"]}-seed{case["seed"]}.npz'
        np.savez_compressed(trace,actions=np.asarray(actions,np.float32),successes=np.asarray(flags,bool),
            sampled_actual_RGB=np.stack([x.numpy() for x in history[::50]]))
        rows.append(dict(task=case['task'],seed=case['seed'],case_sha256=case['sha256'],success=bool(any(flags)),steps=len(actions),
            initial_rgb_replay=gate,trace=str(trace),trace_sha256=sha(trace),planning=planning))
        save(output/'progress.json',dict(utc=now(),rows=rows))
    assert before==tensor_sha(world)
    save(output/'metrics.json',dict(passed=True,utc=now(),episodes=rows,mode=cfg['mode'],step=step,selection=selection,
        weights_sha256=sha(weights),state_before=before,state_after=before,model_updates=0,
        control_hold_prediction=5,execute_native=1,candidates=16,flow_euler_steps=16,budget_native=budget,
        goal_only_in_planner=True,inverse_target='SAMEknowninputa0usedatALLfiveforwardsteps;neverunknownfuture/optimalcommands.',
        scope='Ownscratch12k+newjointupdates;EXTRAordinaryrandominterventionsandoldconstant-inputlogs. GN/samplelevel/FP32. '
            'CameraEVERYnative;8inspecteddevelopmentalcasesx2parents,notfairpublicSOTA/independentcases/equalFLOPs.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['config','manifest','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--budget',type=int,default=500)
    p.add_argument('--selection',choices=['direct','forward'],default='forward');a=p.parse_args()
    try:run(a.config,a.step,a.manifest,a.output,a.budget,a.selection)
    except BaseException:
        save(a.output/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise
