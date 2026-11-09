"""Goal-image CEM/MPC; the model never sees a desired goal or inverse output."""
import argparse
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment
from .common import read,save,sha,tensor_sha,now
from .train import build


def rgb_tensor(frame,model):
    raw=torch.from_numpy(np.asarray(frame).copy()).permute(2,0,1)[None].cuda()
    # Match offline storage exactly: native rendering ->224uint8 ->128uint8.
    if raw.shape[-2:] != (224,224):
        raw=torch.nn.functional.interpolate(raw.float(),(224,224),mode='bilinear',align_corners=False,antialias=True).round().clamp(0,255).byte()
    return model.pixels(raw)


@torch.no_grad()
def plan(model,hidden,image,goal,generator,horizon,warm=None,particles=64,iterations=4):
    assert horizon in [30,60] and horizon%5==0
    blocks=horizon//5
    mean=image.new_zeros(blocks,4) if warm is None else warm.reshape(blocks,5,4).mean(1)
    std=torch.ones_like(mean);best=None;best_score=float('inf');first_bank=None
    for iteration in range(iterations):
        # Draw the common maximum horizon before truncation, so H30/H60 have
        # identical first-decision action-bank prefixes for the same case.
        raw=torch.randn(particles,12,4,device=image.device,generator=generator)
        candidates=(mean[None]+std[None]*raw[:,:blocks]).clamp(-1,1)
        candidates[0]=mean
        if best is not None:candidates[1]=best
        if iteration==0:first_bank=__import__('hashlib').sha256(candidates[:,:6].cpu().numpy().tobytes()).hexdigest()
        state=hidden.expand(particles,-1,-1,-1).clone();future=image.expand(particles,-1,-1,-1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for t in range(horizon):state,future=model.predict(state,future,candidates[:,t//5])
        costs=(future-goal).square().mean((1,2,3))
        assert torch.isfinite(costs).all()
        values,indices=costs.topk(min(10,particles),largest=False)
        elite=candidates[indices]
        if float(values[0])<best_score:best_score=float(values[0]);best=elite[0].clone()
        mean=elite.mean(0);std=elite.std(0,unbiased=False).clamp_min(1e-6)
    return best[:,None].expand(-1,5,-1).reshape(horizon,4),dict(predicted_terminal_goal_mse=best_score,first_bank_prefix_sha256=first_bank)


def run(config,step,manifest,output,horizon,execute,budget=500,particles=64,iterations=4):
    cfg=read(config);torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    assert execute in [2,15]
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    model=build(cfg).cuda().eval().requires_grad_(False)
    weights=Path(cfg['output'])/f'model-step{step:07d}/weights.pt'
    model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True),strict=True)
    before=tensor_sha(model);cases=read(manifest)['rows'];rows=[];started=time.monotonic()
    with torch.inference_mode():
        for case in cases:
            assert sha(case['path'])==case['sha256'];cached=np.load(case['path'])
            env,state,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
            assert np.array_equal(state,cached['initial_state'])
            assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
            gate=rgb_replay_check(cached['initial'],pixels(env));assert gate['passed'],gate
            image=rgb_tensor(cached['initial'],model);goal=rgb_tensor(cached['goal'],model)
            with torch.autocast('cuda',dtype=torch.bfloat16):hidden=model.observe(model.initial(1),image)
            actions=[];successes=[];planning=[];warm=None
            generator=torch.Generator(device='cuda').manual_seed(610000+case['seed'])
            try:
                while len(actions)<budget:
                    commands,detail=plan(model,hidden,image,goal,generator,horizon,warm,particles,iterations)
                    count=min(execute,budget-len(actions));predicted_hidden=hidden;predicted_image=image
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        for command in commands[:count]:
                            _,_,terminated,truncated,info=env.step(command.cpu().numpy().astype(np.float32))
                            actions.append(command.cpu().numpy().copy());successes.append(bool(info['success']))
                            predicted_hidden,predicted_image=model.predict(predicted_hidden,predicted_image,command[None])
                            if successes[-1] or terminated or truncated:break
                    actual=rgb_tensor(pixels(env),model)
                    detail.update(executed=len(actions)-(sum(p['executed'] for p in planning)),
                        factual_prefix_rgb_mse=float((predicted_image-actual).square().mean()),
                        predicted_prefix_goal_mse=float((predicted_image-goal).square().mean()),
                        actual_prefix_goal_mse=float((actual-goal).square().mean()))
                    planning.append(detail)
                    image=actual
                    with torch.autocast('cuda',dtype=torch.bfloat16):hidden=model.observe(predicted_hidden,image)
                    if successes[-1] or terminated or truncated:break
                    warm=torch.cat([commands[count:],commands[-1:].expand(count,-1)],0)
            finally:close_environment(env)
            trace=output/f'{case["task"]}-seed{case["seed"]}.npz'
            np.savez_compressed(trace,actions=np.asarray(actions,dtype=np.float32),successes=np.asarray(successes,dtype=bool))
            row=dict(task=case['task'],seed=case['seed'],case_sha256=case['sha256'],success=bool(any(successes)),
                steps=len(actions),initial_rgb_replay=gate,trace=str(trace),trace_sha256=sha(trace),
                expert_goal_success=case['expert_any_success'],planning=planning)
            rows.append(row);save(output/'progress.json',dict(rows=rows,utc=now()))
            print(case['task'],case['seed'],row['success'],len(actions),flush=True)
    assert tensor_sha(model)==before
    result=dict(passed=True,utc=now(),episodes=rows,weights_sha256=sha(weights),model_state_before=before,model_state_after=before,
        horizon_native=horizon,execute_native=execute,particles=particles,iterations=iterations,action_block_native=5,
        budget_native=budget,checkpoint_step=step,seconds=time.monotonic()-started,model_updates=0,
        goal_only_in_planner=True,inverse_not_deployed=True,expert_intermediate_frames_or_actions_used=False,
        scope='Samecachedphysicalcases/RGBgoals;native-steprecurrentmodel. CEMterminalRGBMSE,ownGaussian/warmcontrols,held5nativeactionblocks. '
              'H30/H60andexecute2/15factorial;not identical to prior20DcommonCEM300x15 or publishedprotocol. '
              'No state/reward/taskID/language/goal inputs to worldmodel;simulatorstateonlyreset/audit. Scriptedexpertgoalsarefinalimagesonly;goal failuresretained.')
    save(output/'metrics.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['config','manifest','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--horizon',type=int,choices=[30,60],required=True)
    p.add_argument('--execute',type=int,choices=[2,15],required=True);p.add_argument('--budget',type=int,default=500)
    p.add_argument('--particles',type=int,default=64);p.add_argument('--iterations',type=int,default=4);a=p.parse_args()
    try:run(a.config,a.step,a.manifest,a.output,a.horizon,a.execute,a.budget,a.particles,a.iterations)
    except BaseException:
        save(a.output/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise
