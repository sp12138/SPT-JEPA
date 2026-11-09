"""A desired final image is scored as a hypothetical endpoint at inference."""
import argparse
import hashlib
from pathlib import Path
import traceback
import numpy as np
import torch
from experiments.metaworld_transition_only_20261109.common import read,save,sha,tensor_sha,now
from experiments.metaworld_transition_only_20261109.evaluate import rgb_tensor
from experiments.metaworld_endpoint_density_20261109.model import advance
from experiments.metaworld_endpoint_density_20261109.energy import cost as image_cost
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment
from .train import setup


class Readout:
    def __init__(self,world):self.world=world
    def __call__(self,noisy,time,hidden,current,durations):
        return self.world.velocity(noisy,time,hidden,current,durations)


@torch.no_grad()
def cost(world,prior,current,goal,seed):
    goal=goal.expand(len(prior),-1,-1,-1)
    with torch.autocast('cuda',dtype=torch.bfloat16):p,logp,q,logq=world.distribution(prior,goal)
    result=world.kl(p,logp,q,logq)
    generator=torch.Generator(device=prior.device).manual_seed(seed+1)
    for _ in range(2):
        noise=torch.randn(1,128,16,16,device=prior.device,generator=generator)
        z=q+logq.exp()*noise
        result+=image_cost(Readout(world),z,current,goal[:1],60,world.flow_mode,seed)/2
    assert torch.isfinite(result).all();return result


@torch.no_grad()
def plan(world,hidden,image,goal,generator,warm):
    mean=image.new_zeros(12,4) if warm is None else warm.reshape(12,5,4).mean(1)
    std=torch.ones_like(mean);best=None;best_score=float('inf');first=None
    energy_seed=int(hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()[:15],16)
    for iteration in range(4):
        raw=torch.randn(64,12,4,device=image.device,generator=generator)
        candidates=(mean[None]+std[None]*raw).clamp(-1,1);candidates[0]=mean
        if best is not None:candidates[1]=best
        if iteration==0:first=hashlib.sha256(candidates[:,:6].cpu().numpy().tobytes()).hexdigest()
        state=hidden.expand(64,-1,-1,-1).clone()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for t in range(60):state=advance(world.parent,state,candidates[:,t//5])
        scores=cost(world,state,(image*2-1).expand(64,-1,-1,-1),goal*2-1,energy_seed)
        values,indices=scores.topk(10,largest=False);elite=candidates[indices]
        if float(values[0])<best_score:best_score=float(values[0]);best=elite[0].clone()
        mean=elite.mean(0);std=elite.std(0,unbiased=False).clamp_min(1e-6)
    return best[:,None].expand(-1,5,-1).reshape(60,4),dict(planner_score=best_score,
        score_kind=('expected_mean_RGB_goal_error' if world.flow_mode=='mean' else 'expected_finite_image_NELBO'),first_bank_prefix_sha256=first,
        renderer_queries_per_candidate=10,latent_samples=2)


def run(config,step,manifest,output,budget):
    cfg=read(config);device,world,_,_=setup(cfg);world.eval().requires_grad_(False);parent=world.parent
    weights=Path(cfg['output'])/f'model-step{step:07d}/weights.pt'
    assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256']
    world.load_state_dict(torch.load(weights,map_location=device,weights_only=True),strict=True)
    output=Path(output);output.mkdir(exist_ok=False);before=tensor_sha(world);rows=[]
    with torch.inference_mode():
        for case in read(manifest)['rows']:
            assert sha(case['path'])==case['sha256'];cached=np.load(case['path'])
            env,state,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
            assert np.array_equal(state,cached['initial_state'])
            assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
            gate=rgb_replay_check(cached['initial'],pixels(env));assert gate['passed']
            image=rgb_tensor(cached['initial'],parent);goal=rgb_tensor(cached['goal'],parent)
            with torch.autocast('cuda',dtype=torch.bfloat16):hidden=parent.observe(parent.initial(1),image)
            generator=torch.Generator(device='cuda').manual_seed(610000+case['seed'])
            actions=[];flags=[];planning=[];warm=None
            try:
                while len(actions)<budget:
                    commands,detail=plan(world,hidden,image,goal,generator,warm)
                    count=min(15,budget-len(actions))
                    for command in commands[:count]:
                        _,_,terminated,truncated,info=env.step(command.cpu().numpy().astype(np.float32))
                        actions.append(command.cpu().numpy().copy());flags.append(bool(info['success']))
                        with torch.autocast('cuda',dtype=torch.bfloat16):hidden=advance(parent,hidden,command[None])
                        if flags[-1] or terminated or truncated:break
                    image=rgb_tensor(pixels(env),parent)
                    with torch.autocast('cuda',dtype=torch.bfloat16):hidden=parent.observe(hidden,image)
                    detail.update(native_steps=len(actions),actual_rgb_goal_mse=float((image-goal).square().mean()))
                    planning.append(detail)
                    if flags[-1] or terminated or truncated:break
                    warm=torch.cat([commands[count:],commands[-1:].expand(count,-1)],0)
            finally:close_environment(env)
            trace=output/f'{case["task"]}-seed{case["seed"]}.npz'
            np.savez_compressed(trace,actions=np.asarray(actions,np.float32),successes=np.asarray(flags,bool))
            rows.append(dict(task=case['task'],seed=case['seed'],case_sha256=case['sha256'],success=bool(any(flags)),
                steps=len(actions),initial_rgb_replay=gate,expert_goal_success=case['expert_any_success'],
                trace=str(trace),trace_sha256=sha(trace),planning=planning))
            save(output/'progress.json',dict(utc=now(),rows=rows))
    assert before==tensor_sha(world)
    save(output/'metrics.json',dict(passed=True,utc=now(),episodes=rows,mode=cfg['mode'],step=step,
        weights_sha256=sha(weights),state_before=before,state_after=before,model_updates=0,
        horizon_native=60,execute_native=15,control_hold_native=5,particles=64,iterations=4,budget_native=budget,
        renderer_queries_per_candidate=10,goal_only_in_planner=True,expert_paths_or_action_targets=False,
        scope='Twoownfactualparents12000+ownhead2000+jointflowupdates. Same8developmentalcases;notpublicSOTA/equalbackboneFLOPs.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['config','manifest','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--budget',type=int,default=500);a=p.parse_args()
    try:run(a.config,a.step,a.manifest,a.output,a.budget)
    except BaseException:
        save(a.output/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise
