"""Public DROID GD/hierarchical planner transferred as180-native-step blocks.

Uses the pinned solver code, including native subgoal construction and queue
priors. Kept separate from the shared-CEM track because compute/feedback differ.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch

from experiments.metaworld_baselines_20261107.models import PublicBaseline,load_file
from experiments.metaworld_baselines_20261107.train_native import digest,save
from experiments.metaworld_eval_20261103.evaluate import resize_tensor,file_sha
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment

BASE=Path('/efs/prescottsun/SPT-JEPA')


def build(model,lower,upper,seed):
    # Load only the exact public solver namespace; retain the current environment
    # and data packages. Both source files are pinned by the native queue plan.
    import stable_worldmodel
    package=types.ModuleType('stable_worldmodel.solver')
    package.__path__=[str(BASE/'runs/hjepa_droid_20261105/public/stable_worldmodel/solver')]
    sys.modules['stable_worldmodel.solver']=package;stable_worldmodel.solver=package
    gd=importlib.import_module('stable_worldmodel.solver.gd')
    native=load_file('hjepa_hierarchical_solver',model.sources/'hjepa/h_jepa/hierarchical_solver.py')
    solvers={};levels=list(model.core.jepas)

    class PhysicalBox(native.LevelCostModel):
        def get_cost(self,info,actions):
            if self.level==1:
                with torch.no_grad():actions.clamp_(min=lower,max=upper)
            return super().get_cost(info,actions)

    class FlatPhysicalBox(torch.nn.Module):
        def __init__(self,core):
            super().__init__();self.model=core;self.level=1
        def get_cost(self,info,actions):
            with torch.no_grad():actions.clamp_(min=lower,max=upper)
            # Flat JEPA owns goal encoding; LevelCostModel expects embeddings
            # supplied by the hierarchy and cannot replace this public method.
            return self.model.get_cost(info,actions)
        def rollout(self,info,actions):return self.model.rollout(info,actions)
        def get_latent_action_queue(self):return self.model.get_latent_action_queue()

    for level,core in enumerate(levels,start=1):
        wrapped=(FlatPhysicalBox(core) if len(levels)==1 else PhysicalBox(model=core,upper_models=levels[level:]))
        # H-JEPA's actual DROID release cell uses lr.01 at BOTH levels;
        # the config default lr.3 is the HWM cell, not H-JEPA (droid_eval.py).
        samples=32 if len(levels)==1 else (16 if level==1 else 4)
        solvers[level]=gd.GradientSolver(wrapped,n_steps=90,min_its=30,batch_size=1,var_scale=1.5,
            num_samples=samples,action_noise=0.,early_stop_patience=5,early_stop_rel_delta=.01,
            action_clip_sigma=2.,device='cuda',seed=seed,optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs=dict(lr=.01,weight_decay=.01))
    def cfg(**extra):
        return types.SimpleNamespace(receding_horizon=36,action_block=1,action_dim=None,
            pre_decay_horizon=None,num_subgoals=1,cost_last_n=1,intermediate_cost_weight=1.,
            action_cost_weight=0.,horizon_one_goal_cost_space='lower',**extra)
    # Policy-level units are macro steps; its native scheduler counts physical
    # steps through action_block. Each plan is the original single open-loop36
    # macro-step DROID window; repeat cold blocks for the500step MetaWorld budget.
    fine=cfg(horizon=36);fine.action_block=5
    if len(levels)==2:
        fine.cost_last_n=999;fine.pre_decay_horizon=36;fine.intermediate_cost_weight=.5
        coarse=cfg(horizon=12);coarse.action_dim=8;coarse.num_subgoals=12
        config=types.SimpleNamespace(receding_horizon=36,level1=fine,level2=coarse)
        solver=native.HierarchicalSolver(level_solvers=solvers,level_models=levels)
    else:config=fine;solver=solvers[1]
    import gymnasium as gym
    solver.configure(action_space=gym.spaces.Box(-1.,1.,shape=(1,4),dtype=np.float32),n_envs=1,config=config)
    return solver,solvers


def main(config,checkpoint,manifest,output,budget):
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    cfg=json.loads(Path(config).read_text());assert cfg['method'] in ['lewm-idm','h-jepa']
    model=PublicBaseline(cfg['method'],cfg['sources'])
    weights=torch.load(Path(checkpoint)/'weights.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(weights,strict=True);del weights
    model=model.cuda().eval().requires_grad_(False)
    before=digest(model)
    card=json.loads((Path(cfg['data'])/'dataset_card.json').read_text())
    mean,std=np.asarray(card['action_mean'],np.float32),np.asarray(card['action_std'],np.float32)
    lower=torch.tensor((-1-mean)/std,device='cuda');upper=torch.tensor((1-mean)/std,device='cuda')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);rows=[];started=time.monotonic()
    for case in json.loads(Path(manifest).read_text())['rows']:
        assert file_sha(case['path'])==case['sha256'];cached=np.load(case['path'])
        env,initial,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
        assert np.array_equal(initial,cached['initial_state'])
        assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
        replay=rgb_replay_check(cached['initial'],pixels(env));assert replay['passed']
        frame=cached['initial'];goal=model.normalize(resize_tensor([cached['goal']])[None].cuda())
        solver,children=build(model,lower,upper,20261103+case['seed']*1009)
        success=[];actions=[];calls=0;terminated=truncated=False
        try:
            while len(success)<budget:
                info=dict(pixels=model.normalize(resize_tensor([frame])[None].cuda()),goal=goal,
                    action=torch.zeros(1,1,20,device='cuda'))
                # No expert commands, action labels, or desired-goal inverse head.
                # Gradients are with respect to proposed controls only.
                with torch.enable_grad():
                    if cfg['method']=='h-jepa':result=solver.solve(info,steps_taken=0,eval_budget=180)
                    else:result=solver.solve(info,planning_horizon=36,steps_taken=0,eval_budget=180)
                plan=result['actions'][0].numpy();calls+=1
                for normalized in plan:
                    raw=np.clip(normalized*std+mean,-1,1).reshape(5,4)
                    for action in raw:
                        _,_,terminated,truncated,info=env.step(action.astype(np.float32))
                        actions.append(action.copy());success.append(bool(info['success']))
                        if success[-1] or terminated or truncated or len(success)>=budget:break
                    if success[-1] or terminated or truncated or len(success)>=budget:break
                if success[-1] or terminated or truncated or len(success)>=budget:break
                frame=pixels(env)
        finally:close_environment(env)
        trace=out/f"{case['task']}-seed{case['seed']}.npz"
        np.savez_compressed(trace,actions=np.asarray(actions),successes=np.asarray(success))
        row=dict(task=case['task'],seed=case['seed'],success=any(success),steps=len(success),plan_calls=calls,
            expert_goal_success=case['expert_any_success'],case_sha256=case['sha256'],initial_rgb_replay=replay,
            trace=str(trace),trace_sha256=file_sha(trace),compute={str(k):v.solve_records for k,v in children.items()})
        rows.append(row);print(json.dumps(row),flush=True)
    assert digest(model)==before
    save(out/'metrics.json',dict(passed=True,method=cfg['method'],episodes=rows,model_updates=0,
        model_state_before=before,model_state_after=digest(model),weights_sha256=file_sha(Path(checkpoint)/'weights.pt'),
        budget=budget,seconds=time.monotonic()-started,
        controller=dict(native_public_droid_GD=True,fine_horizon_macro=36,coarse_horizon=12 if cfg['method']=='h-jepa' else None,
            native_subgoals=12 if cfg['method']=='h-jepa' else None,execute_macro_per_block=36,frameskip=5,
            observations_per_plan=1,physical_action_box_projection=True,native_prior_clip_sigma=2.,
            note='Cold180native-step open-loop blocks repeat to500native steps; no global decay between blocks. '
                'Preserves the source DROID single-window planner. Feedback/computation differ from shared CEM; report separately.'),
        scope='Native complete hierarchical planner transfer with learned action priors, without future expert controls. '
            'Not the common-controller comparison or a published MetaWorld reproduction.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['config','checkpoint','manifest','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--budget',type=int,default=500);a=p.parse_args();main(a.config,a.checkpoint,a.manifest,a.output,a.budget)
