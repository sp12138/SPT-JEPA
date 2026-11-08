"""Actual realized action-bank diagnostics. Audit data MUST NOT train learners.

The same physical branch point follows7zero native controls/8real observations.
Every candidate is actually simulated. No action optimum is used for training.
"""
import copy
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import mujoco
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,sha,tensor_sha,now
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment
from experiments.metaworld_transition_only_20261109.evaluate import rgb_tensor
from experiments.metaworld_transition_only_20261109.model import FactualRecurrent
from .model import build

ROOT=BASE/'runs/metaworld_realized_bank_20261109'
OLD=BASE/'runs/metaworld_transition_only_20261109'
MATCH=BASE/'runs/metaworld_horizon_match_20261109'
SPEC=int(mujoco.mjtState.mjSTATE_INTEGRATION)


def mjstate(env):
    a=np.empty(mujoco.mj_stateSize(env.model,SPEC));mujoco.mj_getState(env.model,env.data,a,SPEC);return a


def capture(env):
    return dict(mj=mjstate(env),python={k:copy.deepcopy(v) for k,v in env.__dict__.items()
        if isinstance(v,(np.ndarray,int,float,bool,np.generic))},rng=copy.deepcopy(env.np_random.bit_generator.state),
        global_rng=np.random.get_state(),model={k:np.asarray(getattr(env.model,k)).copy()
            for k in ['body_pos','site_pos','geom_pos','geom_quat','geom_rgba','site_rgba']})


def restore(env,s):
    for k,v in s['python'].items():env.__dict__[k]=copy.deepcopy(v)
    for k,v in s['model'].items():getattr(env.model,k)[:]=v
    env.np_random.bit_generator.state=copy.deepcopy(s['rng']);np.random.set_state(s['global_rng'])
    mujoco.mj_setState(env.model,env.data,s['mj'],SPEC);mujoco.mj_forward(env.model,env.data)


def rgb(env):return rgb_tensor(pixels(env),FactualRecurrent)


def byte(image):return (image[0].detach().cpu().clamp(0,1)*255).round().byte().permute(1,2,0).numpy()


def ranks(values):
    order=np.argsort(values,kind='stable');result=np.empty(len(values),dtype=float);start=0
    while start<len(order):
        stop=start+1
        while stop<len(order) and values[order[stop]]==values[order[start]]:stop+=1
        result[order[start:stop]]=(start+stop-1)/2;start=stop
    return result


def correlation(a,b):
    x,y=ranks(a),ranks(b)
    return float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else None


@torch.no_grad()
def physical(case):
    assert sha(case['path'])==case['sha256'];cached=np.load(case['path']);task=case['task']
    env,state,_=make_environment(task,case['seed'],cached['rand_vec'])
    assert np.array_equal(state,cached['initial_state'])
    assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
    gate=rgb_replay_check(cached['initial'],pixels(env));assert gate['passed']
    warm=[byte(rgb_tensor(cached['initial'],FactualRecurrent))]
    try:
        for _ in range(7):env.step(np.zeros(4,np.float32));warm.append(byte(rgb(env)))
        snapshot=capture(env);reference_raw=pixels(env);restore(env,snapshot);canonical=mjstate(env).copy()
        controls=np.random.default_rng(403009+case['seed']).standard_normal((64,12,4)).clip(-1,1).astype(np.float32)
        controls[0]=0;controls=np.repeat(controls,5,axis=1)
        real={30:[],60:[]};states={30:[],60:[]};successes=[];gates=[];zero=None
        for i in range(65):
            restore(env,snapshot);assert np.array_equal(mjstate(env),canonical),'Integrationstate restoration mismatch'
            replay=rgb_replay_check(reference_raw,pixels(env));assert replay['passed'],replay;gates.append(replay)
            flags=[];frames={};observations={}
            for t,a in enumerate(controls[i if i<64 else 0],1):
                obs,_,terminated,truncated,info=env.step(a);assert np.isfinite(obs).all();flags.append(bool(info['success']))
                assert not terminated and not truncated
                if t in [30,60]:frames[t]=byte(rgb(env));observations[t]=obs.copy()
            final=mjstate(env).copy()
            if i==0:zero=dict(state=final,obs=observations[60].copy(),success=flags,raw=pixels(env))
            if i==64:
                assert np.array_equal(final,zero['state']) and np.array_equal(observations[60],zero['obs']) and flags==zero['success'],'Zero replay after64interventions changed'
                repeat=rgb_replay_check(zero['raw'],pixels(env));assert repeat['passed']
            else:
                for h in [30,60]:real[h].append(frames[h]);states[h].append(observations[h])
                successes.append(flags)
        path=ROOT/f'{task}-seed{case["seed"]}.npz'
        np.savez_compressed(path,controls=controls,warm_history=np.stack(warm),goal=byte(rgb_tensor(cached['goal'],FactualRecurrent)),
            actual30=np.stack(real[30]),actual60=np.stack(real[60]),states30=np.stack(states[30]),states60=np.stack(states[60]),successes=np.asarray(successes))
        save(path.with_suffix('.json'),dict(passed=True,task=task,seed=case['seed'],case_sha256=case['sha256'],bank=str(path),bank_sha256=sha(path),
            source_initial_physics_exact=True,source_initial_rgb=gate,all65branch_initial_integrationstates_exact=True,
            zero_path_after64interventions_physics_obs_success_exact=True,zero_rgb=repeat,branch_rgb_gates=gates,
            candidates=64,horizon_native=60,warmup_native=7,physics_steps=7+65*60,expert_goal_success=case['expert_any_success'],utc=now()))
    finally:close_environment(env)


@torch.no_grad()
def compare(job,path):
    cfg=read(job['config']);weights=Path(cfg['output'])/f'model-step{job["step"]:07d}/weights.pt';seal=read(weights.parent/'SEALED.json')
    assert sha(weights)==seal['weights_sha256'];model=build(cfg).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True),strict=True);before=tensor_sha(model)
    data=np.load(path);past=torch.from_numpy(data['warm_history']).permute(0,3,1,2)[None].cuda()
    hidden,image=model.prepare(dict(history_pixels=past,history_actions=torch.zeros(1,7,4,device='cuda'),history_valid=torch.ones(1,8,device='cuda',dtype=torch.bool)))
    state=hidden.expand(64,-1,-1,-1).clone();future=image.expand(64,-1,-1,-1);actions=torch.from_numpy(data['controls']).cuda()
    goal=FactualRecurrent.pixels(torch.from_numpy(data['goal'].copy()).permute(2,0,1)[None].cuda());result={}
    for t in range(60):
        with torch.autocast('cuda',dtype=torch.bfloat16):state,future=model.predict(state,future,actions[:,t])
        if t+1 in [30,60]:
            h=t+1;actual=FactualRecurrent.pixels(torch.from_numpy(data['actual'+str(h)].copy()).permute(0,3,1,2).cuda())
            predicted=(future-goal).square().mean((1,2,3)).cpu().numpy();real=(actual-goal).square().mean((1,2,3)).cpu().numpy()
            error=(future-actual).square().mean((1,2,3)).cpu().numpy();copy_error=(image-actual).square().mean((1,2,3)).cpu().numpy()
            chosen=int(predicted.argmin());oracle=int(real.argmin());flags=data['successes'][:,:h].any(1)
            result[str(h)]=dict(predicted_goal_cost=predicted.tolist(),realized_goal_cost=real.tolist(),spearman=correlation(predicted,real),
                mean_forecast_mse=float(error.mean()),mean_copy_mse=float(copy_error.mean()),forecast_over_copy=float(error.sum()/max(copy_error.sum(),1e-20)),
                selected=chosen,realized_pixel_oracle=oracle,selected_success=bool(flags[chosen]),any_bank_success=bool(flags.any()),
                selected_cost=float(real[chosen]),best_realized_cost=float(real[oracle]),bank_mean_realized_cost=float(real.mean()),
                realized_cost_regret_over_random=float((real[chosen]-real.min())/max(real.mean()-real.min(),1e-20)))
    assert before==tensor_sha(model)
    save(ROOT/'predictions'/f'{job["name"]}-{path.stem}.json',dict(passed=True,job=job['name'],task=path.stem,
        weights_sha256=sha(weights),bank_sha256=sha(path),model_before=before,model_after=tensor_sha(model),horizons=result,
        world_updates=0,head_updates=0,desired_goal_in_learner=False,utc=now()))


def run():
    ROOT.mkdir(exist_ok=False);torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.30)
    save(ROOT/'AUDIT_ONLY_NO_TRAINING.json',dict(utc=now(),reason='PreviouslyinspectedEVALUATIONphysicalcases;allbanks/frames/states/successes areAUDIT-ONLY. NeveruseforSGD/proposals/teachers.',
        candidates='Own64Gaussianheld5controls;allactuallyexecutedfromsamepost7zero-stepstate. GoalimageONLYscoring. Not512independentphysicalcases orpolicyepisodes.'))
    try:
        for case in read(OLD/'manifests/screen.json')['rows']:physical(case)
        jobs=[]
        for seed in [1,2]:jobs.append(dict(name=f'parentH15-seed{seed}',config=str(OLD/f'training/h15-inverse1-seed{seed}/config.json'),step=10000))
        for seed in [1,2]:
            for h in [15,60]:jobs.append(dict(name=f'continuedH{h}-seed{seed}',config=str(MATCH/f'training/h{h}-seed{seed}/config.json'),step=12000))
        deadline=time.time()+4*3600
        for job in jobs:
            cfg=read(job['config']);sealed=Path(cfg['output'])/f'model-step{job["step"]:07d}/SEALED.json'
            while not sealed.exists():assert time.time()<deadline;time.sleep(10)
            for path in sorted(ROOT.glob('*.npz')):compare(job,path)
        save(ROOT/'CLOSED.json',dict(passed=True,utc=now(),physical_cases=8,actual_candidate_paths=512,physics_steps=8*(7+65*60),
            frozen_models=6,prediction_comparisons=48,model_updates=0,scope='Realizedcounterfactualaudit,notphysicalpolicyperformance orindependent512caseconfirmation.'))
    except BaseException:
        save(ROOT/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise


if __name__=='__main__':run()
