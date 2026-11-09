"""Protected contact-state interventions. NEVER training/teacher/proposal data."""
import argparse
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,sha,tensor_sha,now
from experiments.metaworld_transition_only_20261109.evaluate import rgb_tensor
from experiments.metaworld_transition_only_20261109.model import FactualRecurrent
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment
from experiments.metaworld_horizon_match_20261109.bank import capture,restore,mjstate,byte,correlation
from experiments.metaworld_endpoint_density_20261109.model import advance
from experiments.metaworld_belief_state_20261109.train import setup
from experiments.metaworld_belief_state_20261109.evaluate import cost

PARENT=BASE/'runs/metaworld_belief_state_20261109'
ROOT=PARENT/'contact-audit-v1'


@torch.no_grad()
def physical(prefix):
    reference=read(PARENT/'evaluation/joint_anchor-seed2-s500-push-v3/metrics.json')['episodes'][0]
    assert reference['success'] and reference['steps']==92 and sha(reference['trace'])==reference['trace_sha256']
    case=read(BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json')['rows']
    case=next(c for c in case if c['task']=='push-v3');assert case['sha256']==reference['case_sha256']
    assert sha(case['path'])==case['sha256'];cached=np.load(case['path']);logged=np.load(reference['trace'])
    env,state,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
    assert np.array_equal(state,cached['initial_state'])
    assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
    initial_gate=rgb_replay_check(cached['initial'],pixels(env));assert initial_gate['passed']
    observations=[byte(rgb_tensor(cached['initial'],FactualRecurrent))]
    try:
        for t,action in enumerate(logged['actions'][:prefix],1):
            state,_,term,trunc,info=env.step(action)
            assert bool(info['success'])==bool(logged['successes'][t-1]) and not term and not trunc
            if t%15==0:observations.append(byte(rgb_tensor(pixels(env),FactualRecurrent)))
        snapshot=capture(env);raw=pixels(env);restore(env,snapshot);canonical=mjstate(env).copy()
        initial_obj=np.asarray(state)[4:7].copy();target=np.asarray(env._target_pos).copy()
        controls=np.random.default_rng(470009).standard_normal((64,12,4)).clip(-1,1).astype(np.float32)
        controls[0]=0;controls=np.repeat(controls,5,axis=1)
        actual=[];states=[];flags=[];gates=[];zero=None
        for i in range(65):
            restore(env,snapshot);assert np.array_equal(mjstate(env),canonical)
            gate=rgb_replay_check(raw,pixels(env));assert gate['passed'];gates.append(gate)
            row=[]
            for a in controls[i if i<64 else 0]:
                state,_,term,trunc,info=env.step(a);assert not term and not trunc
                row.append(bool(info['success']))
            frame=byte(rgb_tensor(pixels(env),FactualRecurrent));integration=mjstate(env)
            if i==0:zero=dict(integration=integration.copy(),state=state.copy(),flags=row,frame=pixels(env))
            if i==64:
                assert np.array_equal(integration,zero['integration']) and np.array_equal(state,zero['state']) and row==zero['flags']
                final_gate=rgb_replay_check(zero['frame'],pixels(env));assert final_gate['passed']
            else:actual.append(frame);states.append(state.copy());flags.append(row)
        path=ROOT/f'prefix{prefix}.npz'
        np.savez_compressed(path,controls=controls,observations=np.stack(observations),prefix_actions=logged['actions'][:prefix],
            goal=byte(rgb_tensor(cached['goal'],FactualRecurrent)),actual60=np.stack(actual),states60=np.stack(states),
            successes=np.asarray(flags,bool),initial_object=initial_obj,target_audit_only=target)
        save(path.with_suffix('.json'),dict(passed=True,utc=now(),prefix_native=prefix,case_sha256=case['sha256'],
            source_trace_sha256=reference['trace_sha256'],selected_successful_developmental_case=True,
            bank=str(path),bank_sha256=sha(path),initial_physics_exact=True,initial_rgb_gate=initial_gate,
            all65_integration_restores_exact=True,branch_rgb_gates=gates,zero_after64_interventions_exact=True,
            final_zero_rgb_gate=final_gate,candidates=64,physics_steps=prefix+65*60,
            observed_history_scope='InitialRGBplusactuallyobserved15stepfeedbackonrecordedprefix;neverfutureframes.',
            protected_audit_only=True,training_teacher_proposal_eligible=False))
    finally:close_environment(env)
    return path


@torch.no_grad()
def compare(prefix,seed,step,path):
    cfg=read(PARENT/f'training/joint_anchor-seed{seed}/config.json');device,world,_,_=setup(cfg)
    weights=Path(cfg['output'])/f'model-step{step:07d}/weights.pt'
    assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256']
    world.load_state_dict(torch.load(weights,map_location=device,weights_only=True),strict=True)
    world.eval().requires_grad_(False);before=tensor_sha(world);data=np.load(path)
    observed=torch.from_numpy(data['observations']).permute(0,3,1,2).to(device).float()/255
    history=torch.from_numpy(data['prefix_actions']).to(device);image=observed[:1]
    with torch.autocast('cuda',dtype=torch.bfloat16):hidden=world.parent.observe(world.parent.initial(1),image)
    for t,action in enumerate(history,1):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            hidden=advance(world.parent,hidden,action[None])
            if t%15==0:image=observed[t//15:t//15+1];hidden=world.parent.observe(hidden,image)
    state=hidden.expand(64,-1,-1,-1).clone();actions=torch.from_numpy(data['controls']).to(device)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for t in range(60):state=advance(world.parent,state,actions[:,t])
    goal=torch.from_numpy(data['goal'].copy()).permute(2,0,1)[None].to(device).float()/255
    scores=cost(world,state,(image*2-1).expand(64,-1,-1,-1),goal*2-1,470109).cpu().numpy()
    actual=data['actual60'].astype(float)/255;rgb=((actual-data['goal'].astype(float)/255)**2).mean((1,2,3))
    distances=np.linalg.norm(data['states60'][:,4:7]-data['target_audit_only'],axis=1)
    flags=data['successes'].any(1);chosen=int(scores.argmin());auc=None
    if flags.any() and (~flags).any():
        a,b=scores[flags,None],scores[None,~flags];auc=float(((a<b)+.5*(a==b)).mean())
    assert before==tensor_sha(world)
    save(ROOT/'predictions'/f'prefix{prefix}-seed{seed}-step{step}.json',dict(passed=True,utc=now(),prefix_native=prefix,
        parent_seed=seed,step=step,weights_sha256=sha(weights),bank_sha256=sha(path),state_before=before,state_after=before,
        candidate_paths=64,goal_energy=scores.tolist(),realized_RGB_goal_cost=rgb.tolist(),
        realized_object_goal_distances_audit_only=distances.tolist(),bank_success_flags=flags.tolist(),
        rank_RGB=correlation(scores,rgb),rank_object_goal_distance=correlation(scores,distances),
        success_pairwise_AUC=auc,selected=chosen,selected_success=bool(flags[chosen]),bank_successes=int(flags.sum()),
        selected_object_goal_distance=float(distances[chosen]),best_object_goal_distance=float(distances.min()),
        model_updates=0,protected_audit_only=True,training_teacher_proposal_eligible=False))


def run(prefix):
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.30)
    save(ROOT/f'prefix{prefix}-AUDIT_ONLY.json',dict(utc=now(),protected_evaluation_case=True,
        prohibition='No frames/states/controls/successes/metrics may be used for SGD,teacher,proposals,or candidateperformancescores.'))
    try:
        path=physical(prefix);deadline=time.time()+7200
        for step in [2,500,2000]:
            for seed in [1,2]:
                out=PARENT/f'training/joint_anchor-seed{seed}'
                while not (out/f'AUDIT-{step:07d}.json').exists():assert time.time()<deadline;time.sleep(10)
                compare(prefix,seed,step,path)
        save(ROOT/f'prefix{prefix}-DONE.json',dict(passed=True,utc=now(),frozen_comparisons=6,actual_paths=64,
            model_updates=0,protected_audit_only=True))
    except BaseException:
        save(ROOT/f'prefix{prefix}-ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',type=int,choices=[0,30,60],required=True);a=p.parse_args();run(a.prefix)
