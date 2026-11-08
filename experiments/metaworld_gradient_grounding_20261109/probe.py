"""AUDIT ONLY: are goal-cost gradients grounded in actual interventions?

Two frozen factual parents, eight shared physical branch points. Own Gaussian
starts are chosen by model predictions only. No oracle-selected control, expert
path, action label, parameter update, or policy-success claim is involved.
"""
import argparse
import hashlib
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from scipy.optimize import minimize
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,sha,tensor_sha,now
from experiments.metaworld_transition_only_20261109.evaluate import rgb_tensor
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment
from experiments.metaworld_horizon_match_20261109.model import build
from experiments.metaworld_horizon_match_20261109.bank import capture,mjstate
from experiments.metaworld_world_vs_planner_20261109.oracle import restore_exact

ROOT=BASE/'runs/metaworld_gradient_grounding_20261109'
IEEE=False


class EvaluationBudget(Exception):pass


def forecast(model,hidden,image,commands):
    state=hidden.expand(len(commands),-1,-1,-1).clone()
    future=image.expand(len(commands),-1,-1,-1)
    for t in range(60):state,future=model.predict(state,future,commands[:,t//5])
    return future


def refine(model,hidden,image,goal,generator):
    with torch.no_grad():
        candidates=torch.randn(64,12,4,device='cuda',generator=generator).clamp(-1,1)
        candidates[0]=0
        initial_scores=(forecast(model,hidden,image,candidates)-goal).square().mean((1,2,3))
        starts=candidates[initial_scores.topk(4,largest=False).indices].detach().clone()
        baseline=(forecast(model,hidden,image,starts)-goal).square().mean((1,2,3)).cpu().numpy()
    normalizer=max(float(baseline.mean()),1e-12)
    best=starts.cpu().numpy().copy();best_values=baseline.copy();calls=0;history=[]

    def objective(value):
        nonlocal calls,best,best_values
        if calls>=48:raise EvaluationBudget()
        commands=torch.tensor(value.reshape(4,12,4),device='cuda',dtype=torch.float32,requires_grad=True)
        future=forecast(model,hidden,image,commands)
        costs=(future-goal).square().mean((1,2,3));loss=costs.mean()/normalizer
        gradient=torch.autograd.grad(loss,commands)[0]
        assert torch.isfinite(gradient).all() and torch.isfinite(costs).all()
        actual=costs.detach().cpu().numpy();improved=actual<best_values
        best[improved]=value.reshape(4,12,4)[improved];best_values=np.minimum(best_values,actual)
        calls+=1;history.append(dict(call=calls,predicted_goal_mse=actual.tolist(),best_goal_mse=best_values.tolist()))
        return float(loss.detach()),gradient.detach().cpu().numpy().astype(np.float64).ravel()

    # Directional derivative check of the actual FP32 model at an interior point.
    point=(starts.cpu().numpy().astype(np.float64)*.9).ravel()
    value,gradient=objective(point)
    direction=np.random.default_rng(20261109).standard_normal(point.shape)
    direction/=np.linalg.norm(direction)
    def scalar(x):
        with torch.no_grad():
            controls=torch.tensor(x.reshape(4,12,4),device='cuda',dtype=torch.float32)
            return float((forecast(model,hidden,image,controls)-goal).square().mean()/normalizer)
    epsilon=.002
    finite_difference=(scalar(point+epsilon*direction)-scalar(point-epsilon*direction))/(2*epsilon)
    analytic=float(gradient@direction)
    error=abs(finite_difference-analytic)/max(abs(analytic),abs(finite_difference),1e-5)
    # Clamp kinks can make centered finite differences unstable. Retain the
    # actual readout rather than pretending every point is differentiable.
    derivative=dict(analytic=analytic,finite_difference=finite_difference,relative_error=error,
        smooth_agreement=error<.1,precision='FP32')
    # Calibration point above is audit-only, not an optimization proposal.
    best=starts.cpu().numpy().copy();best_values=baseline.copy();calls=0;history=[]
    try:
        result=minimize(objective,starts.cpu().numpy().astype(np.float64).ravel(),jac=True,method='L-BFGS-B',
            bounds=[(-1,1)]*(4*12*4),options=dict(maxiter=32,maxfun=48,ftol=1e-7,gtol=1e-5))
        status=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit))
    except EvaluationBudget:status=dict(success=False,message='Declared48function-evaluationbudget',iterations=None)
    assert np.all(best_values<=baseline+1e-7)
    return starts.cpu().numpy(),best,dict(baseline_predicted=baseline.tolist(),refined_predicted=best_values.tolist(),
        function_calls=calls,trajectory_forward_evaluations=64+4*calls,
        backward_passes=calls,objective_normalizer=normalizer,optimizer=status,derivative_audit=derivative,history=history)


def realize(env,snapshot,controls,goal,model):
    restore_exact(env,snapshot);reference=mjstate(env).copy();observation=None;flags=[]
    for command in np.repeat(controls,5,axis=0):
        observation,_,terminated,truncated,info=env.step(command.astype(np.float32))
        assert not terminated and not truncated
        flags.append(bool(info['success']))
    actual=rgb_tensor(pixels(env),model)
    return dict(rgb_goal_mse=float((actual-goal).square().mean()),success=bool(any(flags)),
        observation=observation.copy(),integration=mjstate(env).copy(),flags=np.asarray(flags),
        initial=reference)


def run(seed):
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.30)
    if IEEE:
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    save(out/'AUDIT_ONLY.json',dict(candidate_performance=False,training_data_eligible=False,model_updates=0,
        policy_episodes=0,own_random_actions=True,no_oracle_or_expert_proposals=True,ieee_fp32=IEEE,utc=now()))
    cfg=read(BASE/f'runs/metaworld_horizon_match_20261109/training/h60-seed{seed}/config.json')
    weights=Path(cfg['output'])/'model-step0012000/weights.pt'
    assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256']
    model=build(cfg).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True));before=tensor_sha(model)
    rows=[];started=time.monotonic();total_calls=0
    try:
        for case in read(BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json')['rows']:
            assert time.monotonic()-started<5400
            assert sha(case['path'])==case['sha256'];cached=np.load(case['path'])
            env,state,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
            try:
                assert np.array_equal(state,cached['initial_state'])
                assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
                assert rgb_replay_check(cached['initial'],pixels(env))['passed']
                with torch.no_grad():
                    image=rgb_tensor(cached['initial'],model);goal=rgb_tensor(cached['goal'],model)
                    hidden=model.observe(model.initial(1),image)
                    for _ in range(7):
                        control=torch.zeros(1,4,device='cuda');env.step(np.zeros(4,np.float32))
                        hidden,_=model.predict(hidden,image,control)
                        image=rgb_tensor(pixels(env),model);hidden=model.observe(hidden,image)
                snapshot=capture(env);generator=torch.Generator(device='cuda').manual_seed(771009+case['seed'])
                starts,refined,detail=refine(model,hidden,image,goal,generator)
                results=[]
                for kind,controls in [('start',starts),('refined',refined)]:
                    for index,commands in enumerate(controls):
                        actual=realize(env,snapshot,commands,goal,model)
                        replay=realize(env,snapshot,commands,goal,model)
                        assert all(np.array_equal(actual[k],replay[k]) for k in ['initial','integration','observation','flags'])
                        assert abs(actual['rgb_goal_mse']-replay['rgb_goal_mse'])<1e-6
                        results.append(dict(kind=kind,index=index,rgb_goal_mse=actual['rgb_goal_mse'],
                            any_success=actual['success'],object_goal_distance_audit_only=float(np.linalg.norm(actual['observation'][4:7]-actual['observation'][-3:]))))
                trace=out/(case['task']+'.npz')
                np.savez_compressed(trace,start_controls=starts,refined_controls=refined)
                total_calls+=7+16*60
                row=dict(task=case['task'],case_seed=case['seed'],case_sha256=case['sha256'],details=detail,
                    results=results,trace=str(trace),trace_sha256=sha(trace),all_physics_replays_exact=True,
                    candidate_performance=False,training_data_eligible=False)
                rows.append(row);save(out/'progress.json',dict(utc=now(),rows=rows,model_updates=0))
                print(case['task'],detail['function_calls'],[(r['kind'],r['index'],r['rgb_goal_mse']) for r in results],flush=True)
            finally:close_environment(env)
        assert before==tensor_sha(model)
        save(out/'DONE.json',dict(passed=True,utc=now(),rows=rows,physics_steps=total_calls,
            model_state_before=before,model_state_after=before,weights_sha256=sha(weights),
            model_updates=0,policy_episodes=0,candidate_performance=False,training_data_eligible=False,
            ieee_fp32=IEEE,
            scope='Eight previously-inspected physical states after7zero controls,4 own model-selected starts and4 refined paths/model/state. '
                'Every60step path actually simulated twice with exact full-state/RNG restoration. '
                'Two parents repeat same8 cases;not formal500step policy performance. FP32 differentiable inference differs from previousBF16CEM,not equalFLOPs. '
                'Protected audit controls/frames/states cannot train or seed a deployed proposal library.'))
    except BaseException:
        save(out/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[1,2],required=True);p.add_argument('--ieee',action='store_true');a=p.parse_args()
    IEEE=a.ieee
    if IEEE:ROOT=ROOT/'ieee-v2'
    run(a.seed)
