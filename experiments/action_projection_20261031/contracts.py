"""Small CPU native-CEM/policy contracts; no learned checkpoint or real env."""
from __future__ import annotations
import argparse, copy, hashlib, inspect, json, sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from gymnasium.spaces import Box
from gymnasium.vector.utils import batch_space
from adapter import BoxProjection, install

B=Path('/efs/prescottsun/SPT-JEPA')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def scaler_for(task):
    card=json.loads((B/f'data/hardtasks-v1/{task}-metadata/dataset_card.json').read_text())
    s=StandardScaler();s.mean_=np.asarray(card['action_mean']);s.scale_=np.asarray(card['action_std'])
    s.var_=s.scale_**2;s.n_features_in_=len(s.mean_)
    return s

def algebra(s):
    d=len(s.mean_); p=BoxProjection(Box(-1,1,shape=(d,),dtype=np.float32),s)
    bank=np.random.default_rng(1042).normal(0,3,(2,64,5,5*d)).astype(np.float32)
    original=bank.copy(); projected,raw,executed=p.numpy_project(bank)
    # Independent explicit rounding oracle for the installed sklearn operations.
    mean=s.mean_.astype(np.float32).astype(np.float64);std=s.scale_.astype(np.float32).astype(np.float64)
    oracle=(bank.reshape(-1,d).astype(np.float64)*std).astype(np.float32)
    oracle=(oracle.astype(np.float64)+mean).astype(np.float32)
    oracle=np.minimum(np.maximum(oracle,p.low),p.high)
    physical=oracle.copy()
    oracle=(oracle.astype(np.float64)-mean).astype(np.float32)
    oracle=(oracle.astype(np.float64)/std).astype(np.float32).reshape(bank.shape)
    np.testing.assert_array_equal(projected,oracle);np.testing.assert_array_equal(executed,physical)
    np.testing.assert_array_equal(bank,original)
    repeated,_,executed_twice=p.numpy_project(projected)
    tol=float(8*np.finfo(np.float32).eps*max(1,float(np.abs(projected).max())))
    assert np.max(np.abs(repeated-projected))<=tol and np.max(np.abs(executed_twice-executed))<=tol
    returned=p.execution(raw.reshape(-1,d));np.testing.assert_array_equal(returned,executed)
    np.testing.assert_array_equal(s.transform(returned.copy()).reshape(bank.shape),projected)
    rng=torch.get_rng_state().clone(); value=torch.from_numpy(bank)
    np.testing.assert_array_equal(p.candidates(value).numpy(),projected)
    assert torch.equal(rng,torch.get_rng_state()) and np.array_equal(bank,original)
    # General, non-symmetric physical bounds are honored without task branches.
    q=BoxProjection(Box(np.arange(d,dtype=np.float32)-3,np.arange(d,dtype=np.float32)+.7),s)
    _,_,ex=q.numpy_project(bank);assert np.all(ex>=q.low) and np.all(ex<=q.high)
    dead=np.full((1,d),np.nan,dtype=np.float32);assert np.isnan(p.execution(dead)).all()
    try:p.execution(np.full((1,d),np.inf,dtype=np.float32))
    except AssertionError:pass
    else:raise AssertionError('Infinite actions must fail')
    return dict(independent_sklearn_oracle_exact=True,model_and_returned_action_normalization_exact=True,
        input_unchanged=True,no_global_rng_consumption=True,generic_asymmetric_bounds=True,dead_nan_preserved=True,infinite_execution_rejected=True,
        idempotence_max=float(np.max(np.abs(repeated-projected))),physical_reprojection_max=float(np.max(np.abs(executed_twice-executed))),roundoff_tolerance=tol)

def native_run(s, mode):
    import stable_worldmodel as swm
    from stable_worldmodel.planning import ShootingCostEvaluator, GoalMSE
    from stable_worldmodel.planning.solver.cem import CEMSolver
    class Dynamics(torch.nn.Module):
        def __init__(self):super().__init__();self.dummy=torch.nn.Parameter(torch.zeros(()),requires_grad=False);self.actions=[];self.history=[];self.pixels=[]
        def rollout(self,info,actions):
            self.actions.append(actions.clone())
            self.history.append(info.get('action_history',torch.empty(0)).clone())
            self.pixels.append(info['pixels'].clone())
            info['predicted_emb']=actions.sum(-1,keepdim=True).cumsum(2)
            return info
    class Goal(GoalMSE):
        def __init__(self):super().__init__(reduction='sum');self.values=[]
        def forward(self,info):
            y=super().forward(info);self.values.append(y.clone());return y
    class Solver(CEMSolver):
        def solve(self,*args,**kwargs):
            result=super().solve(*args,**kwargs);self.last_plan=result['actions'].clone();return result
    d=len(s.mean_);space=Box(-1,1,shape=(d,),dtype=np.float32)
    env=SimpleNamespace(num_envs=1,single_action_space=space,action_space=batch_space(space,1))
    model=Dynamics();goal=Goal();cost=ShootingCostEvaluator(model,goal,encode_goal=lambda m,i:i['goal'][:,0])
    solver=Solver(cost,num_samples=64,n_steps=2,topk=8,seed=42,device='cpu')
    policy=swm.policy.WorldModelPolicy(solver=solver,config=swm.PlanConfig(horizon=5,receding_horizon=5,action_block=5,history_len=3),process={'action':s})
    methods=(swm.policy.WorldModelPolicy.set_env,swm.policy.WorldModelPolicy.get_action,ShootingCostEvaluator._rollout)
    global_rng=torch.get_rng_state().clone();commands=[];expectations=[]
    with (install(mode) if mode else nullcontext([])) as projections:
        policy.set_env(env)
        # Deliberately out-of-range warm start makes returned-command clipping
        # and next-replan history observable, rather than vacuous bound checks.
        policy._next_init=torch.full((1,5,5*d),6.,dtype=torch.float32)
        previous=np.zeros((1,1,d),dtype=np.float32)
        for t in range(31):
            info=dict(pixels=np.full((1,1,3),t,dtype=np.float32),goal=np.full((1,1,1),10,dtype=np.float32),action=previous)
            command=policy.get_action(info)
            chosen=solver.last_plan.numpy().reshape(1,25,d)[:,t%25]
            raw=s.inverse_transform(chosen.copy())
            expected=np.clip(raw,space.low,space.high) if mode=='bounded' else raw
            np.testing.assert_array_equal(command,expected)
            if mode=='bounded':
                z=projections[0].numpy_project(chosen)[0]
                np.testing.assert_array_equal(s.transform(command.copy()),z)
            commands.append(command.copy());expectations.append(expected);previous=command[:,None]
        stats=[x.report()for x in projections]
        assert model.history[0].numel()==0 and model.pixels[0].shape[2]==1
        expected_history=s.transform(np.stack(commands[15:25]).reshape(10,d)).reshape(1,1,2,5*d)
        np.testing.assert_array_equal(model.history[2].numpy(),np.repeat(expected_history,64,axis=1))
        np.testing.assert_array_equal(model.pixels[2][0,0,:,0].numpy(),np.array([15,20,25],np.float32))
    assert methods==(swm.policy.WorldModelPolicy.set_env,swm.policy.WorldModelPolicy.get_action,ShootingCostEvaluator._rollout)
    assert torch.equal(global_rng,torch.get_rng_state())
    return dict(commands=np.stack(commands),model_actions=model.actions,costs=goal.values,private_rng=solver.torch_gen.get_state(),stats=stats)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-record',default=str(B/'runs/local_basis2000_20261031/source-record.json'))
    parser.add_argument('--output',default=str(B/'reports/action-projection-20261031/cpu-contracts.json'));args=parser.parse_args()
    source=json.loads(Path(args.source_record).read_text());sys.path[:0]=source['pythonpath'].split(':')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');assert not torch.cuda.is_initialized()
    rows={}
    for task in ['tworoom','cube']:
        s=scaler_for(task);checks=algebra(s);plain=native_run(s,None);identity=native_run(s,'original');bounded=native_run(s,'bounded')
        np.testing.assert_array_equal(plain['commands'],identity['commands'])
        for key in ['model_actions','costs']:
            assert len(plain[key])==len(identity[key])
            assert all(torch.equal(a,b)for a,b in zip(plain[key],identity[key]))
        assert torch.equal(plain['private_rng'],identity['private_rng']) and torch.equal(plain['private_rng'],bounded['private_rng'])
        assert np.all(np.abs(bounded['commands'])<=1) and bounded['stats'][0]['returned_command_violations']==0
        assert bounded['stats'][0]['execution_clipped_coordinates']>0
        # Same first unbounded bank is generated in original and bounded runs;
        # model sees precisely the Box-projected version of it in bounded mode.
        proj=BoxProjection(Box(-1,1,shape=(len(s.mean_),),dtype=np.float32),s)
        np.testing.assert_array_equal(bounded['model_actions'][0].numpy(),proj.numpy_project(plain['model_actions'][0].numpy())[0])
        rows[task]=dict(**checks,native_original_policy_cost_candidates_rng_bitexact=True,
            bounded_native_returned_commands_match_model_projection=True,first_cem_proposal_unchanged=True,
            solver_random_draw_count_unchanged=True,policy_calls_31_two_replans=True,out_of_range_warmstart6_exercises_actual_execution_clipping=True,
            second_replan_history_exact_actual_returned_last10_commands=True,observation_history_frames_15_20_25_unchanged=True,stats=bounded['stats'][0])
    from stable_worldmodel.policy import WorldModelPolicy
    from stable_worldmodel.planning import ShootingCostEvaluator,GoalMSE
    from stable_worldmodel.planning.solver.cem import CEMSolver
    from stable_worldmodel.buffer import HistoryBuffer
    import sklearn
    native_files={inspect.getfile(c):sha(inspect.getfile(c))for c in [WorldModelPolicy,ShootingCostEvaluator,GoalMSE,CEMSolver,HistoryBuffer]}
    result=dict(passed=True,cpu_only=True,learned_models_loaded=0,real_environments_created=0,new_environment_rollouts=0,
        source=source,rows=rows,native_files=native_files,sklearn_version=sklearn.__version__,helper_files={str(Path(__file__).with_name(n)):sha(Path(__file__).with_name(n))for n in ['adapter.py','contracts.py']},
        scope='CPU native CEM/GoalMSE/WorldModelPolicy with a toy deterministic dynamics module, real train scalers, history3/block5 and31policy calls. Identity mode is exact; bounded mode jointly changes model candidate input and physical returned command. No trained model or task result asserted.')
    assert not torch.cuda.is_initialized();out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
