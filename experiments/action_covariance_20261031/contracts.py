"""Synthetic CPU contracts against the actual native CEM, no learned models/envs."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from gymnasium.spaces import Box
from stable_worldmodel.planning import ShootingCostEvaluator
from stable_worldmodel.planning.solver.cem import CEMSolver
from stable_worldmodel.policy import WorldModelPolicy,PlanConfig
from adapter import CorrelationMap,install

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.p=torch.nn.Parameter(torch.tensor(0.));self.seen=[];self.histories=[]
    def rollout(self,info,actions):
        self.seen.append(actions.clone());self.histories.append(None if 'action_history'not in info else info['action_history'].clone());return dict(info,action_candidates=actions)
def objective(info):return ((info['action_candidates']-.17)**2).sum((-1,-2))
def make(dim):
    model=Toy();cost=ShootingCostEvaluator(model,objective,encode_goal=None)
    solver=CEMSolver(cost,device='cpu',seed=42)
    policy=WorldModelPolicy(solver,PlanConfig(horizon=5,receding_horizon=5,action_block=5,history_len=3))
    env=SimpleNamespace(num_envs=1,action_space=Box(-1,1,shape=(1,dim),dtype=np.float32),single_action_space=Box(-1,1,shape=(dim,),dtype=np.float32))
    policy.set_env(env);return model,solver

def run(dim,mode=None,L=None):
    def body():
        model,solver=make(dim);rng_before=torch.random.get_rng_state().clone()
        output=solver.solve({'dummy':torch.zeros(1,1)})
        assert torch.equal(rng_before,torch.random.get_rng_state())
        return model.seen,output,solver.torch_gen.get_state().clone()
    if mode is None:return body()
    with install(mode,L):return body()

def exact(a,b):
    for x,y in zip(a[0],b[0]):assert torch.equal(x,y)
    assert len(a[0])==len(b[0])==30
    assert torch.equal(a[2],b[2])
    for key in ['actions','mean','var']:
        x,y=a[1][key],b[1][key]
        if isinstance(x,list):
            for xx,yy in zip(x,y):assert torch.equal(xx,yy)
        else:assert torch.equal(x,y)
    assert a[1]['costs']==b[1]['costs']

def policy_contract(dim,L,calibration):
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler();scaler.mean_=np.asarray(calibration['action_mean']);scaler.scale_=np.asarray(calibration['action_std']);scaler.var_=scaler.scale_**2;scaler.n_features_in_=dim
    with install('correlated',L,expected_scaler=calibration):
        model=Toy();cost=ShootingCostEvaluator(model,objective,encode_goal=None);solver=CEMSolver(cost,device='cpu',seed=42)
        policy=WorldModelPolicy(solver,PlanConfig(horizon=5,receding_horizon=5,action_block=5,history_len=3),process={'action':scaler})
        env=SimpleNamespace(num_envs=1,action_space=Box(-1,1,shape=(1,dim),dtype=np.float32),single_action_space=Box(-1,1,shape=(dim,),dtype=np.float32))
        policy.set_env(env);native=solver.solve;plans=[]
        def capture(info_dict,init_action=None):
            result=native(info_dict,init_action=init_action);plans.append(result['actions'].clone());return result
        solver.solve=capture
        last=np.zeros((1,dim),dtype=np.float32);commands=[]
        for step in range(50):
            last=policy.get_action({'pixels':np.full((1,1),step,dtype=np.float32),'action':last.copy()});commands.append(last.copy())
        assert len(plans)==2 and len(model.seen)==60 and policy._next_init is None
        expected_normalized=torch.cat(plans,dim=1).numpy().reshape(50,dim)
        expected_raw=scaler.inverse_transform(expected_normalized.copy())
        actual=np.concatenate(commands);np.testing.assert_array_equal(actual,expected_raw)
        # At raw25, history represents executed actions15..24. It undergoes the native scaler roundtrip.
        expected_history=scaler.transform(actual[15:25].copy()).reshape(1,1,2,5*dim)
        assert model.histories[30]is not None
        np.testing.assert_array_equal(model.histories[30][:,:1].numpy(),expected_history)
        assert solver._action_correlation.stats['cost_calls']==60 and solver._action_correlation.stats['execution_plans']==2
    return 'actual scaler,50 returned raw commands,2 replans,second-plan executed history contain mapped original coordinates'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--calibration',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();tests=[]
    for task,dim in [('tworoom',2),('cube',5)]:
        with np.load(Path(a.calibration)/task/'matrix.npz')as f:L=f['cholesky'];R=f['correlation']
        base=run(dim);exact(base,run(dim,'original'));exact(base,run(dim,'correlated',np.eye(dim*25)))
        tests.append(task+': native/original/L=I exact all30 banks, costs, moments, returned actions and solver/global RNG')
        mapping=CorrelationMap(L);x=torch.arange(2*dim*25,dtype=torch.float32).reshape(2,5,dim*5)/117
        expected=(x.numpy().reshape(2,-1).astype(np.float64)@L.T).astype(np.float32).reshape(x.shape)
        assert torch.equal(mapping.transform(x),torch.from_numpy(expected))
        np.testing.assert_allclose(L@L.T,R,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(np.square(L).sum(1),1,rtol=1e-12,atol=1e-12)
        test=run(dim,'correlated',L)
        # First draw's white coordinates are exactly the original first bank. The later ones intentionally diverge.
        assert torch.equal(test[0][0],mapping.transform(base[0][0]))
        assert torch.equal(test[0][0][:,0],torch.zeros_like(test[0][0][:,0]))
        assert torch.equal(test[1]['actions'],mapping.transform(test[1]['mean'][0]))
        assert torch.equal(test[2],base[2])and not torch.equal(test[0][0],base[0][0])
        tests.append(task+': Cholesky covariance/unit marginal, same first random innovations, zero forced mean, mapped output, equal RNG consumption')
        with install('correlated',L):
            _,solver=make(dim)
            try:solver.solve({'dummy':torch.zeros(1,1)},init_action=torch.zeros(1,1,5*dim))
            except AssertionError as e:assert 'Partial warmstarts'in str(e)
            else:raise AssertionError('Unsupported warmstart was accepted')
        tests.append(task+': unused partial warmstart rejected before solver execution')
        calibration=json.loads((Path(a.calibration)/task/'calibration.json').read_text())
        tests.append(task+': '+policy_contract(dim,L,calibration))
    before=(WorldModelPolicy.set_env,CEMSolver.solve,ShootingCostEvaluator._rollout)
    try:
        with install('original'):raise RuntimeError('intentional')
    except RuntimeError:pass
    assert before==(WorldModelPolicy.set_env,CEMSolver.solve,ShootingCostEvaluator._rollout)
    tests.append('all hooks restored on deliberate exception')
    result=dict(passed=True,tests=tests,scope='CPU synthetic quadratic costs only; no learned encoder, rollout, RGB, simulator, training or planning evaluation',
        helper_files={str(Path(__file__).with_name(n)):hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()for n in ['adapter.py','contracts.py']})
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
