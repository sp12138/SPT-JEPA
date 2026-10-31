"""CPU contracts of the exact two frozen adapters composed on native CEM."""
import argparse
from contextlib import ExitStack
from types import SimpleNamespace
import tempfile
import numpy as np
import torch
from gymnasium.spaces import Box
from sklearn.preprocessing import StandardScaler
from run import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1)
    ngc,corrsource=read(NGC),read(CORR);source_check(ngc);source_check(corrsource);corr,obj=dependencies(ngc,corrsource)
    from stable_worldmodel.planning import GoalMSE,ShootingCostEvaluator
    from stable_worldmodel.planning.solver.cem import CEMSolver
    from stable_worldmodel.policy import WorldModelPolicy,PlanConfig
    from vendor.graph import build_graph
    nodes,distances,_=build_graph(np.random.default_rng(19).normal(size=(40,3)).astype(np.float32),15,'mutual')
    class Toy(torch.nn.Module):
        def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor(0.));self.seen=[];self.hist=[]
        def rollout(self,info,actions):
            self.seen.append(actions.clone());self.hist.append(None if 'action_history'not in info else info['action_history'].clone())
            z=torch.stack([actions.sum(-1),actions[...,0],actions[...,-1]],dim=-1).cumsum(-2)
            return dict(info,predicted_emb=z+self.p)
    def make(dim,cal=None):
        model=Toy();cost=ShootingCostEvaluator(model,GoalMSE(),encode_goal=lambda m,i:torch.zeros(len(i['pixels']),1,3))
        solver=CEMSolver(cost,device='cpu',seed=42);process={}
        if cal:
            scaler=StandardScaler();scaler.mean_=np.asarray(cal['action_mean']);scaler.scale_=np.asarray(cal['action_std']);scaler.var_=scaler.scale_**2;scaler.n_features_in_=dim;process={'action':scaler}
        policy=WorldModelPolicy(solver,PlanConfig(horizon=5,receding_horizon=5,action_block=5,history_len=3),process=process)
        env=SimpleNamespace(num_envs=1,action_space=Box(-1,1,shape=(1,dim),dtype=np.float32),single_action_space=Box(-1,1,shape=(dim,),dtype=np.float32))
        policy.set_env(env);return model,solver,policy
    def solve(dim,L=None,metric=None):
        with ExitStack()as stack:
            if L is not None:stack.enter_context(corr.install('correlated',L))
            recorder=None
            if metric:
                directory=stack.enter_context(tempfile.TemporaryDirectory());recorder=obj.Recorder(directory)
                stack.enter_context(obj.install(nodes,distances,metric,recorder))
            model,solver,_=make(dim);before=torch.random.get_rng_state().clone()
            result=solver.solve(dict(pixels=torch.zeros(1,1,3)));assert torch.equal(before,torch.random.get_rng_state())
            if recorder:
                assert len(recorder.rows)==30
                for seen,row in zip(model.seen,recorder.rows):assert obj.digest_tensor(seen)==row['actions_sha256']
                recorder.finish()
            return model.seen,result,solver.torch_gen.get_state().clone()
    def exact(a,b):
        assert len(a[0])==len(b[0])==30 and all(torch.equal(x,y)for x,y in zip(a[0],b[0]))and torch.equal(a[2],b[2])
        for key in ['actions','mean','var']:
            x,y=a[1][key],b[1][key]
            if isinstance(x,list):assert len(x)==len(y)and all(torch.equal(xx,yy)for xx,yy in zip(x,y))
            else:assert torch.equal(x,y)
        assert a[1]['costs']==b[1]['costs']
    tests=[]
    for task,dim in [('tworoom',2),('cube',5)]:
        path=B/'runs/action_covariance_evaluation_20261031/calibration'/task/'calibration.json';cal=read(path)
        assert sha(path.with_name('matrix.npz'))==cal['matrix_sha256']
        with np.load(path.with_name('matrix.npz'))as f:L=f['cholesky'];R=f['correlation']
        np.testing.assert_allclose(L@L.T,R,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(np.square(L).sum(1),1,rtol=1e-12,atol=1e-12)
        white=solve(dim);exact(white,solve(dim,np.eye(dim*25),'l2'));tests.append(task+': composed identity L2 exact native banks/costs/moments/actions/RNG')
        oldcorr=solve(dim,L);combined=solve(dim,L,'l2');exact(oldcorr,combined);tests.append(task+': composed L2 exact frozen f4 alone, recorder sees mapped model actions')
        nonlinear=solve(dim,L,'ngc');mapping=corr.CorrelationMap(L)
        assert torch.equal(combined[0][0],nonlinear[0][0])and torch.equal(combined[0][0],mapping.transform(white[0][0]))
        assert torch.equal(combined[2],nonlinear[2])and torch.equal(combined[2],white[2])
        tests.append(task+': common initial correlated bank across metrics, unit coordinate marginals and unchanged solver RNG consumption')
        for metric in ['l2','ngc']:
            with corr.install('correlated',L,expected_scaler=cal):
                with obj.install(nodes,distances,metric):
                    model,solver,policy=make(dim,cal);native=solver.solve;plans=[]
                    def capture(info_dict,init_action=None):
                        result=native(info_dict,init_action=init_action);plans.append(result['actions'].clone());return result
                    solver.solve=capture;last=np.zeros((1,dim),np.float32);commands=[]
                    for step in range(50):
                        last=policy.get_action(dict(pixels=np.full((1,3),step,dtype=np.float32),action=last.copy()));commands.append(last.copy())
                    assert len(plans)==2 and len(model.seen)==60 and policy._next_init is None
                    normal=torch.cat(plans,dim=1).numpy().reshape(50,dim);scaler=policy.process['action'];raw=np.concatenate(commands)
                    np.testing.assert_array_equal(raw,scaler.inverse_transform(normal.copy()))
                    history=scaler.transform(raw[15:25].copy()).reshape(1,1,2,5*dim)
                    np.testing.assert_array_equal(model.hist[30][:,:1].numpy(),history)
                    try:native(dict(pixels=torch.zeros(1,1,3)),init_action=torch.zeros(1,1,5*dim))
                    except AssertionError as e:assert 'Partial warmstarts'in str(e)
                    else:raise AssertionError('Unsupported warmstart accepted')
            tests.append(task+': '+metric+' actual scaler/50 commands/two replans/mapped history/partial-warmstart rejection')
    before=(WorldModelPolicy.set_env,CEMSolver.solve,ShootingCostEvaluator.__init__,ShootingCostEvaluator._rollout)
    try:
        with corr.install('correlated',np.eye(50)):
            with obj.install(nodes,distances,'ngc'):raise RuntimeError('contract')
    except RuntimeError:pass
    assert before==(WorldModelPolicy.set_env,CEMSolver.solve,ShootingCostEvaluator.__init__,ShootingCostEvaluator._rollout)
    tests.append('both nested hooks restore on exception')
    write(a.output,dict(passed=True,tests=tests,helper_files={f:sha(Path(__file__).with_name(f))for f in FILES},
        ngc_source=ngc,correlation_source=corrsource,trained_model_calls=0,gpu_calls=0,simulation_calls=0,
        scope='Synthetic CPU action-conditioned toy only, exact pinned native solver/policy and both frozen adapters; no graph encoding, world-model checkpoint or environment rollout.'))
    print(json.dumps(dict(passed=True,tests=len(tests))))


if __name__=='__main__':main()
