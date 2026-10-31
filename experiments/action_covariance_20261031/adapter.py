"""Fixed train-only Cholesky coordinates around the original diagonal CEM.

No planner implementation is copied. Original/L=I calls native methods directly.
Model cost receives x=L y; native CEM updates y; returned commands also become x.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import torch


def array_sha(a):
    x=np.ascontiguousarray(a)
    return hashlib.sha256(str(x.shape).encode()+str(x.dtype).encode()+x.tobytes()).hexdigest()


class CorrelationMap:
    def __init__(self,factor):
        factor=np.asarray(factor,dtype=np.float64)
        assert factor.ndim==2 and factor.shape[0]==factor.shape[1]and np.isfinite(factor).all()
        assert np.array_equal(factor,np.tril(factor))and np.all(factor.diagonal()>0)
        np.testing.assert_allclose(np.square(factor).sum(1),1,rtol=1e-12,atol=1e-12)
        self.factor=factor.copy();self.dimension=len(factor)
        self.identity=np.array_equal(factor,np.eye(self.dimension));self.cache={}
        self.stats=dict(cost_calls=0,execution_plans=0,solves=0)

    def transform(self,value,device=None):
        assert value.shape[-2]*value.shape[-1]==self.dimension and not value.requires_grad
        if self.identity:return value
        target=value.device if device is None else torch.device(device)
        key=str(target)
        if key not in self.cache:self.cache[key]=torch.as_tensor(self.factor,dtype=torch.float64,device=target)
        # FP64 for this small fixed linear map avoids TF32 dependence; cast once to native cost dtype.
        flat=value.to(device=target,dtype=torch.float64).flatten(-2)
        mapped=(flat@self.cache[key].T).reshape(value.shape).to(dtype=value.dtype)
        assert torch.isfinite(mapped).all()
        return mapped.to(value.device)

    def report(self):
        return dict(dimension=self.dimension,factor_sha256=array_sha(self.factor),identity_exact_bypass=self.identity,
            transform_precision='FP64 fixed factor/multiply, one cast to native dtype',**self.stats)


@contextmanager
def install(mode,factor=None,expected_scaler=None,startup_path=None,expected_model_root=None):
    assert mode in ('original','correlated')
    from stable_worldmodel.policy import WorldModelPolicy
    from stable_worldmodel.planning import ShootingCostEvaluator
    from stable_worldmodel.planning.solver.cem import CEMSolver
    from stable_worldmodel.protocols import Actionable
    old_set=WorldModelPolicy.set_env;old_solve=CEMSolver.solve;old_rollout=ShootingCostEvaluator._rollout
    records=[]

    def set_env(policy,env):
        result=old_set(policy,env)
        assert isinstance(policy.solver,CEMSolver)and isinstance(policy.solver.cost,ShootingCostEvaluator)
        cfg=policy.cfg
        assert (cfg.horizon,cfg.receding_horizon,cfg.action_block,cfg.history_len)==(5,5,5,3)
        assert policy.solver.n_steps==30 and policy.solver.num_samples==300 and policy.solver.topk==30
        assert policy.solver.var_scale==1 and not policy.solver.callbacks
        assert not isinstance(policy.solver.cost,Actionable),'Only native world-model costs, no actor warmstart'
        if expected_scaler is not None:
            scaler=policy.process['action']
            np.testing.assert_array_equal(scaler.mean_,expected_scaler['action_mean'])
            np.testing.assert_array_equal(scaler.scale_,expected_scaler['action_std'])
        dim=policy.solver.horizon*policy.solver.action_dim
        mapping=CorrelationMap(np.eye(dim)if mode=='original'else factor)
        assert mapping.dimension==dim
        policy.solver._action_correlation=mapping;policy.solver.cost._action_correlation=mapping
        mapping.stats['model_class']=type(policy.solver.cost.model).__module__+'.'+type(policy.solver.cost.model).__qualname__
        mapping.stats['model_imports']={str(inspect.getfile(cls)):hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()for cls in type(policy.solver.cost.model).__mro__ if cls.__module__.startswith(('as_jepa_lewm','stable_worldmodel.models'))}
        if expected_model_root is not None:
            assert Path(inspect.getfile(type(policy.solver.cost.model))).is_relative_to(expected_model_root)
        records.append(mapping)
        return result

    def rollout(cost,info,candidates):
        mapping=cost._action_correlation
        if mode=='original'or mapping.identity:return old_rollout(cost,info,candidates)
        effective=mapping.transform(candidates);mapping.stats['cost_calls']+=1
        if mapping.stats['cost_calls']==1:
            mapping.stats['first_bank']=dict(shape=list(candidates.shape),white_sha256=array_sha(candidates.detach().cpu().numpy()),
                normalized_action_sha256=array_sha(effective.detach().cpu().numpy()))
        return old_rollout(cost,info,effective)

    def solve(solver,info_dict,init_action=None):
        mapping=solver._action_correlation
        if mode=='original'or mapping.identity:
            outputs=old_solve(solver,info_dict,init_action=init_action)
        else:
            # Actual benchmark consumes the whole plan each time; no tail exists to transform.
            assert init_action is None or init_action.shape[1]==0,'Partial warmstarts require a separate protocol'
            outputs=old_solve(solver,info_dict,init_action=init_action)
            white=outputs['actions'];outputs['actions']=mapping.transform(white,device=solver.device)
            # Native mean/var/callback coordinates remain white and are explicitly labelled.
            outputs['search_coordinate_system']='white; actions returned in original normalized coordinates'
            mapping.stats['execution_plans']+=len(white);mapping.stats['solves']+=1
        if startup_path is not None and not Path(startup_path).exists():
            assert np.isfinite(outputs['costs']).all()and torch.isfinite(outputs['actions']).all()
            evidence=mapping.report();evidence.update(first_plan_finite=True,first_costs=outputs['costs'],mode=mode)
            if mode=='correlated'and not mapping.identity:
                expected=(white.numpy().reshape(len(white),-1).astype(np.float64)@mapping.factor.T).astype(np.float32).reshape(white.shape)
                error=float(np.max(np.abs(expected-outputs['actions'].numpy())))
                assert error<=2e-6,'GPU/CPU fixed-map disagreement'
                evidence['first_return_map_max_absolute_error']=error
            path=Path(startup_path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(evidence,indent=2)+'\n')
        return outputs

    try:
        WorldModelPolicy.set_env=set_env;CEMSolver.solve=solve;ShootingCostEvaluator._rollout=rollout
        yield records
    finally:
        WorldModelPolicy.set_env=old_set;CEMSolver.solve=old_solve;ShootingCostEvaluator._rollout=old_rollout
