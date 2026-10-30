"""CPU contracts for covariance calibration and native objective-only intervention."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from stable_worldmodel.planning import GoalMSE,ShootingCostEvaluator
from stable_worldmodel.planning.solver.cem import CEMSolver
from experiments.carrier_progress_20261031.metric_core import (
    MetricGoal,MetricRecorder,fit_metrics,fit_whitener,install_objective)


class ToyDynamics(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.eye(3));self.calls=[]
    def rollout(self,info,actions):
        self.calls.append((info["pixels"].clone(),actions.clone()))
        result=dict(info)
        result["predicted_emb"]=info["pixels"]+actions.cumsum(-2)@self.weight
        return result


def matrices():
    return dict(observation_whitening=np.diag([1.,2.,3.]),increment_whitening=np.diag([3.,1.,2.]))


class MetricContracts(unittest.TestCase):
    def test_fit_uses_observations_and_only_within_clip_increments(self):
        rng=np.random.default_rng(17);clips=rng.normal(size=(128,8,3))
        arrays,report=fit_metrics(clips)
        expected,_=fit_whitener(np.diff(clips,axis=1).reshape(-1,3))
        np.testing.assert_array_equal(expected["covariance"],arrays["increment_covariance"])
        self.assertEqual(report["observation"]["samples"],1024)
        self.assertEqual(report["increment"]["samples"],896)
        for name in ["observation","increment"]:
            w=arrays[name+"_whitening"];c=arrays[name+"_covariance"]
            np.testing.assert_allclose(w.T@c@w,np.eye(3),atol=1e-12)
        repeated,_=fit_metrics(clips)
        for key in arrays:np.testing.assert_array_equal(arrays[key],repeated[key])

    def test_rank_deficiency_is_regularized_without_test_tuning(self):
        x=np.arange(128,dtype=float)[:,None]*np.array([[1.,2.,0.]])
        arrays,report=fit_whitener(x)
        self.assertGreater(report["shrinkage"],0)
        self.assertTrue(np.isfinite(arrays["whitening"]).all())
        with self.assertRaises(ValueError):fit_whitener(np.ones((128,3)))

    def test_identity_is_exact_native_and_readonly(self):
        torch.manual_seed(17)
        info=dict(predicted_emb=torch.randn(2,7,5,3),goal_emb=torch.randn(2,2,3))
        before={k:v.clone()for k,v in info.items()};rng=torch.get_rng_state().clone()
        for reduction in ["sum","mean"]:
            native=GoalMSE(reduction=reduction);adapted=MetricGoal(native,matrices(),"identity")
            self.assertTrue(torch.equal(native(info),adapted(info)))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key in info:self.assertTrue(torch.equal(info[key],before[key]))

    def test_whitened_cost_equals_explicit_quadratic_and_order(self):
        pred=torch.tensor([[[[2.,0.,0.]],[[0.,1.,0.]],[[0.,0.,.5]]]])
        info=dict(predicted_emb=pred,goal_emb=torch.zeros(1,1,3))
        for name in ["observation","increment"]:
            value=MetricGoal(GoalMSE(),matrices(),name)(info)
            w=torch.tensor(matrices()[name+"_whitening"],dtype=torch.float32)
            delta=pred[:,:,0]
            explicit=torch.einsum("bsd,de,bse->bs",delta,w@w.T,delta)
            self.assertTrue(torch.equal(value,explicit))
            self.assertTrue(torch.equal(value.argsort(),explicit.argsort()))

    def test_native_rollout_inputs_and_call_budget_unchanged(self):
        actions=torch.arange(24,dtype=torch.float32).reshape(1,2,4,3)/10
        info=dict(pixels=torch.ones(1,2,4,3),goal_emb=torch.zeros(1,1,3))
        native_model=ToyDynamics();native=ShootingCostEvaluator(native_model,GoalMSE(),encode_goal=None)
        expected=native.get_cost(copy.deepcopy(info),actions)
        for selected in ["identity","observation","increment"]:
            model=ToyDynamics()
            with install_objective(matrices(),selected):
                evaluator=ShootingCostEvaluator(model,GoalMSE(),encode_goal=None)
                actual=evaluator.get_cost(copy.deepcopy(info),actions)
            self.assertEqual(len(model.calls),1)
            for a,b in zip(model.calls[0],native_model.calls[0]):self.assertTrue(torch.equal(a,b))
            if selected=="identity":self.assertTrue(torch.equal(actual,expected))

    def test_identity_native_cem_actions_and_costs_equal(self):
        import gymnasium as gym
        from types import SimpleNamespace
        info=dict(pixels=torch.ones(2,4,3),goal=torch.zeros(2,1,3))
        outputs=[]
        for instrumented in [False,True]:
            model=ToyDynamics()
            objective=MetricGoal(GoalMSE(),matrices(),"identity")if instrumented else GoalMSE()
            cost=ShootingCostEvaluator(model,objective,encode_goal=lambda m,i:i["goal"][:,0])
            solver=CEMSolver(cost,batch_size=1,num_samples=12,n_steps=3,topk=3,device="cpu",seed=1234)
            solver.configure(action_space=gym.spaces.Box(-1,1,shape=(2,3)),n_envs=2,
                             config=SimpleNamespace(action_block=1,horizon=4))
            outputs.append(solver.solve(copy.deepcopy(info)))
        self.assertTrue(torch.equal(outputs[0]["actions"],outputs[1]["actions"]))
        self.assertEqual(outputs[0]["costs"],outputs[1]["costs"])

    def test_cost_archive_contains_identical_candidate_banks(self):
        torch.manual_seed(18)
        with tempfile.TemporaryDirectory()as directory:
            recorder=MetricRecorder(directory)
            info=dict(predicted_emb=torch.randn(1,5,2,3),goal_emb=torch.randn(1,1,3),
                      action_candidates=torch.randn(1,5,2,3))
            objective=MetricGoal(GoalMSE(),matrices(),"identity",recorder)
            expected=objective(info);audit=recorder.finish()
            with np.load(Path(directory)/"metric-costs.npz")as saved:
                np.testing.assert_array_equal(saved["costs"][0],expected.numpy())
                self.assertEqual(saved["costs"].shape,(3,1,5))
            self.assertEqual(audit["manual_identity_max_abs_error"],0)
            self.assertEqual(audit["extra_model_rollouts"],0)


if __name__=="__main__":unittest.main()
