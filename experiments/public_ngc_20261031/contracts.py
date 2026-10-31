"""Bounded CPU numerical/data-seam contracts; no RGB/checkpoint/simulator calls."""
import argparse
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch import nn
from stable_worldmodel.planning import GoalMSE,ShootingCostEvaluator
from stable_worldmodel.planning.solver.cem import CEMSolver
from common import read,write,sha,select_rows
from objective import Goal,install,Recorder
from vendor.graph import build_graph,neighbor_links
from vendor.cost import NGCCost


def graph():
    z=np.random.default_rng(1337).normal(size=(40,3)).astype(np.float32)
    return build_graph(z,15,'mutual')[:2]
class SpyGoal(GoalMSE):
    def forward(self,info):self.last=super().forward(info);return self.last
class Toy(nn.Module):
    def __init__(self):super().__init__();self.p=nn.Parameter(torch.ones(()));self.calls=[]
    def rollout(self,info,actions):
        self.calls.append((info['pixels'].clone(),actions.clone()))
        return dict(info,predicted_emb=info['pixels']+actions.cumsum(-2)*self.p)


class Tests(unittest.TestCase):
    def test_public_files_exact(self):
        audited=read(Path(__file__).with_name('source-audit.json'))
        for f in ['cost.py','graph.py']:
            self.assertEqual(sha(Path(__file__).parent/'vendor'/f),audited['files']['ngc/'+f])
    def test_cohort_same_training_rows_and_bounds(self):
        ids=np.arange(100)+1000;lengths=np.arange(100)%40+36;offsets=np.r_[0,np.cumsum(lengths)[:-1]]
        args=(ids,offsets,lengths,ids[:80],ids[80:])
        a=select_rows(*args,count=50);self.assertEqual(a,select_rows(*args,count=50))
        self.assertEqual(len({x['episode']for x in a}),50)
        for x in a:
            i=x['episode_array_index'];self.assertLess(x['start']+35,lengths[i]);self.assertIn(x['episode'],ids[:80])
            self.assertEqual(x['row'],offsets[i]+x['start']+5*x['frame'])
        with self.assertRaises(AssertionError):select_rows(ids,offsets,lengths,ids,ids[-1:],count=50)
    def test_public_graph_against_independent_floyd(self):
        z=np.random.default_rng(4).normal(size=(40,3));z[20:]+=100
        nodes,actual,bridges=build_graph(z,4,'mutual');self.assertGreater(len(bridges),0)
        dist=np.linalg.norm(nodes[:,None]-nodes[None],axis=-1);np.fill_diagonal(dist,np.inf)
        linked=neighbor_links(dist,4,'mutual')
        for i,j in bridges:linked[i,j]=linked[j,i]=True
        expected=np.where(linked,dist,np.inf);np.fill_diagonal(expected,0)
        for k in range(len(nodes)):expected=np.minimum(expected,expected[:,k,None]+expected[None,k,:])
        np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)
    def test_cost_independent_formula_shapes_zero(self):
        nodes,distances=graph();q=torch.tensor(np.random.default_rng(5).normal(size=(2,7,3)),dtype=torch.float32);g=q[:,0].clone()
        obj=NGCCost(nodes,distances,15,4);info=dict(predicted_emb=q[:,:,None],goal_emb=g[:,None]);value=obj(info)
        z=nodes.astype(np.float32);expected=np.zeros((2,7))
        for b in range(2):
            dg=np.linalg.norm(z-g[b].numpy(),axis=1);jg=np.argsort(dg,kind='stable')
            for s in range(7):
                dq=np.linalg.norm(z-q[b,s].numpy(),axis=1);iq=np.argsort(dq,kind='stable');direct=float(torch.linalg.vector_norm(q[b,s]-g[b]))
                expected[b,s]=direct if direct<=min(dq[iq[14]],dg[jg[14]])else min(float(dq[i])+distances[i,j]+float(dg[j])for i in iq[:4]for j in jg[:4])
        np.testing.assert_allclose(value.numpy(),expected,rtol=2e-6,atol=2e-6);self.assertTrue(torch.equal(value[:,0],torch.zeros(2)))
    def test_radius_and_nonfinite_fail_closed(self):
        z=np.zeros((40,3),dtype=np.float32);obj=NGCCost(z,np.zeros((40,40)),15,4)
        with self.assertRaises(AssertionError):obj.attach(torch.zeros(2,3))
        z[0,0]=np.nan
        with self.assertRaises(AssertionError):build_graph(z,15,'mutual')
        with self.assertRaises(AssertionError):NGCCost(np.ones((40,3)),np.full((40,40),np.inf),15,4)
    def test_native_tensor_same_and_inputs_rng_unchanged(self):
        nodes,distances=graph();torch.manual_seed(91)
        info=dict(predicted_emb=torch.randn(2,7,2,3),goal_emb=torch.randn(2,1,3))
        saved={k:v.clone()for k,v in info.items()};rng=torch.get_rng_state().clone()
        for reduction in ['sum','mean']:
            native=SpyGoal(reduction=reduction);value=Goal(native,nodes,distances,'l2')(info);self.assertIs(value,native.last)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in saved.items():self.assertTrue(torch.equal(v,info[k]))
    def test_original_cem_full_path_exact(self):
        import gymnasium as gym
        nodes,distances=graph();outputs=[];calls=[]
        for adapted in [False,True]:
            model=Toy();obj=Goal(GoalMSE(),nodes,distances,'l2')if adapted else GoalMSE()
            cost=ShootingCostEvaluator(model,obj,encode_goal=lambda m,i:i['goal'][:,0])
            solver=CEMSolver(cost,batch_size=1,num_samples=12,n_steps=3,topk=3,device='cpu',seed=1234)
            solver.configure(action_space=gym.spaces.Box(-1,1,shape=(2,3)),n_envs=2,config=SimpleNamespace(action_block=1,horizon=4))
            outputs.append(solver.solve(dict(pixels=torch.ones(2,4,3),goal=torch.zeros(2,1,3))));calls.append(model.calls)
        self.assertTrue(torch.equal(outputs[0]['actions'],outputs[1]['actions']));self.assertEqual(outputs[0]['costs'],outputs[1]['costs'])
        self.assertEqual(len(calls[0]),len(calls[1]))
        for aa,bb in zip(calls[0],calls[1]):
            for x,y in zip(aa,bb):self.assertTrue(torch.equal(x,y))
    def test_raw_predictor_actions_and_hook_restore(self):
        nodes,distances=graph();original=ShootingCostEvaluator.__init__
        actions=torch.arange(24,dtype=torch.float32).reshape(1,2,4,3)/10
        info=dict(pixels=torch.ones(1,2,4,3),goal_emb=torch.zeros(1,1,3));reference=Toy()
        expected=ShootingCostEvaluator(reference,GoalMSE(),encode_goal=None).get_cost(copy.deepcopy(info),actions)
        for mode in ['l2','ngc']:
            model=Toy()
            with install(nodes,distances,mode):cost=ShootingCostEvaluator(model,GoalMSE(),encode_goal=None)
            value=cost.get_cost(copy.deepcopy(info),actions);self.assertEqual(len(model.calls),1)
            for x,y in zip(model.calls[0],reference.calls[0]):self.assertTrue(torch.equal(x,y))
            if mode=='l2':self.assertTrue(torch.equal(value,expected))
        try:
            with install(nodes,distances,'ngc'):raise RuntimeError('contract')
        except RuntimeError:pass
        self.assertIs(ShootingCostEvaluator.__init__,original)
    def test_native_recorder_seam_and_serialization(self):
        nodes,distances=graph();actions=torch.ones(1,2,4,3)
        info=dict(pixels=torch.ones(1,2,4,3),goal_emb=torch.zeros(1,1,3))
        with tempfile.TemporaryDirectory()as directory:
            recorder=Recorder(directory);model=Toy()
            with install(nodes,distances,'l2',recorder):cost=ShootingCostEvaluator(model,GoalMSE(),encode_goal=None)
            value=cost.get_cost(info,actions);audit=recorder.finish()
            self.assertEqual(audit['calls'],1);self.assertEqual(audit['metric_order'],['l2','ngc'])
            with np.load(Path(directory)/'costs.npz')as f:
                self.assertEqual(f['costs'].shape,(2,1,2));np.testing.assert_array_equal(f['costs'][0],value.detach().numpy())
            self.assertEqual(len(read(Path(directory)/'banks.json')),1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    files=['common.py','objective.py','vendor/cost.py','vendor/graph.py']
    write(a.output,dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        files={f:sha(Path(__file__).parent/f)for f in files},torch_version=torch.__version__,model_calls=0,simulation_calls=0,
        scope='CPU synthetic and native-solver contracts; no world-model checkpoints or RGB data used.'))
    raise SystemExit(0 if result.wasSuccessful()else 1)
