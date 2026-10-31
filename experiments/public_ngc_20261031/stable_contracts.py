"""Stable attachment plus all existing native-controller CPU contracts."""
import argparse
from pathlib import Path
import unittest
import numpy as np
import torch
import objective
import contracts
from common import write, sha
from stable_cost import StableNGCCost

objective.NGCCost = contracts.NGCCost = StableNGCCost


class StableTests(unittest.TestCase):
    def test_extreme_common_offset_independent_direct_formula(self):
        z = np.ones((40,3), np.float32)*1000
        z[:,0] += np.arange(40, dtype=np.float32)*.001
        ref = np.linalg.norm(z.astype(np.float64)[:,None]-z.astype(np.float64)[None], axis=-1)
        obj = StableNGCCost(z, ref, 15, 4)
        ids, distances, radius = obj.attach(torch.from_numpy(z))
        order = np.argsort(ref, axis=1, kind='stable')
        np.testing.assert_array_equal(ids.numpy(), order[:,:4])
        expected = np.take_along_axis(ref, order, axis=1)
        np.testing.assert_allclose(distances.numpy(), expected[:,:4], rtol=1e-6, atol=0)
        np.testing.assert_allclose(radius.numpy(), expected[:,14], rtol=1e-6, atol=0)
        self.assertTrue(torch.all(radius>0))
        self.assertTrue(torch.equal(obj({'predicted_emb':torch.from_numpy(z)[:,None,None],
                                        'goal_emb':torch.from_numpy(z)[:,None]}),torch.zeros(40,1)))
    def test_true_duplicates_remain_rejected(self):
        z=np.arange(120,dtype=np.float32).reshape(40,3);z[:16]=z[0]
        distance=np.linalg.norm(z.astype(np.float64)[:,None]-z.astype(np.float64)[None],axis=-1)
        obj=StableNGCCost(z,distance,15,4)
        with self.assertRaises(AssertionError):obj.attach(torch.from_numpy(z[:1]))
    def test_only_attach_overridden(self):
        from vendor.cost import NGCCost
        self.assertIs(StableNGCCost.forward,NGCCost.forward)
        self.assertIs(StableNGCCost.attach_goal,NGCCost.attach_goal)
        self.assertIs(StableNGCCost.__init__,NGCCost.__init__)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(1)
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(t)for t in [contracts.Tests,StableTests]])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    files=['stable_cost.py','stable_contracts.py','common.py','objective.py','contracts.py','vendor/cost.py','vendor/graph.py']
    write(a.output,dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        files={f:sha(Path(__file__).parent/f)for f in files},torch_version=torch.__version__,model_calls=0,gpu_calls=0,
        scope='All9 original contracts run with stable attachment, plus3 independent stability/inheritance contracts. L2 returns original native tensor and exact CEM outputs.'))
    raise SystemExit(0 if result.wasSuccessful()else 1)
