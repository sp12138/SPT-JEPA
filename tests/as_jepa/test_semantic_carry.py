import unittest,torch
from vforge.ops import OpsConfig,initialize_ops
from vforge.models.as_jepa.semantic_carry import SemanticCarry
class SemanticCarryContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2);initialize_ops(OpsConfig(backend="sdpa_math",allow_fallback=False),device=torch.device("cpu"),parallel_state=None)
 def test_transition_and_feedback_shapes(self):
  torch.manual_seed(8);m=SemanticCarry(width=96,renderer_depth=2,window=4).eval()
  z=torch.randn(1,3,256,96);a=torch.randn(1,3,10)
  torch.testing.assert_close(m.advance(z,a),z,rtol=0,atol=0)
  torch.nn.init.normal_(m.transition[-1].weight,std=.01)
  pred=m.advance(z,a);self.assertEqual(pred.shape,z.shape)
  self.assertTrue(all(not p.requires_grad for p in m.causal_semantic_encoder.vit_e.parameters()))
  context=torch.randn(1,3,3,64,64);actions=torch.randn(1,4,10);anchor=torch.randn(1,3,64,64)
  value=m.rollout_frames(context,actions,anchor,horizon=1,steps=2)
  self.assertEqual(tuple(value.shape),(1,1,3,64,64));self.assertTrue(bool(torch.isfinite(value).all()))
if __name__=="__main__":unittest.main()
