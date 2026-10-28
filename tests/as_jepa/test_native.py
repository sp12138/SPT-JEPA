import unittest
import torch
from vforge.ops import OpsConfig,initialize_ops
from vforge.models.as_jepa.model import ASJepaSPT
from vforge.models.as_jepa.spt_baseline import ActionRepWorld

class NativeContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2);initialize_ops(OpsConfig(backend="sdpa_math",allow_fallback=False),device=torch.device("cpu"),parallel_state=None)
 def test_s2_scratch_meta_initialization_and_frame_independence(self):
  with torch.device("meta"):model=ASJepaSPT(width=96,encoder_depth=2,prior_depth=2,renderer_depth=2)
  model.to_empty(device="cpu");model.initialize_weights();model.eval()
  self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.parameters()))
  frames=torch.randn(1,3,3,64,64);changed=frames.clone();changed[:,2]+=20
  torch.testing.assert_close(model.encoded(frames)[:,:2],model.encoded(changed)[:,:2])
  state=torch.randn(1,3,4,16);actions=torch.randn(1,3,10);torch.testing.assert_close(model.prior(state,actions),state,rtol=0,atol=0)
 def test_spt_action_visibility_is_causal(self):
  model=ActionRepWorld(width=96,renderer_depth=2).eval();encoder=model.causal_semantic_encoder
  torch.nn.init.normal_(encoder.action_proj.weight,std=.1)
  source=torch.randn(1,48,4,16,16);actions=torch.randn(1,4,10);encoder._actions=actions
  original=encoder(source);changed=actions.clone();changed[:,2:]+=100;encoder._actions=changed;altered=encoder(source)
  torch.testing.assert_close(original[:,:2],altered[:,:2]);encoder._actions=None
 def test_anchored_action_baseline_initializes_anchor_and_action_projections(self):
  with torch.device("meta"):model=ActionRepWorld(anchored=True,width=96,renderer_depth=2)
  model.to_empty(device="cpu");model.initialize_weights()
  self.assertEqual(float(model.net.anchor_embed.proj.weight.abs().sum()),0)
  self.assertEqual(float(model.causal_semantic_encoder.action_proj.weight.abs().sum()),0)

 def test_s2_doubled_state_width_and_rate_diagnostics(self):
  torch.manual_seed(42)
  model=ASJepaSPT(width=96,encoder_depth=2,prior_depth=2,renderer_depth=2,state_width=32)
  frames=torch.randn(2,3,3,64,64);actions=torch.randn(2,3,10);result=model(frames,actions,frames[:,0],return_metrics=True)
  self.assertTrue(bool(torch.isfinite(result["loss"])))
  result["loss"].backward()
  self.assertEqual(model.state_dim,128);self.assertEqual(tuple(model.encoded(frames).shape),(2,3,4,32))
  self.assertEqual(tuple(model.training_variable_rate.shape),(128,))
  self.assertIn("rate_share",result["metrics"])
  state=model.encode(dict(pixels=frames))["emb"];self.assertEqual(tuple(model.predict(state[:,:2],actions[:,:2]).shape),(2,2,128))
if __name__=="__main__":unittest.main()
