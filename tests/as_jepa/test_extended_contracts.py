import unittest,torch
from dataclasses import replace
from vforge.ops import OpsConfig,initialize_ops
from vforge.models.as_jepa.renderer import AnchoredWorldDiT,renderer_config,SlotLinear
from vforge.models.as_jepa.flow import timestep_shift,draw_timesteps
from vforge.models.as_jepa.codes import entry_nats,surprise_nats
from vforge.models.rep_world.world import WorldDiT
from vforge.models.rep_world.model import RepWorld,FlowConfig
from vforge.models.rep_world.config import CodecGeometry,ConditionerConfig
class ExtendedContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2);initialize_ops(OpsConfig(backend="sdpa_math",allow_fallback=False),device=torch.device("cpu"),parallel_state=None)
 def test_same_explicit_flow_shift_and_uniform_draw(self):
  geo=CodecGeometry("pixel64","rgb1p4",1,48,1,16,True)
  cfg=replace(renderer_config(16),width=48,depth=2,heads=3,adaln_lora_dim=48)
  rw=RepWorld(geo,cfg,ConditionerConfig(input="latent",width=16,depth=2,heads=2,feature_layers=(1,)),FlowConfig())
  self.assertAlmostEqual(timestep_shift(cfg),rw.timestep_shift,places=12)
  u=torch.tensor([[.05],[.25],[.5],[.75],[.95]])
  torch.testing.assert_close(draw_timesteps(5,"cpu",timestep_shift(cfg),uniform=u),rw.draw_timesteps((5,1),torch.zeros(5,1),uniform=u))
 def test_zero_anchor_matches_plain_and_anchor_learns(self):
  cfg=replace(renderer_config(16),width=48,depth=2,heads=3,adaln_lora_dim=48)
  anchored=AnchoredWorldDiT(cfg,state_dim=16,num_slots=4).eval();plain=WorldDiT(cfg).eval()
  plain.semantic_raw_projector=torch.nn.Sequential(torch.nn.Identity(),SlotLinear(16,48,4))
  torch.nn.init.normal_(anchored.final_layer.linear.weight,std=.02)
  result=plain.load_state_dict(anchored.state_dict(),strict=False)
  self.assertTrue(all(k.startswith("anchor_embed.") for k in result.unexpected_keys));self.assertFalse(result.missing_keys)
  y=torch.randn(2,48,1,16,16);anchor=torch.randn_like(y);state=torch.randn(2,4,16);t=torch.rand(2,1)
  with anchored.anchored(anchor):a=anchored(y,t,condition=state)
  b=plain(y,t,condition=state);torch.testing.assert_close(a,b,rtol=0,atol=0)
  with self.assertRaises(RuntimeError):anchored(y,t,condition=state)
  optimizer=torch.optim.SGD(anchored.parameters(),lr=.01);optimizer.zero_grad()
  with anchored.anchored(anchor):loss=anchored(y,t,condition=state).square().mean()
  loss.backward();optimizer.step();self.assertGreater(float(anchored.anchor_embed.proj.weight.detach().abs().sum()),0)
 def test_gaussian_prices_all_noise_levels(self):
  a=torch.randn(2,3,4,16);m=torch.randn_like(a)
  for sigma in [.03,.1,.3]:
   q=torch.distributions.Normal(a,torch.full_like(a,sigma));p=torch.distributions.Normal(m,torch.full_like(a,sigma))
   torch.testing.assert_close(surprise_nats(a,m,sigma),torch.distributions.kl_divergence(q,p).sum((-1,-2)))
   q0=torch.distributions.Normal(a[:,0],torch.full_like(a[:,0],sigma));p0=torch.distributions.Normal(torch.zeros_like(a[:,0]),torch.ones_like(a[:,0]))
   torch.testing.assert_close(entry_nats(a[:,0],sigma),torch.distributions.kl_divergence(q0,p0).sum((-1,-2)))
if __name__=="__main__":unittest.main()
