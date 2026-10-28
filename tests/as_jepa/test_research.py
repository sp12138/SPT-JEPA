import tempfile,unittest,json
from pathlib import Path
import torch,numpy as np
from vforge.ops import OpsConfig,initialize_ops
from as_jepa_lewm.research import ResearchLeWM
class ResearchContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2);initialize_ops(OpsConfig(backend="sdpa_math",allow_fallback=False),device=torch.device("cpu"),parallel_state=None)
 def test_all_variants_are_finite_and_checkpoint_roundtrip(self):
  for method in ["baseline","snr_weighted","geometry_noise","correlated_noise","canonical_color","forecast_render","gradient_constraint","predictable_weight","action_forks"]:
   with self.subTest(method=method):
    torch.manual_seed(14)
    cfg=dict(method=method,encoder_depth=1,renderer_width=48,renderer_depth=2,renderer_heads=3)
    model=ResearchLeWM(**cfg);model.train()
    # Perturb output heads so this checks real conditioning gradients beyond trivial zero initialization.
    torch.nn.init.normal_(model.renderer.final_layer.linear.weight,std=.01)
    batch=dict(pixels=torch.randint(0,256,(1,3,3,224,224),dtype=torch.uint8),render=torch.randint(0,256,(1,3,3,64,64),dtype=torch.uint8),anchor=torch.randint(0,256,(1,3,64,64),dtype=torch.uint8),anchor224=torch.randint(0,256,(1,3,224,224),dtype=torch.uint8),action=torch.randn(1,3,10))
    if method=="action_forks":
     batch.update(fork_context=torch.randn(1,3,224,224),fork_futures=torch.randn(1,4,3,224,224),fork_actions=torch.randn(1,4,8,10))
    loss,metrics=model.loss(batch,step=2);self.assertTrue(bool(torch.isfinite(loss)),method);loss.backward()
    self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.encoder.parameters()))
    fresh=ResearchLeWM(**cfg);fresh.load_state_dict(model.state_dict(),strict=True)
    self.assertTrue(all(bool(torch.isfinite(p).all()) for p in fresh.parameters()))
 def test_correlated_context_noise_preserves_marginal_variance(self):
  m=ResearchLeWM(method="correlated_noise",encoder_depth=1,renderer_width=48,renderer_depth=2,renderer_heads=3)
  samples=m.context_samples(torch.zeros(1,3,64),4096,torch.Generator().manual_seed(5))
  empirical=samples[:,:,0]
  self.assertLess(abs(float(empirical.var())-.01),.001)
  correlation=torch.corrcoef(empirical.T)[0,1];self.assertLess(abs(float(correlation)-.8),.025)
if __name__=="__main__":unittest.main()
