import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from vforge.ops import OpsConfig,initialize_ops
from as_jepa_lewm.model import ASJepaLeWM
from as_jepa_tools.data import FullPushTWindows,full_phase_sampler

class CompletionContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2)
  initialize_ops(OpsConfig(backend="sdpa_math",allow_fallback=False),device=torch.device("cpu"),parallel_state=None)
 def test_configurable_state_width_and_strict_restore(self):
  cfg=dict(encoder_depth=1,renderer_width=48,renderer_depth=2,renderer_heads=3,state_width=32)
  torch.manual_seed(6);m=ASJepaLeWM(**cfg).eval();other=ASJepaLeWM(**cfg).eval()
  other.load_state_dict(m.state_dict(),strict=True)
  frames=torch.randn(1,2,3,224,224)
  mu=m.encode(dict(pixels=frames))["emb"];self.assertEqual(tuple(mu.shape),(1,2,128))
  action=m.action_encoder(torch.randn(1,2,10))
  torch.testing.assert_close(m.predict(mu,action),mu,rtol=0,atol=0)
  states=mu[:,0].unflatten(-1,(4,32))
  y=torch.randn(1,48,1,16,16)
  with m.renderer.anchored(y):out=m.renderer(y,torch.ones(1,1),condition=states)
  self.assertEqual(tuple(out.shape),tuple(y.shape))
 def test_all_phase_windows_do_not_cross_episode_boundaries(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);lengths=np.array([51,64]);offsets=np.array([0,51]);n=115
   (p/"dataset_card.json").write_text(json.dumps(dict(frameskip=5,phases=5)))
   arrays=dict(lengths=lengths,offsets=offsets,test_episodes=np.array([1]),raw_rows=np.arange(n),states=np.zeros((n,7),np.float32),actions=np.zeros((n,10),np.float32),pixels224=np.zeros((n,2,2,3),np.uint8),pixels64=np.zeros((n,2,2,3),np.uint8))
   for name,value in arrays.items():np.save(p/(name+".npy"),value)
   d=FullPushTWindows(p,window=8,gap=2,split="train")
   self.assertEqual(len(d),16)
   starts=[]
   for i in range(len(d)):
    b=d[i];rows=b["rows"].numpy();starts.append(b["start"])
    self.assertTrue(np.array_equal(np.diff(rows),np.full(7,5)))
    self.assertLess(int(rows[-1]),51);self.assertEqual((b["anchor_row"]-int(rows[0]))%5,0)
   self.assertEqual(starts,list(range(16)))
   sampler=full_phase_sampler(d,dp_rank=0,dp_world_size=1,batch_size=1,seed=0,shuffle=False,drop_last=False)
   sampler.set_epoch(1);encoded=list(sampler)
   self.assertEqual(encoded,list(range(16,32)))
   self.assertEqual([d.locate(i) for i in encoded],[d.locate(i) for i in range(16)])

if __name__=="__main__":unittest.main()
