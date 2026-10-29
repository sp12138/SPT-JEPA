"""Check causal data boundaries and the exact training-only corruption change."""
import json,tempfile,unittest
from pathlib import Path
import h5py,numpy as np,torch
from as_jepa_tools.benchmark_data import BenchmarkWindows
from as_jepa_lewm.model import ASJepaLeWM
from vforge.models.as_jepa.flow import draw_timesteps
from vforge.ops import initialize_ops,OpsConfig

class BenchmarkContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);initialize_ops(OpsConfig(backend="sdpa_math",allow_fallback=False),device=torch.device("cpu"),parallel_state=None)
    def test_shift_preserves_weights_sampler_and_increases_corruption(self):
        cfg=dict(mode="L2",window=3,encoder_depth=1,renderer_width=48,renderer_depth=2,renderer_heads=3,action_dim=25)
        torch.manual_seed(11);a=ASJepaLeWM(**cfg)
        torch.manual_seed(11);b=ASJepaLeWM(**cfg,flow_logit_shift=.8)
        self.assertEqual(a.shift,b.shift)
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
        u=torch.linspace(.001,.999,1000)[:,None]
        x=draw_timesteps(1000,"cpu",a.training_shift,uniform=u);y=draw_timesteps(1000,"cpu",b.training_shift,uniform=u)
        self.assertTrue(bool((y>x).all()));torch.testing.assert_close(torch.logit(y)-torch.logit(x),torch.full_like(x,.8),atol=2e-5,rtol=2e-5)
    def test_windows_do_not_cross_episodes_or_use_future_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);raw=p/"data.h5";n=42;actions=np.arange(2*n*2,dtype=np.float32).reshape(2*n,2)
            with h5py.File(raw,"w") as f:
                f["pixels"]=np.broadcast_to(np.arange(2*n,dtype=np.uint8)[:,None,None,None],(2*n,16,16,3));f["action"]=actions
            for k,v in dict(offsets=[0,n],lengths=[n,n],episode_ids=[10,20],train_episodes=[10],test_episodes=[20]).items():np.save(p/(k+".npy"),v)
            (p/"dataset_card.json").write_text(json.dumps(dict(h5_path=str(raw),frameskip=5,action_mean=[1,2],action_std=[2,3])))
            ds=BenchmarkWindows(p);self.assertEqual(len(ds),7)
            x=ds[6];self.assertEqual(x["episode"],10);self.assertEqual(x["anchor_row"],6)
            torch.testing.assert_close(x["anchor224"],x["pixels"][0],atol=0,rtol=0)
            np.testing.assert_array_equal(x["rows"].numpy(),np.arange(6,42,5))
            np.testing.assert_allclose(x["action"][0].numpy(),((actions[6:11]-[1,2])/[2,3]).reshape(-1),rtol=1e-6)
            self.assertEqual(BenchmarkWindows(p,split="test")[0]["episode"],20)
            ds.file.close()
if __name__=="__main__":unittest.main()
