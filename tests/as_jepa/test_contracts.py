import json,unittest
from pathlib import Path
import torch
from vforge.ops import OpsConfig,initialize_ops
from vforge.models.as_jepa.pixels import pack,unpack
from vforge.models.as_jepa.codes import entry_nats,surprise_nats,kl_const
from vforge.models.as_jepa.prior import StatePrior
from as_jepa_lewm.model import ASJepaLeWM

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        initialize_ops(OpsConfig(backend="sdpa_math",allow_fallback=False),device=torch.device("cpu"),parallel_state=None)
    def test_pixel_packing_is_lossless(self):
        x=torch.randn(3,3,64,64);torch.testing.assert_close(unpack(pack(x)),x,rtol=0,atol=0)
    def test_kl_is_gaussian_kl_and_both_targets_have_gradients(self):
        a=torch.randn(2,4,16,requires_grad=True);m=torch.randn(2,4,16,requires_grad=True)
        sigma=.1;rate=surprise_nats(a,m,sigma)
        q=torch.distributions.Normal(a,torch.full_like(a,sigma));p=torch.distributions.Normal(m,torch.full_like(m,sigma))
        torch.testing.assert_close(rate,torch.distributions.kl_divergence(q,p).sum((-2,-1)))
        rate.sum().backward();self.assertGreater(float(a.grad.abs().sum()),0);self.assertGreater(float(m.grad.abs().sum()),0)
        torch.testing.assert_close(entry_nats(a.detach(),sigma),torch.distributions.kl_divergence(q,torch.distributions.Normal(torch.zeros_like(a),torch.ones_like(a))).sum((-2,-1)))
    def test_prior_initialization_is_identity_and_causal(self):
        prior=StatePrior(4,16,10,width=48,depth=2,heads=3);prior.initialize_weights();prior.eval()
        s=torch.randn(2,5,4,16);a=torch.randn(2,5,10)
        torch.testing.assert_close(prior(s,a),s,rtol=0,atol=0)
        torch.nn.init.normal_(prior.head.weight,std=.02)
        changed=s.clone();changed[:,3:]+=100
        torch.testing.assert_close(prior(s,a)[:,:3],prior(changed,a)[:,:3])
    def test_lewm_prior_identity_and_causality_and_frame_isolation(self):
        torch.manual_seed(7);model=ASJepaLeWM(encoder_depth=2,renderer_width=96,renderer_depth=2,renderer_heads=3).eval()
        s=torch.randn(2,5,64);a=model.action_encoder(torch.randn(2,5,10))
        torch.testing.assert_close(model.predict(s,a),s,rtol=0,atol=0)
        torch.nn.init.normal_(model.predictor.transformer.output_proj.weight,std=.02)
        changed=s.clone();changed[:,3:]+=100
        torch.testing.assert_close(model.predict(s,a)[:,:3],model.predict(changed,a)[:,:3])
        frames=torch.randn(1,2,3,224,224);other=frames.clone();other[:,1]+=10
        torch.testing.assert_close(model.encode(dict(pixels=frames))["emb"][:,0],model.encode(dict(pixels=other))["emb"][:,0])
    def test_renderer_anchor_has_no_encoder_path_and_state_slot_matters(self):
        model=ASJepaLeWM(encoder_depth=2,renderer_width=96,renderer_depth=2,renderer_heads=3).eval()
        render=model.renderer
        self.assertEqual(float(render.anchor_embed.proj.weight.abs().sum()),0)
        self.assertEqual(float(render.final_layer.linear.weight.abs().sum()),0)
        torch.nn.init.normal_(render.final_layer.linear.weight,std=.02);torch.nn.init.normal_(render.anchor_embed.proj.weight,std=.02)
        y=torch.randn(2,48,1,16,16);anchor=torch.randn_like(y);s=torch.randn(2,4,16);t=torch.ones(2,1)
        with render.anchored(anchor):first=render(y,t,condition=s)
        with render.anchored(anchor*0):second=render(y,t,condition=s)
        self.assertGreater(float((first-second).abs().mean()),1e-6)
        reordered=s[:,[1,0,2,3]]
        with render.anchored(anchor):third=render(y,t,condition=reordered)
        self.assertGreater(float((first-third).abs().mean()),1e-7)
    def test_pinned_runtime_hashes_are_unchanged(self):
        import hashlib
        root=Path(__file__).resolve().parents[2]
        for name,expected in json.loads((root/"provenance/upstream_runtime_hashes.json").read_text()).items():
            self.assertEqual(hashlib.sha256((root/"runtime"/name).read_bytes()).hexdigest(),expected,name)

if __name__=="__main__":unittest.main()
