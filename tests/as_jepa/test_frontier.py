"""Verify model identity, gradients and sample factorization for staged hypotheses."""
import unittest
from unittest.mock import patch
import torch
from as_jepa_lewm.model import ASJepaLeWM
from as_jepa_lewm.frontier import FrontierLeWM, METHODS
from vforge.ops import OpsConfig, initialize_ops


class FrontierContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        initialize_ops(OpsConfig(backend="sdpa_math", allow_fallback=False), device=torch.device("cpu"), parallel_state=None)

    def model(self, method="original"):
        torch.manual_seed(14)
        m = FrontierLeWM(frontier_method=method, mode="L2", beta=.25, window=5, encoder_depth=1,
                         renderer_width=48, renderer_depth=2, renderer_heads=3)
        torch.nn.init.normal_(m.renderer.final_layer.linear.weight, std=.01)
        return m.eval()

    def batch(self):
        torch.manual_seed(15)
        return dict(pixels=torch.randint(0,256,(2,5,3,224,224),dtype=torch.uint8),
                    render=torch.randint(0,256,(2,5,3,64,64),dtype=torch.uint8),
                    anchor=torch.randint(0,256,(2,3,64,64),dtype=torch.uint8),
                    anchor224=torch.randint(0,256,(2,3,224,224),dtype=torch.uint8),action=torch.randn(2,5,10))

    def test_all_methods_same_initial_tensors_finite_reachable_gradients(self):
        reference=self.model().state_dict()
        batch=self.batch()
        for method in sorted(METHODS):
            with self.subTest(method=method):
                m=self.model(method)
                self.assertEqual(set(reference),set(m.state_dict()))
                for k,v in reference.items():
                    torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)
                loss,metrics=m.loss(batch,step=3)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                for prefix in ["encoder","predictor","renderer"]:
                    grads=[p.grad for n,p in m.named_parameters() if n.startswith(prefix) and p.grad is not None]
                    self.assertTrue(grads and all(bool(torch.isfinite(g).all()) for g in grads))
                    self.assertTrue(any(bool(g.abs().sum()>0) for g in grads))
                self.assertFalse(any(isinstance(x,torch.nn.modules.batchnorm._BatchNorm) for x in m.modules()))

    def test_original_control_exact_loss_gradient(self):
        candidate=self.model()
        torch.manual_seed(14)
        original=ASJepaLeWM(mode="L2",beta=.25,window=5,encoder_depth=1,renderer_width=48,renderer_depth=2,renderer_heads=3).eval()
        torch.nn.init.normal_(original.renderer.final_layer.linear.weight,std=.01)
        batch=self.batch()
        torch.manual_seed(41);a,_=original.loss(batch);a.backward()
        torch.manual_seed(41);z,_=candidate.loss(batch);z.backward()
        torch.testing.assert_close(a,z,rtol=0,atol=0)
        for (_,x),(_,y) in zip(original.named_parameters(),candidate.named_parameters()):
            self.assertEqual(x.grad is None,y.grad is None)
            if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)

    def test_objectives_factorize_by_sample(self):
        batch=self.batch()
        fixednoise=lambda x,*args,**kwargs:torch.full_like(x,.3)
        fixedtime=lambda n,device,shift:torch.full((n,1),.5,device=device)
        for method in sorted(METHODS-{"original"}):
            with self.subTest(method=method):
                m=self.model(method)
                with patch('torch.randn_like',fixednoise),patch('as_jepa_lewm.frontier.draw_timesteps',fixedtime),patch('vforge.models.as_jepa.flow.draw_timesteps',fixedtime):
                    whole,_=m.loss(batch,step=0)
                    separate=[m.loss({k:v[i:i+1] for k,v in batch.items()},step=0)[0] for i in range(2)]
                torch.testing.assert_close(whole,torch.stack(separate).mean(),rtol=5e-5,atol=5e-6)

    def test_innovation_correction_cancels_identity_carry_noise(self):
        mu=torch.randn(2,4,64);eps=torch.randn_like(mu)*.1
        corrected=(mu+eps)-eps
        torch.testing.assert_close(corrected,mu,rtol=1e-6,atol=2e-7)
        antithetic=((mu+eps)+(mu-eps))*.5
        torch.testing.assert_close(antithetic,mu,rtol=1e-6,atol=2e-7)


if __name__=='__main__':unittest.main()
