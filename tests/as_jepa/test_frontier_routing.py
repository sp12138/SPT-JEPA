"""Verify that routing changes gradients, not forward values or model size."""
import unittest
from unittest.mock import patch
import torch
import test_frontier as contracts

ROUTES = ["original", "target-stop", "input-stop", "rate-stop", "render-stop"]


class RoutingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        contracts.FrontierContracts.setUpClass()

    def test_same_values_and_owned_gradients(self):
        helper = contracts.FrontierContracts()
        batch = helper.batch()
        models, gradients = {}, {}
        expected = None
        for method in ROUTES:
            model = helper.model(method)
            torch.manual_seed(64)
            loss, _ = model.loss(batch, step=7)
            if expected is None:
                expected = loss.detach()
            torch.testing.assert_close(loss.detach(), expected, rtol=0, atol=0)
            loss.backward()
            gradients[method] = {k: p.grad.clone() if p.grad is not None else None for k,p in model.named_parameters()}
            self.assertTrue(all(g is None or bool(torch.isfinite(g).all()) for g in gradients[method].values()))
            models[method] = model
            self.assertEqual(set(model.state_dict()), set(models["original"].state_dict()))
            for key in model.state_dict():
                torch.testing.assert_close(model.state_dict()[key], models["original"].state_dict()[key], rtol=0, atol=0)
        base = gradients["original"]
        for method in ROUTES[1:]:
            # Detaching encoder states does not change prior/action/renderer parameter gradients.
            for name, value in base.items():
                if name.startswith(("predictor.", "action_encoder.", "renderer.")):
                    other = gradients[method][name]
                    self.assertEqual(value is None, other is None)
                    if value is not None:
                        torch.testing.assert_close(value, other, rtol=0, atol=0)
            self.assertTrue(any(value is not None and gradients[method][name] is not None and
                                bool((value-gradients[method][name]).abs().max()>1e-10)
                                for name,value in base.items() if name.startswith(("encoder.", "pool."))))
        for name,value in base.items():
            if value is not None and name.startswith(("encoder.", "pool.")):
                # Both entry and FM gradients are present twice on either side.
                torch.testing.assert_close(value+gradients["rate-stop"][name],
                    gradients["target-stop"][name]+gradients["input-stop"][name], rtol=3e-4, atol=2e-5)

    def test_sample_factorization_with_fixed_randomness(self):
        helper = contracts.FrontierContracts()
        batch = helper.batch()
        fixednoise = lambda x,*args,**kwargs: torch.full_like(x,.3)
        fixedtime = lambda n,device,shift: torch.full((n,1),.5,device=device)
        for method in ROUTES:
            with self.subTest(method=method):
                model = helper.model(method)
                with patch("torch.randn_like",fixednoise), patch("vforge.models.as_jepa.flow.draw_timesteps",fixedtime):
                    whole,_ = model.loss(batch)
                    separate = [model.loss({k:v[i:i+1] for k,v in batch.items()})[0] for i in range(2)]
                torch.testing.assert_close(whole,torch.stack(separate).mean(),rtol=5e-5,atol=5e-6)


if __name__ == "__main__":
    unittest.main()
