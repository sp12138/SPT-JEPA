"""Controls for preserving the original model, sample objective and causal forecast path."""
import math
import unittest
from unittest.mock import patch
import torch
from as_jepa_lewm.model import ASJepaLeWM
from as_jepa_lewm.robust import RobustLeWM
from vforge.ops import OpsConfig, initialize_ops


class RobustnessContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        initialize_ops(OpsConfig(backend="sdpa_math", allow_fallback=False), device=torch.device("cpu"), parallel_state=None)

    def model(self, cls=RobustLeWM, **extra):
        torch.manual_seed(11)
        value = cls(mode="L2", beta=.25, window=3, encoder_depth=1,
                    renderer_width=48, renderer_depth=2, renderer_heads=3, **extra)
        torch.nn.init.normal_(value.renderer.final_layer.linear.weight, std=.01)
        return value.eval()

    def batch(self):
        torch.manual_seed(18)
        return dict(pixels=torch.randint(0, 256, (2, 3, 3, 224, 224), dtype=torch.uint8),
                    render=torch.randint(0, 256, (2, 3, 3, 64, 64), dtype=torch.uint8),
                    anchor=torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8),
                    anchor224=torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8),
                    action=torch.randn(2, 3, 10))

    def test_disabled_extension_exactly_matches_original_loss_and_gradients(self):
        original, control = self.model(ASJepaLeWM), self.model()
        self.assertEqual(set(original.state_dict()), set(control.state_dict()))
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, control.state_dict()[key], rtol=0, atol=0)
        batch = self.batch()
        torch.manual_seed(29)
        a, _ = original.loss(batch)
        a.backward()
        torch.manual_seed(29)
        b, _ = control.loss(batch)
        b.backward()
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        for (_, x), (_, y) in zip(original.named_parameters(), control.named_parameters()):
            self.assertEqual(x.grad is None, y.grad is None)
            if x.grad is not None:
                torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)

    def test_all_extensions_preserve_initial_tensors_and_parameter_count(self):
        original = self.model(ASJepaLeWM)
        for rho, fraction in [(.8, 0), (0, .5), (.8, .5)]:
            candidate = self.model(rho=rho, forecast_fraction=fraction)
            self.assertEqual(sum(x.numel() for x in original.parameters()), sum(x.numel() for x in candidate.parameters()))
            self.assertEqual(set(original.state_dict()), set(candidate.state_dict()))
            for key, value in original.state_dict().items():
                torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
            self.assertFalse(any(isinstance(x, torch.nn.modules.batchnorm._BatchNorm) for x in candidate.modules()))

    def test_forecast_condition_is_causal_and_outer_rng_is_unchanged(self):
        original, forecast = self.model(ASJepaLeWM), self.model(forecast_fraction=1.)
        batch = self.batch()
        recorded = []
        def capture(renderer, y, state, anchor, shift):
            recorded.append(state.detach().clone())
            return state.square().mean()
        torch.manual_seed(39)
        with patch("as_jepa_lewm.model.flow_loss", capture):
            original.loss(batch, step=7)
        original_rng = torch.get_rng_state()
        torch.manual_seed(39)
        with patch("as_jepa_lewm.robust.flow_loss", capture):
            loss, _ = forecast.loss(batch, step=7)
        torch.testing.assert_close(original_rng, torch.get_rng_state(), rtol=0, atol=0)
        condition = recorded[-1]
        changed = {**batch, "pixels": batch["pixels"].clone()}
        changed["pixels"][:, 1:] = 255 - changed["pixels"][:, 1:]
        torch.manual_seed(39)
        with patch("as_jepa_lewm.robust.flow_loss", capture):
            forecast.loss(changed, step=7)
        torch.testing.assert_close(condition, recorded[-1], rtol=0, atol=0)
        loss.backward()
        self.assertTrue(any(x.grad is not None and bool(x.grad.abs().sum() > 0) for x in forecast.predictor.parameters()))

    def test_markov_marginals_and_conditional_gaussian_kl(self):
        model = self.model(rho=.8)
        draws = model.context_samples(torch.zeros(1, 3, 64), 8192, torch.Generator().manual_seed(23))
        torch.testing.assert_close(draws.var((0, 1)), torch.full((64,), .01), rtol=.04, atol=.0001)
        self.assertLess(abs(float(torch.corrcoef(draws[:, :, 0].T)[0, 1]) - .8), .02)
        mu, previous_noise, predicted = torch.randn(3, 64), torch.randn(3, 64) * .1, torch.randn(3, 64)
        conditional_mean = mu + .8 * previous_noise
        q = torch.distributions.Normal(conditional_mean, .1 * math.sqrt(1 - .8 ** 2))
        p = torch.distributions.Normal(predicted, .1)
        expected = torch.distributions.kl_divergence(q, p)
        actual = (conditional_mean - predicted).square() / (.02) + .5 * (-.8 ** 2 - math.log1p(-.8 ** 2))
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
