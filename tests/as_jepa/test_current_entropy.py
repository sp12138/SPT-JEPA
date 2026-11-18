"""Units, collapse and isolation contracts for factual forward prediction."""
import unittest

import torch

from experiments.robocasa_stable_scaling.current_entropy import factual_prediction_rate
from experiments.robocasa_stable_scaling.visible_effect import (
    fit_visible_posterior, visible_effect_rate, visible_probability)


class FactualRateContracts(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(9017)
        self.error = torch.randn(3, 1, 8, generator=generator)
        self.action = torch.randn(3, 1, 8, generator=generator)

    def test_original_rate_is_bitwise_unchanged(self):
        expected = self.error.float().square().sum((-2, -1)) / (2*.1**2)
        self.assertTrue(torch.equal(expected, factual_prediction_rate(
            self.error, self.action, .1)))

    def test_common_length_change_preserves_relative_prediction(self):
        reference = factual_prediction_rate(self.error, self.action, .1, True)
        for scale in (.01, .3, 7.):
            actual = factual_prediction_rate(
                self.error*scale, self.action*scale, .1, True)
            torch.testing.assert_close(actual, reference)

    def test_collapsed_nonzero_commands_have_a_recovery_gradient(self):
        action = torch.tensor([[[.01, 0.]], [[.8, 0.]]])
        error = (-action).detach().requires_grad_()
        rate = factual_prediction_rate(error, action, .1, True)
        torch.testing.assert_close(rate, torch.full((2,), 50.))
        rate.sum().backward()
        torch.testing.assert_close(error.grad.norm(dim=-1).flatten()*
                                   action.norm(dim=-1).flatten(),
                                   torch.full((2,), 100.))
        perfect = factual_prediction_rate(torch.zeros_like(action), action, .1, True)
        self.assertTrue(torch.equal(perfect, torch.zeros_like(perfect)))

    def test_zero_commands_keep_consistency_and_samples_stay_independent(self):
        action = self.action.clone(); action[0] = 0
        error = self.error.clone().requires_grad_()
        rate = factual_prediction_rate(error, action, .1, True)
        torch.testing.assert_close(rate[0], factual_prediction_rate(
            error[:1], action[:1], .1)[0])
        grad = torch.autograd.grad(rate[0], error)[0]
        self.assertTrue(torch.equal(grad[1:], torch.zeros_like(grad[1:])))
        alone = factual_prediction_rate(error[:1], action[:1], .1, True)
        self.assertTrue(torch.equal(rate[:1], alone))


class VisibleEffectContracts(unittest.TestCase):
    def test_rgb_posterior_uses_only_the_observed_transition(self):
        pixels = torch.zeros(2, 2, 4, 3, 5, 5, dtype=torch.uint8)
        pixels[0, 1, 1:] = 2
        p = visible_probability(pixels, 2., -1.)
        log_change = torch.tensor(1 + 1/150).log()
        torch.testing.assert_close(p[0, 0], torch.sigmoid(2 * log_change - 1))
        pixels[1] = 171
        pixels[0, :, 2:] = 219
        self.assertTrue(torch.equal(p[0, :1], visible_probability(pixels.flip(1), 2., -1.)[0, :1]))

    def test_factual_and_silent_endpoints_and_collapse_gradient(self):
        code = torch.zeros(2, 2, 3, dtype=torch.float64, requires_grad=True)
        actions = torch.tensor([[[1., 0., 0.]], [[1., 0., 0.]]], dtype=torch.float64)
        p = torch.tensor([[0.], [1.]], requires_grad=True)
        loss, _ = visible_effect_rate(code, actions, p, .1)
        torch.testing.assert_close(loss[:, 0], .5 * torch.tensor([1., 101.], dtype=code.dtype).log())
        loss.sum().backward()
        self.assertEqual(code.grad[0].abs().sum(), 0)
        self.assertLess(code.grad[1, 1, 0], 0)
        self.assertIsNone(p.grad)

    def test_calibration_absorbs_sensor_units(self):
        import numpy as np
        rng = np.random.default_rng(11)
        x = np.r_[rng.normal(-4, .6, 400), rng.normal(2, .6, 1600)]
        fit = fit_visible_posterior(x)
        shifted = fit_visible_posterior(x + .8)
        np.testing.assert_allclose(fit['slope']*x + fit['bias'],
            shifted['slope']*(x+.8) + shifted['bias'], atol=1e-4)


if __name__ == '__main__':
    unittest.main()
