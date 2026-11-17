"""Units, collapse and isolation contracts for factual forward prediction."""
import unittest

import torch

from experiments.robocasa_stable_scaling.current_entropy import factual_prediction_rate


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


if __name__ == '__main__':
    unittest.main()
