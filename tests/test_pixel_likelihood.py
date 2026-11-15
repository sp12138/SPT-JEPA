"""Independent likelihood and gradient checks for learned image precision."""
import math
import unittest

import torch

from experiments.robocasa_stable_scaling.pixel_likelihood import gaussian_pixel_nll


class PixelLikelihoodTest(unittest.TestCase):
    def test_matches_normal_distribution_values_and_gradients(self):
        target = torch.tensor([-.6, -.2, .15, .8])
        for initial in (-4., 0., 3.):
            prediction = torch.tensor([-.5, -.1, .3, .7], requires_grad=True)
            ratio = torch.tensor(initial, requires_grad=True)
            loss = gaussian_pixel_nll((prediction-target).square().mean(), ratio)
            reference = -torch.distributions.Normal(
                prediction, (.5*ratio.exp()).sqrt()).log_prob(target).mean()
            reference = reference-.5*math.log(math.pi)
            self.assertTrue(torch.allclose(loss, reference, atol=2e-6, rtol=2e-6))
            actual = torch.autograd.grad(loss, (prediction, ratio), retain_graph=True)
            expected = torch.autograd.grad(reference, (prediction, ratio))
            for a, b in zip(actual, expected):
                self.assertTrue(torch.allclose(a, b, atol=2e-6, rtol=2e-6))

    def test_initial_loss_and_image_gradient_match_existing_mse(self):
        error = torch.tensor([.01, -.03, .2], requires_grad=True)
        mse = error.square().mean()
        loss = gaussian_pixel_nll(mse, torch.zeros((), requires_grad=True))
        self.assertTrue(torch.equal(loss, mse))
        actual = torch.autograd.grad(loss, error, retain_graph=True)[0]
        expected = torch.autograd.grad(mse, error)[0]
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
