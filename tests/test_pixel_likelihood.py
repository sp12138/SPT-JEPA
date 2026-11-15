"""Independent likelihood and gradient checks for learned image precision."""
import math
import unittest

import torch

from experiments.robocasa_stable_scaling.pixel_likelihood import (
    gaussian_pixel_nll, haar_bands, profiled_pixel_nll)


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

    def test_profile_is_per_image_maximum_likelihood(self):
        error = torch.linspace(-.3, .2, 2*3*7*11).reshape(2, 3, 7, 11)
        error.requires_grad_(True)
        actual, _ = profiled_pixel_nll(error, "profile-global")
        variance = error.square().flatten(1).mean(1)
        reference = -torch.distributions.Normal(torch.zeros_like(error),
            variance.sqrt()[:, None, None, None]).log_prob(error).mean()
        reference = reference-.5*math.log(2*math.pi)
        self.assertTrue(torch.allclose(actual, reference, atol=3e-7, rtol=3e-7))
        ga = torch.autograd.grad(actual, error, retain_graph=True)[0]
        gr = torch.autograd.grad(reference, error)[0]
        self.assertTrue(torch.allclose(ga, gr, atol=3e-7, rtol=3e-7))
        zero = torch.zeros(1, 3, 8, 8, requires_grad=True)
        loss, _ = profiled_pixel_nll(zero, "profile-haar")
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.equal(torch.autograd.grad(loss, zero)[0], zero))

    def test_haar_preserves_full_image_energy_and_constant_mode(self):
        generator = torch.Generator().manual_seed(17)
        for height, width in ((224, 224), (32, 16), (7, 11)):
            image = torch.randn(2, 3, height, width, generator=generator)
            bands = list(haar_bands(image))
            self.assertEqual(sum(x.numel() for x in bands), image.numel())
            actual = sum(x.square().flatten(1).sum(1) for x in bands)
            expected = image.square().flatten(1).sum(1)
            self.assertTrue(torch.allclose(actual, expected, atol=.02, rtol=2e-6))
        bands = list(haar_bands(torch.ones(1, 3, 224, 224)))
        self.assertTrue(all(torch.count_nonzero(x) == 0 for x in bands[:-1]))
        self.assertTrue(torch.equal(bands[-1], torch.full((1, 3, 7, 7), 32.)))

    def test_profile_units_and_physical_samples_are_independent(self):
        error = .1*torch.randn(2, 3, 16, 16, generator=torch.Generator().manual_seed(18))
        for mode in ("profile-global", "profile-haar"):
            error.requires_grad_(True)
            actual, _ = profiled_pixel_nll(error, mode)
            a, _ = profiled_pixel_nll(error[:1], mode)
            b, _ = profiled_pixel_nll(error[1:], mode)
            self.assertTrue(torch.allclose(actual, (a+b)/2, atol=3e-7, rtol=3e-7))
            scaled, _ = profiled_pixel_nll(3*error, mode)
            self.assertTrue(torch.allclose(scaled-actual,
                torch.tensor(math.log(3)), atol=5e-7, rtol=5e-7))
            ga = torch.autograd.grad(a, error, retain_graph=True)[0]
            changed = torch.cat((error[:1], 10*error[1:]), 0)
            first, _ = profiled_pixel_nll(changed[:1], mode)
            gc = torch.autograd.grad(first, error)[0]
            self.assertTrue(torch.equal(ga, gc))


if __name__ == "__main__":
    unittest.main()
