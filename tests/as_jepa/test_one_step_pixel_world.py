"""Conditional likelihood units and physical-sample loss contracts."""
import unittest

import torch

from experiments.robocasa_stable_scaling.one_step_pixel_world import (
    conditional_world_terms, input_noise_std, paired_pixel_mse)


class ConditionalWorldContracts(unittest.TestCase):
    def test_value_and_gradient_match_independent_gaussian_likelihoods(self):
        g = torch.Generator().manual_seed(8131)
        prediction = torch.randn(12, 3, generator=g, dtype=torch.float64, requires_grad=True)
        target = torch.randn(12, 3, generator=g, dtype=torch.float64)
        error = torch.randn(2, 3, 2, 3, generator=g, dtype=torch.float64, requires_grad=True)
        mse = paired_pixel_mse(prediction, target, 2, 3)
        actual, _, _ = conditional_world_terms(mse, error, torch.ones(2, 3, 14), .1, 6)
        image = torch.distributions.Normal(prediction, 1.)
        constant = torch.distributions.Normal(target, 1.)
        distortion = (-image.log_prob(target)+constant.log_prob(target)).sum(-1)
        distortion = distortion.reshape(2, 2, 3).sum(1)
        q = torch.distributions.Normal(error, .1)
        p = torch.distributions.Normal(torch.zeros_like(error), .1)
        kl = torch.distributions.kl_divergence(q, p).sum((-2, -1))
        expected = (distortion+kl)/6
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        a = torch.autograd.grad(actual.sum(), (prediction, error), retain_graph=True)
        b = torch.autograd.grad(expected.sum(), (prediction, error))
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)

    def test_input_scale_is_detached_and_zero_retains_absolute_consistency(self):
        commands = torch.tensor([[[2., 0.], [0., 0.], [.1, 0.]]], requires_grad=True)
        error = torch.ones(1, 3, 2, 3, requires_grad=True)
        terms, _, weight = conditional_world_terms(torch.ones(1, 3), error,
            commands, .1, 6, 'action-scale')
        torch.testing.assert_close(weight, torch.tensor([[.25, 1., 100.]]))
        torch.testing.assert_close(terms, 50.5*weight)
        terms.sum().backward()
        self.assertIsNone(commands.grad)
        self.assertTrue(bool(torch.isfinite(error.grad).all()))

    def test_camera_time_pairing_and_other_physical_sample_isolation(self):
        rows = torch.tensor([1., 2., 3., 3., 4., 5., 5., 6., 7., 7., 8., 9.],
                            requires_grad=True)[:, None]
        actual = paired_pixel_mse(rows, torch.zeros_like(rows), 2, 3)
        torch.testing.assert_close(actual, torch.tensor([[5., 10., 17.], [37., 50., 65.]]))
        delta = torch.zeros(2, 3, 1, 2, requires_grad=True)
        terms, _, _ = conditional_world_terms(actual, delta, torch.ones(2, 3, 3), .1, 2)
        gradients = torch.autograd.grad(terms[0].sum(), (rows, delta))
        self.assertTrue(torch.equal(gradients[0][6:], torch.zeros_like(gradients[0][6:])))
        self.assertTrue(torch.equal(gradients[1][1], torch.zeros_like(gradients[1][1])))

    def test_input_dependent_covariance_matches_gaussian_kl_and_pixel_units(self):
        commands = torch.tensor([[[.1, 0.], [1., 0.], [0., 0.]]],
                                dtype=torch.float64, requires_grad=True)
        sigma = input_noise_std(commands, .1)
        error = torch.tensor([[[[.01, .02]], [[.1, .2]], [[.03, .04]]]],
                             dtype=torch.float64, requires_grad=True)
        mse = torch.ones(1, 3, dtype=torch.float64, requires_grad=True)
        actual, rate, weights = conditional_world_terms(mse, error, commands, sigma, 12)
        q = torch.distributions.Normal(error, sigma[..., None, None])
        p = torch.distributions.Normal(torch.zeros_like(error), sigma[..., None, None])
        independent_kl = torch.distributions.kl_divergence(q, p).sum((-2, -1))
        torch.testing.assert_close(actual, .5*mse+independent_kl/12)
        left = torch.autograd.grad(actual.sum(), (error, mse), retain_graph=True)
        right = torch.autograd.grad((.5*mse+independent_kl/12).sum(), (error, mse))
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(rate[0, 0], rate[0, 1])
        torch.testing.assert_close(weights, torch.ones_like(weights))
        self.assertFalse(sigma.requires_grad)
        self.assertEqual(float(sigma[0, 2]), .1)


if __name__ == '__main__':
    unittest.main()
