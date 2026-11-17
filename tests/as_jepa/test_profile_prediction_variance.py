"""Contracts for profiling a conditional Gaussian code variance."""
import unittest

import torch

from experiments.robocasa_stable_scaling.current_entropy import profile_prediction_rate


class ProfileVarianceContracts(unittest.TestCase):
    def test_value_and_gradient_match_independent_normal_kl(self):
        generator = torch.Generator().manual_seed(9821)
        error = torch.randn(3, 5, generator=generator, dtype=torch.float64,
                            requires_grad=True)
        base = torch.tensor([.001, .03, .1], dtype=torch.float64)
        rate = error.square().sum(-1)/(2*base)
        actual = profile_prediction_rate(rate, 5)
        variance = base+error.square().mean(-1)
        q = torch.distributions.Normal(error, base.sqrt()[:, None])
        p = torch.distributions.Normal(torch.zeros_like(error), variance.sqrt()[:, None])
        expected = torch.distributions.kl_divergence(q, p).sum(-1)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        a = torch.autograd.grad(actual.sum(), error, retain_graph=True)[0]
        b = torch.autograd.grad(expected.sum(), error)[0]
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)

    def test_profile_is_no_worse_than_feasible_predictive_variances(self):
        error = torch.tensor([[.1, .3, -.2]], dtype=torch.float64)
        base = .02
        best = profile_prediction_rate(error.square().sum(-1)/(2*base), 3)
        optimum = base+float(error.square().mean())
        q = torch.distributions.Normal(error, base**.5)
        for variance in (base, max(base, optimum/2), optimum*2, optimum*10):
            p = torch.distributions.Normal(torch.zeros_like(error), variance**.5)
            other = torch.distributions.kl_divergence(q, p).sum(-1)
            self.assertTrue(bool((best <= other+1e-12).all()))

    def test_nonzero_collapsed_effect_retains_recovery_gradient(self):
        error = torch.tensor([[.01, .1, .4]], dtype=torch.float64,
                             requires_grad=True)
        rate = error.square().sum(-1)/.02
        loss = profile_prediction_rate(rate, 3)
        loss.sum().backward()
        self.assertGreater(float(loss.detach()), 0)
        self.assertTrue(bool(torch.isfinite(error.grad).all()))
        self.assertGreater(float(error.grad.norm()), 0)
        self.assertEqual(float(profile_prediction_rate(torch.zeros(1), 3)), 0)

    def test_other_samples_do_not_change_value_or_gradient(self):
        rate = torch.tensor([.1, 10., 100.], requires_grad=True)
        value = profile_prediction_rate(rate, 128)
        grad = torch.autograd.grad(value[0], rate)[0]
        self.assertTrue(torch.equal(grad[1:], torch.zeros_like(grad[1:])))
        torch.testing.assert_close(value[:1], profile_prediction_rate(rate[:1], 128))


class FullCovarianceContracts(unittest.TestCase):
    def test_value_and_gradient_match_full_multivariate_normal_kl(self):
        generator = torch.Generator().manual_seed(9401)
        error = torch.randn(3, 5, generator=generator, dtype=torch.float64,
                            requires_grad=True)
        base = torch.tensor([.001, .03, .1], dtype=torch.float64)
        unit = torch.eye(5, dtype=torch.float64)
        q = torch.distributions.MultivariateNormal(error, base[:, None, None]*unit)
        covariance = base[:, None, None]*unit+error[:, :, None]*error[:, None, :]
        p = torch.distributions.MultivariateNormal(torch.zeros_like(error), covariance)
        expected = torch.distributions.kl_divergence(q, p)
        actual = profile_prediction_rate(error.square().sum(-1)/(2*base), 5, 'full')
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
        a = torch.autograd.grad(actual.sum(), error, retain_graph=True)[0]
        b = torch.autograd.grad(expected.sum(), error)[0]
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-8)

    def test_profile_beats_other_feasible_covariances(self):
        error = torch.tensor([.1, .3, -.2], dtype=torch.float64)
        base = .02
        unit = torch.eye(3, dtype=torch.float64)
        q = torch.distributions.MultivariateNormal(error, base*unit)
        actual = profile_prediction_rate(error.square().sum()/(2*base), 3, 'full')
        generator = torch.Generator().manual_seed(191)
        for _ in range(8):
            factor = torch.randn(3, 3, generator=generator, dtype=torch.float64)
            covariance = base*unit+factor@factor.T
            p = torch.distributions.MultivariateNormal(torch.zeros_like(error), covariance)
            self.assertLessEqual(float(actual), float(torch.distributions.kl_divergence(q, p))+1e-12)

    def test_padding_and_other_samples_leave_full_profile_unchanged(self):
        error = torch.tensor([[.1, .3, -.2], [.4, .2, .9]], requires_grad=True)
        padded = torch.cat([error, error.new_zeros(2, 125)], -1)
        original = profile_prediction_rate(error.square().sum(-1)/.02, 3, 'full')
        actual = profile_prediction_rate(padded.square().sum(-1)/.02, 128, 'full')
        self.assertTrue(torch.equal(original, actual))
        gradient = torch.autograd.grad(actual[0], error)[0]
        self.assertGreater(float(gradient[0].norm()), 0)
        self.assertTrue(torch.equal(gradient[1], torch.zeros_like(gradient[1])))
        rate = torch.tensor([0., .1, 100.])
        self.assertTrue(torch.equal(profile_prediction_rate(rate, 128),
                                    64*torch.log1p(2*rate/128)))


if __name__ == '__main__':
    unittest.main()
