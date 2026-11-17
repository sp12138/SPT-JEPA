"""Independent Gaussian and geometry contracts for a causal covariance split."""
import unittest

import torch

from experiments.robocasa_stable_scaling.current_entropy import (
    block_profile_prediction_rate, profile_prediction_rate)


class BlockProfileContracts(unittest.TestCase):
    def test_matches_multivariate_kl_and_error_and_basis_gradients(self):
        generator = torch.Generator().manual_seed(1923)
        error = torch.randn(3, 1, 7, generator=generator, dtype=torch.float64,
                            requires_grad=True)
        raw = torch.randn(7, 3, generator=generator, dtype=torch.float64,
                          requires_grad=True)
        basis = torch.linalg.qr(raw, mode='reduced').Q
        action = torch.randn(3, 1, 7, generator=generator, dtype=torch.float64)
        action[1].zero_()
        actual = block_profile_prediction_rate(error, action, .13, basis, True)
        e = error[:, 0]
        p = (e@basis)@basis.T
        r = e-p
        length = action.square().sum((-2, -1))
        base = .13**2*torch.where(length > 0, length, torch.ones_like(length))
        unit = torch.eye(7, dtype=torch.float64)
        covariance = base[:, None, None]*unit + p[:, :, None]*p[:, None, :] + r[:, :, None]*r[:, None, :]
        q = torch.distributions.MultivariateNormal(e, base[:, None, None]*unit)
        prior = torch.distributions.MultivariateNormal(torch.zeros_like(e), covariance)
        expected = torch.distributions.kl_divergence(q, prior)
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
        ga = torch.autograd.grad(actual.sum(), (error, raw), retain_graph=True)
        gb = torch.autograd.grad(expected.sum(), (error, raw))
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-8)

    def test_other_block_cannot_hide_translation_gradient(self):
        basis = torch.eye(7, dtype=torch.float64)[:, :3]
        e = torch.tensor([[[.01, .02, -.03, 0., 0., 0., 0.]],
                          [[.01, .02, -.03, 100., -50., 20., 10.]]],
                         dtype=torch.float64, requires_grad=True)
        action = torch.ones_like(e)
        loss = block_profile_prediction_rate(e, action, .1, basis)
        grad = torch.autograd.grad(loss.sum(), e, retain_graph=True)[0]
        torch.testing.assert_close(grad[0, :, :3], grad[1, :, :3], rtol=0, atol=0)
        full = profile_prediction_rate(e.square().sum((-2, -1))/.02, 7, 'full')
        full_grad = torch.autograd.grad(full.sum(), e)[0]
        self.assertLess(float(full_grad[1, :, :3].norm()),
                        float(full_grad[0, :, :3].norm())*1e-5)

    def test_zero_input_and_one_active_block_recover_full_profile(self):
        basis = torch.eye(7, dtype=torch.float64)[:, :3]
        e = torch.tensor([[[.1, .2, -.3, 0., 0., 0., 0.]],
                          [[0., 0., 0., .4, .3, -.1, .2]]], dtype=torch.float64)
        zero = torch.zeros_like(e)
        actual = block_profile_prediction_rate(e, zero, .1, basis, True)
        expected = profile_prediction_rate(e.square().sum((-2, -1))/.02, 7, 'full')
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.equal(block_profile_prediction_rate(zero, zero, .1, basis, True),
                                    torch.zeros(2, dtype=torch.float64)))

    def test_basis_coordinates_and_global_rotation_do_not_change_energy(self):
        generator = torch.Generator().manual_seed(710)
        e = torch.randn(3, 1, 7, generator=generator, dtype=torch.float64)
        action = torch.randn_like(e)
        basis = torch.linalg.qr(torch.randn(7, 3, generator=generator, dtype=torch.float64)).Q
        rotation = torch.linalg.qr(torch.randn(7, 7, generator=generator, dtype=torch.float64)).Q
        within = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64)).Q
        value = block_profile_prediction_rate(e, action, .1, basis, True)
        torch.testing.assert_close(value, block_profile_prediction_rate(e, action, .1, basis@within, True))
        torch.testing.assert_close(value, block_profile_prediction_rate(e@rotation.T,
            action@rotation.T, .1, rotation@basis, True))

    def test_physical_sample_isolation_and_zero_coordinate_padding(self):
        e = torch.randn(3, 1, 7, requires_grad=True)
        action = torch.randn_like(e, requires_grad=True)
        basis = torch.eye(7)[:, :3]
        value = block_profile_prediction_rate(e, action, .1, basis, True)
        grad = torch.autograd.grad(value[0], e, retain_graph=True)[0]
        self.assertTrue(torch.equal(grad[1:], torch.zeros_like(grad[1:])))
        self.assertIsNone(torch.autograd.grad(value.sum(), action, allow_unused=True)[0])
        padded = torch.cat((e, e.new_zeros(3, 1, 5)), -1)
        padded_basis = torch.cat((basis, basis.new_zeros(5, 3)), 0)
        torch.testing.assert_close(value, block_profile_prediction_rate(padded, action,
            .1, padded_basis, True))

    def test_autocast_preserves_geometry_and_restriction_cannot_lower_kl(self):
        e = torch.randn(4, 1, 7)
        action = torch.randn_like(e)
        basis = torch.linalg.qr(torch.randn(7, 3)).Q
        reference = block_profile_prediction_rate(e, action, .1, basis, True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = block_profile_prediction_rate(e, action, .1, basis, True)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        base = .01*action.square().sum((-2, -1))
        full = profile_prediction_rate(e.square().sum((-2, -1))/(2*base), 7, 'full')
        self.assertTrue(bool((actual >= full-1e-5).all()))


if __name__ == '__main__':
    unittest.main()
