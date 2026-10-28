import math
import unittest
import torch
from as_jepa_lewm.density import (
    vp_coefficients, reverse_moments, transition_kl, discrete_gaussian_log_prob,
)


class DensityContracts(unittest.TestCase):
    def test_discrete_transition_weight_equals_exact_gaussian_kl(self):
        torch.manual_seed(38)
        target, estimate, z = [torch.randn(3, 5, dtype=torch.float64) for _ in range(3)]
        noisy = torch.tensor([[-5.], [0.], [5.]], dtype=torch.float64)
        clean = noisy + .3
        qm, qv = reverse_moments(z, target, noisy, clean)
        pm, pv = reverse_moments(z, estimate, noisy, clean)
        exact = torch.distributions.kl_divergence(
            torch.distributions.Normal(qm, qv.sqrt()),
            torch.distributions.Normal(pm, pv.sqrt()))
        torch.testing.assert_close(exact, transition_kl(target, estimate, noisy, clean), rtol=1e-10, atol=1e-10)

    def test_8bit_endpoint_probabilities_normalize_and_have_finite_gradients(self):
        values = torch.linspace(-1, 1, 256)[:, None]
        mean = torch.tensor([[-1.2, -.4, .0, .8, 1.2]], requires_grad=True)
        logp = discrete_gaussian_log_prob(values, mean, .0067)
        torch.testing.assert_close(logp.exp().sum(0), torch.ones(5), rtol=3e-5, atol=3e-5)
        loss = -logp[120].mean()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(mean.grad).all()))

    def test_gaussian_full_bound_approaches_true_entropy_with_boundaries(self):
        levels = torch.linspace(-20, 20, 65537, dtype=torch.float64)
        snr = levels.exp()
        for variance in [.03, 1., 7.]:
            middle = (.5 * (snr[1:] - snr[:-1]) * variance / (1 + snr[:-1] * variance)).sum()
            alpha, sigma = vp_coefficients(levels[0])
            prior = .5 * (alpha.square() * variance + sigma.square() - 1 - sigma.square().log())
            reconstruction = .5 * math.log(2 * math.pi) - levels[-1] / 2 + .5 * snr[-1] * variance / (1 + snr[-1] * variance)
            bound = float(prior + reconstruction + middle)
            entropy = .5 * math.log(2 * math.pi * math.e * variance)
            self.assertGreaterEqual(bound, entropy - 1e-6)
            self.assertLess(bound - entropy, .005)
            # Leaving out the reconstruction boundary does not recover entropy.
            self.assertGreater(abs(float(prior + middle) - entropy), 5)

    def test_anchor_canonicalizer_cannot_read_spatial_arrangement(self):
        from as_jepa_lewm.research import ResearchLeWM
        model=ResearchLeWM(method="anchor_canonical_color",encoder_depth=1,renderer_width=48,renderer_depth=2,renderer_heads=3)
        batch=dict(pixels=torch.randint(0,256,(1,3,3,224,224),dtype=torch.uint8),
                   render=torch.randint(0,256,(1,3,3,64,64),dtype=torch.uint8),
                   anchor=torch.randint(0,256,(1,3,64,64),dtype=torch.uint8),
                   anchor224=torch.randint(0,256,(1,3,224,224),dtype=torch.uint8))
        changed={**batch,"anchor":batch["anchor"].roll(11,-1),"anchor224":batch["anchor224"].roll(37,-1)}
        torch.testing.assert_close(model.prepare(batch)[0],model.prepare(changed)[0],rtol=0,atol=0)
        torch.testing.assert_close(model.prepare(batch)[1],model.prepare(changed)[1],rtol=0,atol=0)

    def test_reverse_sampler_shape_range_and_8bit_support(self):
        from as_jepa_lewm.research import ResearchLeWM
        from vforge.ops import initialize_ops,OpsConfig
        initialize_ops(OpsConfig(backend="sdpa_math",allow_fallback=False),device=torch.device("cpu"),parallel_state=None)
        model=ResearchLeWM(method="vdm_nats",density_steps=3,encoder_depth=1,renderer_width=48,renderer_depth=2,renderer_heads=3).eval()
        value=model.render_states(torch.randn(2,64),torch.randn(2,3,64,64),seed=17)
        self.assertEqual(tuple(value.shape),(2,3,64,64))
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertLessEqual(float(value.max()),1);self.assertGreaterEqual(float(value.min()),-1)
        levels=(value+1)*127.5
        torch.testing.assert_close(levels,levels.round(),rtol=0,atol=2e-5)

    def test_observation_unit_change_preserves_transition_and_discrete_endpoint_nats(self):
        y, estimate = torch.tensor([-.3, .7]), torch.tensor([-.2, .6])
        scale = 3.
        a = transition_kl(y, estimate, torch.tensor(-1.), torch.tensor(1.))
        offset = 2 * math.log(scale)
        b = transition_kl(y * scale, estimate * scale, torch.tensor(-1. - offset), torch.tensor(1. - offset))
        torch.testing.assert_close(a, b)
        p = discrete_gaussian_log_prob(y, estimate, .2)
        q = discrete_gaussian_log_prob(y * scale, estimate * scale, .2 * scale,
                                      bin_width=2 / 255 * scale, limits=(-scale, scale))
        torch.testing.assert_close(p, q, rtol=3e-5, atol=3e-5)


if __name__ == "__main__":
    unittest.main()
