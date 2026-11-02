"""Guard the distinction between teacher-forced and sampled open-loop state interventions."""
import unittest
import torch
import numpy as np

from as_jepa_tools.round2_state_check import state_conditions
from as_jepa_lewm.forecast_scored import ForecastScoredLeWM
from vforge.ops import OpsConfig, initialize_ops, reset_ops


class ReadoutTest(unittest.TestCase):
    def test_known_heldout_pose_mapping(self):
        from as_jepa_tools.latent_eval import fit_probe, targets
        rng = np.random.default_rng(17)
        train, test = rng.normal(size=(256, 8)), rng.normal(size=(64, 8))
        weight, bias = rng.normal(size=(8, 6)), rng.normal(size=(6,))
        read, _ = fit_probe(train, train @ weight + bias)
        self.assertLess(np.mean((read(test) - (test @ weight + bias)) ** 2), 1e-9)
        state = torch.tensor([[[256., 512., 128., 384., torch.pi / 2]]])
        expected = torch.tensor([[[.5, 1., .25, .75, 1., 0.]]])
        torch.testing.assert_close(targets(state), expected, atol=1e-6, rtol=0)


class ConditionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reset_ops()
        initialize_ops(OpsConfig(backend='sdpa_math', allow_fallback=False), device=torch.device('cpu'), parallel_state=None)

    def test_matched_noise_and_horizon(self):
        torch.manual_seed(19)
        model = ForecastScoredLeWM(mode='L2', beta=.25, lam=1., sigma0=.1, window=4,
                                  encoder_depth=1, encoder_size=56, renderer_width=64,
                                  renderer_depth=1, renderer_heads=1)
        # A nontrivial prior is necessary: identity initialization hides horizon mistakes.
        with torch.no_grad():
            model.predictor.transformer.output_proj.weight.normal_(0, .02)
            model.predictor.transformer.output_proj.bias.normal_(0, .01)
        model.eval()
        mu = torch.randn(2, 4, model.state_slots, model.state_width)
        noise, xi = torch.randn_like(mu), torch.randn_like(mu[:, 1:])
        actions = torch.randn(2, 4, 10)
        for horizon in ['one_step', 'rollout']:
            for mode in ['sample', 'mean']:
                model.horizon, model.forecast_noise = horizon, mode
                c = state_conditions(model, mu, actions, noise, xi)
                self.assertTrue(torch.equal(c['posterior_shuffled'] - mu.flip(0)[:, 1:],
                                            (mu.flip(0) + .1 * noise)[:, 1:] - mu.flip(0)[:, 1:]))
                prior = model.predict((mu + .1 * noise).flatten(2)[:, :-1], model.action_encoder(actions)[:, :-1]).reshape_as(mu[:, 1:])
                expected = prior + (.1 * xi if mode == 'sample' else 0.)
                self.assertTrue(torch.allclose(c['one_step'], expected, atol=1e-6))
                if horizon == 'one_step':
                    self.assertTrue(torch.allclose(c['forecast'], expected, atol=1e-6))
                else:
                    self.assertFalse(torch.allclose(c['forecast'], expected))
                self.assertEqual(c['forecast'].shape, mu[:, 1:].shape)
                self.assertFalse(torch.equal(c['one_step'], c['one_step_shuffled']))


if __name__ == '__main__':
    unittest.main()
