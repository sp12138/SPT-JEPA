"""Contracts for training-only channel interventions, not a new algorithm.

The exact-default regression reads the immutable pre-intervention source from
Git revision 421523f, or AS_JEPA_ROBUST_REFERENCE for archived source trees.
"""
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import torch

from as_jepa_lewm.robust import RobustLeWM
from vforge.models.as_jepa.codes import surprise_nats
from vforge.models.as_jepa.pixels import pack
from vforge.ops import OpsConfig, initialize_ops


class ChannelDiagnosticContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        initialize_ops(OpsConfig(backend="sdpa_math", allow_fallback=False),
                       device=torch.device("cpu"), parallel_state=None)

    def model(self, cls=RobustLeWM, **extra):
        torch.manual_seed(11)
        model = cls(mode="L2", beta=.25, window=3, encoder_depth=1,
                    renderer_width=48, renderer_depth=2, renderer_heads=3, **extra)
        torch.nn.init.normal_(model.renderer.final_layer.linear.weight, std=.01)
        return model.eval()

    def batch(self):
        torch.manual_seed(18)
        return dict(pixels=torch.randint(0, 256, (2, 3, 3, 224, 224), dtype=torch.uint8),
                    render=torch.randint(0, 256, (2, 3, 3, 64, 64), dtype=torch.uint8),
                    anchor=torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8),
                    anchor224=torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8),
                    action=torch.randn(2, 3, 10))

    def reference_class(self):
        reference = os.environ.get("AS_JEPA_ROBUST_REFERENCE")
        if reference:
            source = Path(reference).read_text()
        else:
            root = Path(__file__).resolve().parents[2]
            source = subprocess.check_output(
                ["git", "show", "421523f:as_jepa_lewm/robust.py"], cwd=root, text=True)
        namespace = {"__name__": "robust_before_channel_diagnostics"}
        exec(compile(source, "<robust_reference_421523f>", "exec"), namespace)
        return namespace["RobustLeWM"]

    def test_default_matches_reference_all_gradients_and_rng(self):
        reference = self.reference_class()
        batch = self.batch()
        for config in [{}, dict(rho=.8, forecast_fraction=.5),
                       dict(rho=.8, forecast_fraction=.5, rate_prior="forecast")]:
            with self.subTest(config=config):
                old, new = self.model(reference, **config), self.model(**config)
                old.train(); new.train()
                self.assertEqual(set(old.state_dict()), set(new.state_dict()))
                for key, value in old.state_dict().items():
                    torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
                torch.manual_seed(29)
                first, first_metrics = old.loss(batch, step=7)
                first.backward()
                old_rng = torch.get_rng_state()
                torch.manual_seed(29)
                second, second_metrics = new.loss(batch, step=7)
                second.backward()
                torch.testing.assert_close(first, second, rtol=0, atol=0)
                self.assertEqual(set(first_metrics), set(second_metrics))
                for name in first_metrics:
                    torch.testing.assert_close(first_metrics[name], second_metrics[name], rtol=0, atol=0)
                torch.testing.assert_close(old_rng, torch.get_rng_state(), rtol=0, atol=0)
                checked = 0
                for (nx, x), (ny, y) in zip(old.named_parameters(), new.named_parameters()):
                    self.assertEqual(nx, ny)
                    self.assertEqual(x.grad is None, y.grad is None)
                    if x.grad is not None:
                        checked += 1
                        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
                self.assertGreater(checked, 100)

    def test_noise_off_preserves_rate_scale_and_uses_clean_states(self):
        model = self.model(rho=.8, forecast_fraction=0., posterior_noise_scale=0.)
        batch = self.batch()
        pixels, _, _ = model.prepare(batch)
        clean = model.encode(dict(pixels=pixels, action=batch["action"]))["emb"]
        calls, conditions, targets, scales = [], [], [], []
        original_predict = model.predict
        def predict(state, action):
            calls.append(state.detach().clone())
            return original_predict(state, action)
        def flow(renderer, target, state, anchor, shift):
            conditions.append(state.detach().clone())
            return state.square().mean()
        def rate(target, predicted, sigma):
            targets.append(target.detach().clone()); scales.append(sigma)
            return surprise_nats(target, predicted, sigma)
        with patch.object(model, "predict", predict), patch("as_jepa_lewm.robust.flow_loss", flow), \
                patch("as_jepa_lewm.robust.surprise_nats", rate):
            loss, _ = model.loss(batch, step=7)
        torch.testing.assert_close(calls[0], clean[:, :-1], rtol=0, atol=0)
        torch.testing.assert_close(conditions[0].flatten(1), clean[:, 1:].flatten(0, 1), rtol=0, atol=0)
        torch.testing.assert_close(targets[0].flatten(2), clean[:, 1:], rtol=0, atol=0)
        self.assertEqual(scales, [.1])
        self.assertTrue(torch.isfinite(loss))

    def test_pure_noise_target_and_outer_rng_pairing(self):
        batch = self.batch()
        baseline = self.model(rho=.8, forecast_fraction=.5)
        forced = self.model(rho=.8, forecast_fraction=.5, flow_force_t_one=True)
        torch.manual_seed(31)
        baseline.loss(batch, step=7)
        reference_rng = torch.get_rng_state()
        draws, inputs = [], []
        randn_like = torch.randn_like
        def noise(value, *args, **kwargs):
            result = randn_like(value, *args, **kwargs)
            draws.append(result.detach().clone())
            return result
        def renderer(value, time, condition, **kwargs):
            inputs.append((value.detach().clone(), time.detach().clone()))
            return torch.zeros_like(value)
        torch.manual_seed(31)
        with patch("torch.randn_like", noise), patch.object(forced.renderer, "forward", renderer):
            _, metrics = forced.loss(batch, step=7)
        torch.testing.assert_close(reference_rng, torch.get_rng_state(), rtol=0, atol=0)
        torch.testing.assert_close(inputs[-1][0], draws[-1], rtol=0, atol=0)
        torch.testing.assert_close(inputs[-1][1], torch.ones_like(inputs[-1][1]), rtol=0, atol=0)
        _, render, _ = forced.prepare(batch)
        target = pack(render[:, 1:].flatten(0, 1))
        torch.testing.assert_close(metrics["fm"], (target-draws[-1]).square().mean(), rtol=0, atol=0)

    def test_all_cells_keep_initial_state_and_parameter_count(self):
        baseline = self.model(rho=.8, forecast_fraction=.5)
        for scale, forced in [(0., False), (1., True), (0., True)]:
            candidate = self.model(rho=.8, forecast_fraction=.5,
                                   posterior_noise_scale=scale, flow_force_t_one=forced)
            self.assertEqual(set(baseline.state_dict()), set(candidate.state_dict()))
            for key, value in baseline.state_dict().items():
                torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(sum(p.numel() for p in baseline.parameters()),
                             sum(p.numel() for p in candidate.parameters()))

    def test_cells_are_sample_level_and_have_functional_gradients(self):
        batch = self.batch()
        fixed_noise = lambda value, *a, **k: torch.full_like(value, .3)
        fixed_time = lambda count, device, shift: torch.full((count, 1), .5, device=device)
        fixed_choose = lambda *shape, **kw: torch.full(shape, .3, **kw)
        for scale, forced in [(0., False), (1., True), (0., True)]:
            with self.subTest(scale=scale, forced=forced):
                model = self.model(rho=.8, forecast_fraction=.5,
                                   posterior_noise_scale=scale, flow_force_t_one=forced)
                with patch("torch.randn_like", fixed_noise), patch("torch.rand", fixed_choose), \
                        patch("vforge.models.as_jepa.flow.draw_timesteps", fixed_time):
                    whole, _ = model.loss(batch, step=7)
                    separate = [model.loss({k: v[i:i+1] for k, v in batch.items()}, step=7)[0]
                                for i in range(2)]
                torch.testing.assert_close(whole, torch.stack(separate).mean(), rtol=5e-5, atol=5e-6)
                whole.backward()
                for module in [model.encoder, model.predictor, model.renderer]:
                    gradients = [p.grad for p in module.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
                    self.assertTrue(any(bool(g.abs().sum()>0) for g in gradients))


if __name__ == "__main__":
    unittest.main()
