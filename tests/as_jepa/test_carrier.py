"""CPU semantic contracts for the conditional frozen-carrier experiment."""
import io
import math
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from as_jepa_lewm.carrier import CarrierLeWM, ResidualCarrierChart
from as_jepa_lewm.robust import RobustLeWM
from as_jepa_tools.benchmark_spatial_probe import FixedProjection
from vforge.ops import OpsConfig, initialize_ops


MODES = ["identity", "unconstrained", "bounded"]


class CarrierContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        initialize_ops(OpsConfig(backend="sdpa_math", allow_fallback=False),
                       device=torch.device("cpu"), parallel_state=None)

    def model(self, mode="bounded", baseline=False, statistics=True):
        torch.manual_seed(11)
        kwargs = dict(mode="L2", beta=.25, window=3, encoder_depth=1,
                      renderer_width=48, renderer_depth=2, renderer_heads=3,
                      rho=0., forecast_fraction=1.)
        if baseline:
            return RobustLeWM(**kwargs)
        extra = dict(carrier_mean=[.3]*64, carrier_scale=[2.]*64) if statistics else {}
        return CarrierLeWM(carrier_chart=mode, **extra, **kwargs)

    def batch(self):
        generator = torch.Generator().manual_seed(18)
        return dict(pixels=torch.randint(0, 256, (2, 3, 3, 224, 224), generator=generator, dtype=torch.uint8),
                    render=torch.randint(0, 256, (2, 3, 3, 64, 64), generator=generator, dtype=torch.uint8),
                    anchor=torch.randint(0, 256, (2, 3, 64, 64), generator=generator, dtype=torch.uint8),
                    anchor224=torch.randint(0, 256, (2, 3, 224, 224), generator=generator, dtype=torch.uint8),
                    action=torch.randn(2, 3, 10, generator=generator))

    def test_base_parameters_chart_initialization_projection_and_rng_match(self):
        baseline = self.model(baseline=True)
        baseline_rng = torch.get_rng_state().clone()
        original = baseline.state_dict()
        first = None
        numpy_before = np.random.get_state()
        for mode in MODES:
            model = self.model(mode)
            torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
            for name, value in original.items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            chart = model.carrier_chart_module.state_dict()
            if first is None:
                first = {k: v.clone() for k, v in chart.items()}
            for key, value in first.items():
                torch.testing.assert_close(chart[key], value, rtol=0, atol=0)
            expected = torch.from_numpy(FixedProjection().matrix(16*192))
            torch.testing.assert_close(model.carrier_projection, expected, rtol=0, atol=0)
            counts = model.carrier_parameter_counts()
            self.assertEqual(counts["chart"], 8320)
            self.assertEqual(counts["total"]-sum(p.numel() for p in baseline.parameters()), 8320)
            self.assertEqual(counts["chart_trainable"], 0 if mode == "identity" else 8320)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])

    def test_audited_spatial_features_and_fixed_normalization(self):
        model = self.model("identity").eval()
        pixels = model.normalize(self.batch()["pixels"])
        with torch.no_grad():
            raw = model.raw_carrier(pixels)
            tokens = model.encoder(pixels.flatten(0, 1), interpolate_pos_encoding=True).last_hidden_state
            patches = tokens[:, 1:]
            side = math.isqrt(patches.shape[1])
            grid = patches.transpose(1, 2).reshape(6, 192, side, side)
            spatial = F.adaptive_avg_pool2d(grid, (4, 4)).permute(0, 2, 3, 1).flatten(1)
            expected = (spatial @ torch.from_numpy(FixedProjection().matrix(3072))).reshape(2, 3, 64)
            torch.testing.assert_close(raw, expected, rtol=0, atol=0)
            actual = model.encode(dict(pixels=pixels))["emb"]
            torch.testing.assert_close(actual, (expected-.3)/2, rtol=0, atol=0)
            for index in range(2):
                single = model.encode(dict(pixels=pixels[index:index+1]))["emb"]
                torch.testing.assert_close(single, actual[index:index+1], rtol=1e-6, atol=2e-6)
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.pool.training)
        self.assertTrue(model.predictor.training)

    def test_exact_operator_bound_distance_bound_and_inverse_under_large_weights(self):
        torch.manual_seed(21)
        chart = ResidualCarrierChart(64, 64, "bounded", .5)
        with torch.no_grad():
            chart.input.weight.normal_(std=3)
            chart.output.weight.normal_(std=4)
            chart.input.bias.normal_()
            chart.output.bias.normal_()
        certificate = chart.norm_certificate()
        self.assertLessEqual(certificate["input_operator_norm"], 1.000001)
        self.assertLessEqual(certificate["output_operator_norm"], 1.000001)
        self.assertLessEqual(certificate["residual_lipschitz_upper"], .500001)
        self.assertTrue(certificate["global_invertibility_certified"])
        x, y = torch.randn(128,64), torch.randn(128,64)
        ratios = (chart(x)-chart(y)).norm(dim=-1)/(x-y).norm(dim=-1)
        self.assertGreaterEqual(float(ratios.detach().min()), .5-1e-6)
        self.assertLessEqual(float(ratios.detach().max()), 1.5+1e-6)
        torch.testing.assert_close(chart.inverse(chart(x)), x, rtol=2e-6, atol=2e-6)
        chart(x).square().mean().backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in chart.parameters()))

    def test_unconstrained_is_materially_different_and_has_no_inverse_promise(self):
        bounded = ResidualCarrierChart(1,1,"bounded",.5)
        free = ResidualCarrierChart(1,1,"unconstrained",.5)
        with torch.no_grad():
            bounded.input.weight.fill_(2)
            bounded.output.weight.fill_(2)
            bounded.input.bias.zero_()
        free.load_state_dict(bounded.state_dict(), strict=True)
        x = torch.tensor([[-1e-3],[1e-3]])
        ratio_free = float(((free(x)[1]-free(x)[0])/(x[1]-x[0])).detach())
        ratio_bounded = float(((bounded(x)[1]-bounded(x)[0])/(x[1]-x[0])).detach())
        self.assertGreater(ratio_free, 2.9)
        self.assertLessEqual(ratio_bounded, 1.500001)
        with self.assertRaises(ValueError):
            free.inverse(x)

    def test_autocast_keeps_chart_in_float32(self):
        chart = ResidualCarrierChart()
        x = torch.randn(5,64)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = chart(x)
        self.assertEqual(y.dtype, torch.float32)
        torch.testing.assert_close(y, x, rtol=0, atol=0)

    def test_native_loss_initial_equality_gradients_and_rng(self):
        self.assertIs(CarrierLeWM.loss, RobustLeWM.loss)
        self.assertIs(CarrierLeWM.predict, RobustLeWM.predict)
        batch = self.batch()
        reference_loss, reference_metrics, reference_rng, learned_gradients = None, None, None, None
        for mode in MODES:
            model = self.model(mode).train()
            # Expose a nonzero renderer -> condition derivative in addition to
            # the native rate signal, with identical initialization in all arms.
            with torch.no_grad():
                model.renderer.final_layer.linear.weight.fill_(.001)
            torch.manual_seed(29)
            loss, metrics = model.loss(batch, step=7)
            loss.backward()
            if reference_loss is None:
                reference_loss = loss.detach().clone()
                reference_metrics = {k:v.detach().clone() for k,v in metrics.items()}
                reference_rng = torch.get_rng_state().clone()
            torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
            self.assertEqual(set(metrics), set(reference_metrics))
            for key, value in reference_metrics.items():
                torch.testing.assert_close(metrics[key], value, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), reference_rng, rtol=0, atol=0)
            for module in [model.encoder, model.pool]:
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))
            for prefix in ["predictor", "renderer"]:
                grads = [p.grad for name,p in model.named_parameters() if name.startswith(prefix+".") and p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
                self.assertTrue(any(bool(g.abs().sum()>0) for g in grads))
            chart_grads = [p.grad for p in model.carrier_chart_module.parameters()]
            if mode == "identity":
                self.assertTrue(all(g is None for g in chart_grads))
            else:
                self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in chart_grads))
                self.assertTrue(any(bool(g.abs().sum()>0) for g in chart_grads))
                current = {k:None if p.grad is None else p.grad.clone() for k,p in model.named_parameters()}
                if learned_gradients is None:
                    learned_gradients = current
                else:
                    for key, value in learned_gradients.items():
                        self.assertEqual(value is None, current[key] is None)
                        if value is not None and not key.startswith("carrier_chart_module."):
                            torch.testing.assert_close(current[key], value, rtol=0, atol=0)
                    # Constraint normalization may change chart gradients even
                    # though zero-output initialization has identical codes.
                    norm = model.carrier_chart_module.norm_certificate()
                    if norm["input_normalization_active"]:
                        self.assertFalse(torch.equal(current["carrier_chart_module.output.weight"],
                                                     learned_gradients["carrier_chart_module.output.weight"]))

    def test_portable_state_and_optimizer_roundtrip_with_frozen_partition(self):
        model = self.model("bounded").eval()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        x = torch.randn(7,64)
        model.carrier_chart_module(x).square().mean().backward()
        optimizer.step()
        buffer = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = self.model("bounded").eval()
        restored.load_state_dict(saved["model"], strict=True)
        copied_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=.001)
        copied_optimizer.load_state_dict(saved["optimizer"])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(restored.carrier_chart_module(x), model.carrier_chart_module(x), rtol=0, atol=0)
        self.assertEqual(len(optimizer.state), len(copied_optimizer.state))
        self.assertEqual(restored.audit_expected_gradients()["encoder"], "frozen")
        self.assertEqual(restored.audit_expected_gradients()["carrier_chart_module"], "active")

    def test_calibration_required_and_invalid_statistics_refused(self):
        model = self.model(statistics=False)
        with self.assertRaises(RuntimeError):
            model.encode(dict(pixels=torch.zeros(1,1,3,224,224)))
        kwargs = dict(encoder_depth=1, renderer_width=48, renderer_depth=2, renderer_heads=3)
        for extra in [dict(carrier_mean=[0.]*64), dict(carrier_mean=[0.]*63, carrier_scale=[1.]*63),
                      dict(carrier_mean=[0.]*64, carrier_scale=[0.]*64), dict(carrier_bound=1.),
                      dict(batchnorm_state=True), dict(state_slots=2), dict(domain="color")]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                CarrierLeWM(**kwargs, **extra)


if __name__ == "__main__":
    unittest.main()
