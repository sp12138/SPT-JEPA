"""CPU contracts for a raw public interface around the shared local chart."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from as_jepa_lewm.carrier import ResidualCarrierChart, RawConditionChartRenderer
from as_jepa_lewm.spatial_carrier import SpatialCarrierLeWM
from as_jepa_lewm.robust import RobustLeWM
from as_jepa_lewm.model import ASJepaLeWM
from vforge.models.as_jepa.renderer import AnchoredWorldDiT
from vforge.models.as_jepa.flow import flow_loss
from vforge.ops import OpsConfig, initialize_ops

B = Path('/efs/prescottsun/SPT-JEPA')
LEGACY = B/'runs/geometry_research_20261031/source-v1/as_jepa_lewm/spatial_carrier.py'
EVIDENCE = {}


def legacy_class():
    spec = importlib.util.spec_from_file_location('raw_chart_legacy9c', LEGACY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SpatialCarrierLeWM


def batch(window=3, action_dim=10):
    g = torch.Generator().manual_seed(47)
    pixels = torch.randint(0, 256, (1, window, 3, 224, 224), generator=g, dtype=torch.uint8)
    render = torch.randint(0, 256, (1, window, 3, 64, 64), generator=g, dtype=torch.uint8)
    return dict(pixels=pixels, render=render, anchor=render[:, 0], anchor224=pixels[:, 0],
                action=torch.randn(1, window, action_dim, generator=g))


def slim(interface='raw', chart='bounded'):
    torch.manual_seed(31)
    return SpatialCarrierLeWM(spatial_chart=chart, spatial_state_interface=interface,
        spatial_mean=[.2]*1024, spatial_scale=[.7]*1024, encoder_depth=1, window=3,
        renderer_width=48, renderer_depth=1, renderer_heads=3, rho=.8, forecast_fraction=.5, beta=.25)


def activate_chart(chart):
    g = torch.Generator().manual_seed(129)
    with torch.no_grad():
        chart.input.weight.copy_(torch.randn(chart.input.weight.shape, generator=g)*.4)
        chart.input.bias.copy_(torch.randn(chart.input.bias.shape, generator=g)*.1)
        chart.output.weight.copy_(torch.randn(chart.output.weight.shape, generator=g)*.4)
        chart.output.bias.copy_(torch.randn(chart.output.bias.shape, generator=g)*.1)


class RawChartContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        initialize_ops(OpsConfig(backend='sdpa_math', allow_fallback=False), device=torch.device('cpu'), parallel_state=None)

    def test_full_native_identity_loss_all_original_gradients_rng_and_keys(self):
        old_type = legacy_class()
        rows = []
        self.assertIs(SpatialCarrierLeWM.loss, RobustLeWM.loss)
        self.assertIs(ASJepaLeWM.renderer_type, AnchoredWorldDiT)
        for task in ['tworoom', 'cube']:
            cfg = json.loads((B/f'runs/local_isotropic600_20261031/{task}-spatial-local-seed1/config.json').read_text())
            torch.manual_seed(1)
            old = old_type(**cfg['model']).train()
            init_rng = torch.get_rng_state().clone()
            data = batch(cfg['model']['window'], cfg['model']['action_dim'])
            torch.manual_seed(63)
            loss, metrics = old.loss(data, step=600)
            loss.backward()
            rng = torch.get_rng_state().clone()
            old_parameters = dict(old.named_parameters())
            original = old.state_dict()
            self.assertEqual(len(original), 553)
            for mode, interface in [('none', 'chart'), ('identity', 'raw'), ('bounded', 'raw')]:
                torch.manual_seed(1)
                new = SpatialCarrierLeWM(**cfg['model'], spatial_chart=mode, spatial_state_interface=interface).train()
                torch.testing.assert_close(torch.get_rng_state(), init_rng, rtol=0, atol=0)
                for key, value in original.items():
                    torch.testing.assert_close(new.state_dict()[key], value, rtol=0, atol=0)
                extra = set(new.state_dict()) - set(original)
                expected = set() if mode == 'none' else {'carrier_chart_module.'+key for key in
                    ['input.weight', 'input.bias', 'output.weight', 'output.bias']}
                self.assertEqual(extra, expected)
                torch.manual_seed(63)
                actual, actual_metrics = new.loss(data, step=600)
                actual.backward()
                torch.testing.assert_close(actual, loss, rtol=0, atol=0)
                self.assertEqual(metrics.keys(), actual_metrics.keys())
                for key in metrics:
                    torch.testing.assert_close(actual_metrics[key], metrics[key], rtol=0, atol=0)
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                for name, parameter in new.named_parameters():
                    if name.startswith('carrier_chart_module.'):
                        if mode == 'identity': self.assertIsNone(parameter.grad)
                        else: self.assertTrue(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()))
                    else:
                        reference = old_parameters[name].grad
                        self.assertEqual(parameter.grad is None, reference is None)
                        if reference is not None:
                            torch.testing.assert_close(parameter.grad, reference, rtol=0, atol=0)
                if mode == 'bounded':
                    chart = new.carrier_chart_module
                    # Portable state roundtrip binds the restored renderer to
                    # its own chart, without duplicating named parameters.
                    state = io.BytesIO(); torch.save(new.state_dict(), state); state.seek(0)
                    restored = SpatialCarrierLeWM(**cfg['model'], spatial_chart=mode, spatial_state_interface=interface)
                    restored.load_state_dict(torch.load(state, weights_only=True), strict=True)
                    self.assertIs(restored.renderer._raw_condition_chart, restored.carrier_chart_module)
                    for key, value in new.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
                    del restored
                rows.append(dict(task=task, mode=mode, interface=interface, native_config=True,
                    original_initial_tensors_exact=553, loss_metrics_rng_exact=True,
                    all_original_gradients_exact=True, counts=new.spatial_parameter_counts()))
                del new, actual, actual_metrics
            del old, original, old_parameters, loss, metrics
        EVIDENCE['native_identity'] = rows
        EVIDENCE['legacy_spatial_sha256'] = hashlib.sha256(LEGACY.read_bytes()).hexdigest()

    def test_native_chart_gradient_after_original_network_update(self):
        model = slim().train(); data = batch()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
        norms = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.loss(data, step=600+step); loss.backward()
            gradients = [p.grad for p in model.carrier_chart_module.parameters()]
            self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients))
            norms.append(sum(float(g.square().sum()) for g in gradients)**.5)
            optimizer.step()
        self.assertGreater(norms[1], 0)
        self.assertGreater(float(model.carrier_chart_module.output.weight.abs().max()), 0)
        EVIDENCE['native_chart_gradient_entry'] = dict(first_norm=norms[0], second_norm=norms[1],
            scope='Two small CPU semantic updates, not a benchmark or GPU preflight. Exact carry/zero-renderer initialization can cancel the first chart gradient.')

    def test_legacy_nonidentity_chart_path_unchanged(self):
        kw = dict(spatial_chart='bounded', spatial_mean=[.2]*1024, spatial_scale=[.7]*1024,
            encoder_depth=1, window=3, renderer_width=48, renderer_depth=1, renderer_heads=3,
            rho=.8, forecast_fraction=.5, beta=.25)
        torch.manual_seed(31); old = legacy_class()(**kw).train()
        activate_chart(old.carrier_chart_module)
        new = slim(interface='chart').train(); new.load_state_dict(old.state_dict(), strict=True)
        data = batch()
        torch.manual_seed(8); expected, em = old.loss(data, step=600); expected.backward(); rng = torch.get_rng_state()
        torch.manual_seed(8); actual, am = new.loss(data, step=600); actual.backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key in em: torch.testing.assert_close(am[key], em[key], rtol=0, atol=0)
        for (name, p), (other, q) in zip(old.named_parameters(), new.named_parameters()):
            self.assertEqual(name, other); self.assertEqual(p.grad is None, q.grad is None)
            if p.grad is not None: torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        EVIDENCE['legacy_nonidentity_path_exact'] = True

    def test_inverse_values_all_gradients_double_oracle_and_finite_difference(self):
        chart = ResidualCarrierChart(4, 6, 'bounded', .5); activate_chart(chart)
        g = torch.Generator().manual_seed(49)
        value = torch.randn(3, 4, generator=g).requires_grad_()
        direction = torch.randn(3, 4, generator=g)
        parameters = list(chart.parameters())
        result = chart.differentiable_inverse(value)
        actual = torch.autograd.grad((result*direction).sum(), [value]+parameters)
        # Independent double-precision fixed-point tape, including the matrix
        # normalization and its parameter derivatives, no custom backward.
        inputs = [value.detach().double().requires_grad_()] + [p.detach().double().requires_grad_() for p in parameters]
        def oracle(items):
            target, w1, b1, w2, b2 = items
            guard = 1+8*torch.finfo(torch.float32).eps
            a = w1/(torch.linalg.matrix_norm(w1, ord=2)*guard).clamp_min(1)
            b = w2/(torch.linalg.matrix_norm(w2, ord=2)*guard).clamp_min(1)
            x = target
            for _ in range(80): x = target-F.linear(torch.tanh(F.linear(x, a, b1)), .5*b, b2)
            return x
        reference = oracle(inputs)
        expected = torch.autograd.grad((reference*direction.double()).sum(), inputs)
        torch.testing.assert_close(result.double(), reference, rtol=2e-6, atol=2e-6)
        max_error = 0.
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a.double(), b, rtol=2e-5, atol=2e-6)
            max_error = max(max_error, float((a.double()-b).abs().max()))
        finite = []
        for index in range(len(inputs)):
            delta = torch.randn(inputs[index].shape, generator=g, dtype=torch.float64)
            delta /= delta.norm(); epsilon = 1e-5
            plus = [item.detach().clone() for item in inputs]; minus = [item.detach().clone() for item in inputs]
            plus[index] += epsilon*delta; minus[index] -= epsilon*delta
            fd = float(((oracle(plus)-oracle(minus))*direction.double()).sum()/(2*epsilon))
            ad = float((actual[index].double()*delta).sum())
            self.assertAlmostEqual(ad, fd, delta=3e-6)
            finite.append(dict(group=index, finite_difference=fd, implicit=ad, absolute_error=abs(fd-ad)))
        residual = chart(result)-value
        error_bound = torch.linalg.vector_norm(residual, dim=-1)/.5
        self.assertTrue(bool((error_bound <= 1e-6+1e-6*value.norm(dim=-1)).all()))
        # Composition derivative must cancel the explicit inverse-parameter
        # dependence, not merely differentiate through its input.
        x = value.detach().clone().requires_grad_()
        composed = chart.differentiable_inverse(chart(x))
        composition = torch.autograd.grad(composed.sum(), [x]+parameters)
        torch.testing.assert_close(composition[0], torch.ones_like(x), rtol=2e-6, atol=2e-6)
        for gradient in composition[1:]: self.assertLess(float(gradient.abs().max()), 3e-6)
        EVIDENCE['inverse'] = dict(output_max_error=float((result.double()-reference).abs().max()),
            all_gradient_max_error=max_error, finite_differences=finite,
            measured_solution_error_bound_max=float(error_bound.max()),
            certificate=chart.norm_certificate(), composition_chart_gradient_max=max(float(a.abs().max()) for a in composition[1:]))

    def test_inverse_failure_and_higher_order_contract(self):
        chart = ResidualCarrierChart(4, 6); activate_chart(chart)
        with self.assertRaises(RuntimeError): chart.differentiable_inverse(torch.ones(3, 4), iterations=1, atol=1e-12, rtol=1e-12)
        with self.assertRaises(RuntimeError): chart.differentiable_inverse(torch.full((1, 4), float('nan')))
        x = torch.ones(1, 4, requires_grad=True)
        derivative = torch.autograd.grad(chart.differentiable_inverse(x).square().sum(), x, create_graph=True)[0]
        with self.assertRaises(RuntimeError): torch.autograd.grad(derivative.sum(), x)
        with self.assertRaises(ValueError): ResidualCarrierChart(4, 6, 'unconstrained').differentiable_inverse(x)

    def test_conjugacy_native_rho_forecast_mixture_and_renderer_all_paths(self):
        raw = slim().train(); reference = slim(interface='chart', chart='none').train()
        activate_chart(raw.carrier_chart_module)
        data = batch()
        def raw_function(emb, actions): return .65*emb+.03*emb.square()+.1*actions
        def conjugate_predict(model, emb, actions):
            if model is reference: return raw_function(emb, actions)
            state = model.carrier_chart_module.inverse(emb.unflatten(-1, (64, 16))).flatten(-2)
            return model.transform_spatial_chart(raw_function(state, actions))
        captured = []
        def conjugate_decode(renderer, latent, time, *, condition=None, **kwargs):
            captured.append((isinstance(renderer, RawConditionChartRenderer), None if condition is None else condition.detach().clone(), kwargs))
            if condition is None: return latent*.01
            if isinstance(renderer, RawConditionChartRenderer):
                condition = raw.carrier_chart_module.inverse(condition)
            return latent*.01+condition.mean(dim=(-1, -2)).reshape(-1, 1, 1, 1, 1)
        rows = []
        with patch.object(RobustLeWM, 'predict', conjugate_predict), patch.object(AnchoredWorldDiT, 'forward', conjugate_decode):
            for prior in ['one-step', 'forecast']:
                reference.rate_prior = raw.rate_prior = prior
                torch.manual_seed(61); left, lm = reference.loss(data, step=600); left_rng = torch.get_rng_state()
                torch.manual_seed(61); right, rm = raw.loss(data, step=600)
                torch.testing.assert_close(right, left, rtol=2e-6, atol=2e-6)
                torch.testing.assert_close(torch.get_rng_state(), left_rng, rtol=0, atol=0)
                for key in lm: torch.testing.assert_close(rm[key], lm[key], rtol=2e-6, atol=2e-6)
                rows.append(dict(rate_prior=prior, loss_error=float((left-right).abs()), rho=.8, forecast_fraction=.5))
            states = torch.randn(2, 1024, generator=torch.Generator().manual_seed(17))
            anchor = torch.zeros(2, 3, 64, 64)
            latent = torch.zeros(2, 48, 1, 16, 16); time = torch.ones(2, 1)
            captured.clear()
            raw.renderer(latent, time, condition=states, condition_drop=torch.ones(2, dtype=torch.bool))
            raw.renderer(latent, time, condition=states.unflatten(-1, (64, 16)))
            raw.renderer(latent, time, condition=None)
            flow_loss(raw.renderer, latent, states.unflatten(-1, (64, 16)), latent, raw.training_shift)
            raw.eval(); raw.render_states(states, anchor, steps=2, seed=2)
            expected = raw.transform_spatial_chart(states).unflatten(-1, (64, 16))
            for _, condition, _ in captured:
                if condition is not None: torch.testing.assert_close(condition, expected, rtol=0, atol=0)
            self.assertIsNone(captured[2][1]); self.assertGreaterEqual(len(captured), 6)
            self.assertIn('condition_drop', captured[0][2])
        EVIDENCE['conjugacy'] = rows
        EVIDENCE['renderer_all_paths'] = dict(calls=len(captured), flat_and_slots=True, none=True,
            direct_health=True, flow_loss=True, native_generation=True, condition_drop_forwarded=True)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RawChartContracts))
    EVIDENCE.update(passed=result.wasSuccessful(), tests=result.testsRun, device='CPU',
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_sha256={name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for name, module in
            [('carrier', __import__('as_jepa_lewm.carrier', fromlist=['__file__'])),
             ('spatial_carrier', __import__('as_jepa_lewm.spatial_carrier', fromlist=['__file__'])),
             ('model', __import__('as_jepa_lewm.model', fromlist=['__file__']))]},
        test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if os.environ.get('RAW_CHART_CONTRACT_OUTPUT'):
        Path(os.environ['RAW_CHART_CONTRACT_OUTPUT']).write_text(json.dumps(EVIDENCE, indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
