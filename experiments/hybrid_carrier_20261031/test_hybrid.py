"""CPU semantic contracts; no production training or GPU claim."""
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from as_jepa_lewm.hybrid_carrier import HybridCarrierLeWM
from as_jepa_lewm.carrier import CarrierLeWM
from as_jepa_lewm.robust import RobustLeWM
from as_jepa_lewm.factory import build_model, model_target, MODEL_TARGETS
from as_jepa_lewm.train import TrainModule
from experiments.hybrid_carrier_20261031.calibrate import statistics
from experiments.ideas_validate_20261031.carrier.calibrate import tensor_hash
from vforge.ops import OpsConfig, initialize_ops

B = Path('/efs/prescottsun/SPT-JEPA')
EVIDENCE = {}


def original_config(task, seed):
    ref = json.loads((B/f'runs/ideas_validate_calibration_20261031/calibrate-{task}-seed{seed}/calibration.json').read_text())
    return json.loads(Path(ref['config']).read_text())


def kwargs(cfg, mode='fixed', scales=(.3, .001)):
    result = dict(cfg['model'])
    result.update(state_slots=8, state_width=16, hybrid_query=mode,
        hybrid_fixed_mean=[.2]*64, hybrid_fixed_scale=scales[0],
        hybrid_query_mean=[.1]*64, hybrid_query_scale=scales[1])
    return result


def batch(window=8, action_dim=10):
    g = torch.Generator().manual_seed(47)
    pixels = torch.randint(0, 256, (1, window, 3, 224, 224), generator=g, dtype=torch.uint8)
    render = torch.randint(0, 256, (1, window, 3, 64, 64), generator=g, dtype=torch.uint8)
    return dict(pixels=pixels, render=render, anchor=render[:, 0], anchor224=pixels[:, 0],
                action=torch.randn(1, window, action_dim, generator=g))


class HybridContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        initialize_ops(OpsConfig(backend='sdpa_math', allow_fallback=False), device=torch.device('cpu'), parallel_state=None)

    def test_initial_full_states_original64_encoder_pool_and_outer_rng(self):
        rows = []
        self.assertIs(HybridCarrierLeWM.loss, RobustLeWM.loss)
        self.assertIs(HybridCarrierLeWM.predict, RobustLeWM.predict)
        for task in ('tworoom', 'cube'):
            for seed in (1, 2):
                cfg = original_config(task, seed)
                torch.manual_seed(seed)
                original = RobustLeWM(**cfg['model'])
                encoder_hash = tensor_hash(original.encoder.state_dict())
                pool_hash = tensor_hash(original.pool.state_dict())
                del original
                torch.manual_seed(seed)
                base128 = RobustLeWM(**dict(cfg['model'], state_slots=8, state_width=16))
                native_rng = torch.get_rng_state().clone()
                original128 = {k: v.clone() for k, v in base128.state_dict().items() if not k.startswith('pool.')}
                del base128
                torch.manual_seed(seed)
                before_numpy = np.random.get_state()
                fixed = HybridCarrierLeWM(**kwargs(cfg))
                torch.testing.assert_close(torch.get_rng_state(), native_rng, rtol=0, atol=0)
                np.testing.assert_array_equal(before_numpy[1], np.random.get_state()[1])
                self.assertEqual(before_numpy[2:], np.random.get_state()[2:])
                self.assertEqual(tensor_hash(fixed.encoder.state_dict()), encoder_hash)
                self.assertEqual(tensor_hash(fixed.pool.state_dict()), pool_hash)
                for name, value in original128.items():
                    torch.testing.assert_close(fixed.state_dict()[name], value, rtol=0, atol=0)
                torch.manual_seed(seed)
                adaptive = HybridCarrierLeWM(**kwargs(cfg, 'adaptive'))
                torch.testing.assert_close(torch.get_rng_state(), native_rng, rtol=0, atol=0)
                self.assertEqual(fixed.state_dict().keys(), adaptive.state_dict().keys())
                for name, value in fixed.state_dict().items():
                    torch.testing.assert_close(adaptive.state_dict()[name], value, rtol=0, atol=0)
                for name, p in adaptive.named_parameters():
                    q = dict(fixed.named_parameters())[name]
                    self.assertEqual(p.requires_grad, q.requires_grad)
                self.assertEqual(adaptive.pool.attn.dropout, 0.)
                self.assertFalse(any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in adaptive.pool.modules()))
                counts = fixed.hybrid_parameter_counts(), adaptive.hybrid_parameter_counts()
                self.assertEqual(counts[0]['total'], counts[1]['total'])
                self.assertEqual(counts[1]['trainable'], counts[0]['trainable'])
                self.assertEqual(counts[1]['effective_trainable']-counts[0]['effective_trainable'], counts[1]['query_pool'])
                self.assertEqual(counts[1]['query_pool'], 152464)
                rows.append(dict(task=task, seed=seed, tensors=len(fixed.state_dict()),
                    original128_nonpool_tensors=len(original128), all_initial_tensors_rng_equal=True,
                    original64_encoder_pool_exact=True, counts=dict(fixed=counts[0], adaptive=counts[1])))
                del fixed, adaptive, original128
                gc.collect()
        EVIDENCE['initialization'] = rows

    def test_full_native_loss_metrics_rng_shared_grads_and_protected_branch(self):
        rows = []
        for task in ('tworoom', 'cube'):
            cfg = original_config(task, 1)
            kw = kwargs(cfg)
            torch.manual_seed(1)
            fixed = HybridCarrierLeWM(**kw)
            torch.manual_seed(1)
            adaptive = HybridCarrierLeWM(**dict(kw, hybrid_query='adaptive'))
            data = batch(cfg['model']['window'], cfg['model']['action_dim'])
            pixels = fixed.prepare(data)[0]
            # The synthetic branch constants are deliberately small; they are
            # NOT production calibration or empirical health on real episodes.
            for training in (False, True):
                fixed.train(training); adaptive.train(training)
                for grad in (False, True):
                    with torch.set_grad_enabled(grad):
                        a = fixed.encode(dict(pixels=pixels))['emb']
                        b = adaptive.encode(dict(pixels=pixels))['emb']
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    del a, b
            fixed.train(); adaptive.train()
            torch.manual_seed(67)
            loss, metrics = fixed.loss(data, step=0)
            loss.backward()
            end_rng = torch.get_rng_state().clone()
            torch.manual_seed(67)
            other, other_metrics = adaptive.loss(data, step=0)
            other.backward()
            torch.testing.assert_close(other, loss, rtol=0, atol=0)
            for name in metrics:
                torch.testing.assert_close(other_metrics[name], metrics[name], rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), end_rng, rtol=0, atol=0)
            shared_gradients = 0
            pool_norms = {}
            for name, p in adaptive.named_parameters():
                q = dict(fixed.named_parameters())[name]
                if name.startswith('pool.'):
                    self.assertIsNone(q.grad)
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(bool(torch.isfinite(p.grad).all()))
                    pool_norms[name] = float(p.grad.norm())
                else:
                    self.assertEqual(p.grad is None, q.grad is None)
                    if p.grad is not None:
                        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
                        shared_gradients += 1
            self.assertGreater(sum(pool_norms.values()), 0.)
            before = {n: v.clone() for n, v in adaptive.state_dict().items()}
            with torch.no_grad():
                old_emb = adaptive.encode(dict(pixels=pixels))['emb']
            # Two actual native AdamW/scheduler steps on one synthetic CPU batch.
            # Reuse the native factory; no copied optimizer or Trainer.
            optimizer_records = []
            for model in (fixed, adaptive):
                native = TrainModule.configure_optimizers(SimpleNamespace(model=model, cfg=cfg))
                optimizer = native['optimizer']; scheduler = native['lr_scheduler']['scheduler']
                pool_before = {n: p.detach().clone() for n,p in model.pool.named_parameters()}
                rates = []
                for step in range(2):
                    if step:
                        optimizer.zero_grad(set_to_none=True)
                        torch.manual_seed(68)
                        next_loss, _ = model.loss(data, step=step)
                        self.assertTrue(bool(torch.isfinite(next_loss)))
                        next_loss.backward()
                    self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
                    rates.append([group['lr'] for group in optimizer.param_groups])
                    optimizer.step(); scheduler.step()
                    if model.hybrid_query == 'fixed':
                        for n,p in model.pool.named_parameters():
                            torch.testing.assert_close(p, pool_before[n], rtol=0, atol=0)
                            self.assertIsNone(p.grad)
                            self.assertNotIn(p, optimizer.state)
                state_count = sum(p in optimizer.state for p in model.pool.parameters())
                self.assertEqual(state_count, 0 if model.hybrid_query == 'fixed' else len(pool_before))
                optimizer_records.append(dict(arm=model.hybrid_query, native_AdamW_steps=2,
                    optimizer_pool_states=state_count, learning_rates=rates,
                    fixed_pool_tensors_exact=model.hybrid_query == 'fixed'))
                del optimizer, scheduler, native
            with torch.no_grad():
                new_emb = adaptive.encode(dict(pixels=pixels))['emb']
            torch.testing.assert_close(new_emb[..., :64], old_emb[..., :64], rtol=0, atol=0)
            self.assertGreater(float((new_emb[..., 64:]-old_emb[..., 64:]).abs().max()), 0.)
            for name, value in before.items():
                if name.startswith(('encoder.', 'hybrid_')):
                    torch.testing.assert_close(adaptive.state_dict()[name], value, rtol=0, atol=0)
            # Changed future observations/actions cannot change an initial code.
            changed = pixels.clone(); changed[:, 1:] = torch.flip(changed[:, 1:], [-1])
            with torch.no_grad():
                new_input = adaptive.encode(dict(pixels=changed, action=data['action']+100))['emb']
            torch.testing.assert_close(new_input[:, 0], new_emb[:, 0], rtol=0, atol=0)
            self.assertEqual(tuple(new_emb.shape), (1, cfg['model']['window'], 128))
            stream = io.BytesIO(); torch.save(adaptive.state_dict(), stream); stream.seek(0)
            restored = HybridCarrierLeWM(**dict(kw, hybrid_query='adaptive'))
            restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
            with torch.no_grad():
                torch.testing.assert_close(restored.encode(dict(pixels=pixels))['emb'], new_emb, rtol=0, atol=0)
            # Ordinary same-seed F64 raw measurement, before either normalization.
            torch.manual_seed(1)
            legacy = CarrierLeWM(**dict(cfg['model'], carrier_chart='identity'))
            f, _ = adaptive.raw_channels(pixels)
            torch.testing.assert_close(f, legacy.raw_carrier(pixels), rtol=0, atol=0)
            rows.append(dict(task=task, batch=1, window=cfg['model']['window'],
                synthetic_query_scale=.001, loss=float(loss), metrics={n: float(v) for n,v in metrics.items()},
                initial_loss_metrics_rng_exact=True, shared_gradient_tensors_exact=shared_gradients,
                pool_gradient_norms=pool_norms, all_pool_gradients_finite=True,
                native_optimizer_contract=optimizer_records,
                fixed_channel_after_query_update_exact=True, original_F64_raw_exact=True,
                frame_local_no_future_or_action_leak=True, checkpoint_roundtrip_exact=True,
                semantics='Synthetic CPU numerical contract; actual calibrated batch128 GPU health remains unvalidated.'))
            del fixed, adaptive, restored, legacy, before, loss, other
            gc.collect()
        EVIDENCE['native_loss_and_protection'] = rows

    def test_calibration_is_scalar_sample_independent_and_fail_closed(self):
        g = torch.Generator().manual_seed(16)
        v = torch.randn(128,64, generator=g)*1e-5 + .01
        m, s, std = statistics(v)
        self.assertEqual(s.ndim, 0)
        torch.testing.assert_close(s.double(), std.square().mean().sqrt(), rtol=1e-7, atol=0)
        self.assertAlmostEqual(float(((v-m)/s).square().mean()), 1., places=5)
        for bad in (torch.zeros(128,64), torch.full((128,64), float('nan')), torch.zeros(127,64)):
            with self.assertRaises(ValueError): statistics(bad)
        with self.assertRaises(ValueError): HybridCarrierLeWM(hybrid_fixed_scale=[1.]*64)
        EVIDENCE['scalar_calibration'] = dict(raw_small_rms=float(s), no_floor=True, constant_nonfinite_wrong_count_fail=True)

    def test_factory_old_dispatch_functions_unchanged(self):
        for key, expected in [('base','model.ASJepaLeWM'),('robust','robust.RobustLeWM'),
                              ('carrier','carrier.CarrierLeWM'),('spatial_carrier','spatial_carrier.SpatialCarrierLeWM')]:
            self.assertEqual(model_target(dict(model_class=key)), 'as_jepa_lewm.'+expected)
        self.assertEqual(model_target({}), MODEL_TARGETS['base'])
        self.assertEqual(model_target(dict(robust_variant=True)), MODEL_TARGETS['robust'])
        self.assertEqual(model_target(dict(research_variant=True, robust_variant=True)), MODEL_TARGETS['robust'])
        self.assertEqual(model_target(dict(frontier_variant=True, robust_variant=True)), MODEL_TARGETS['frontier'])
        with patch('as_jepa_lewm.factory.import_module') as importer:
            build_model(dict(model_class='hybrid_carrier',model={'sentinel':3}))
            importer.assert_called_once_with('as_jepa_lewm.hybrid_carrier')
            importer.return_value.HybridCarrierLeWM.assert_called_once_with(sentinel=3)
        EVIDENCE['factory'] = dict(legacy_dispatch_preserved=True, new_route='hybrid_carrier')


if __name__ == '__main__':
    program = unittest.main(exit=False)
    result = dict(passed=program.result.wasSuccessful(), tests=program.result.testsRun,
        evidence=EVIDENCE, torch=torch.__version__, device='cpu', source_hashes={})
    for name in ('as_jepa_lewm/hybrid_carrier.py','as_jepa_lewm/factory.py',
                 'experiments/hybrid_carrier_20261031/calibrate.py',__file__):
        result['source_hashes'][str(name)] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    destination = Path(os.environ.get('HYBRID_CONTRACT_OUTPUT', '/tmp/hybrid-contracts.json'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    raise SystemExit(0 if result['passed'] else 1)
