"""Actual-model initialization, native prediction, causality and sample loss."""
import argparse
import gc
from pathlib import Path

import torch
from torch.utils.data import default_collate

from common import B, dataset, read, sha, write
from as_jepa_lewm.fixed_forecast import FixedForecastLeWM
from as_jepa_lewm.spatial_carrier import SpatialCarrierLeWM
from as_jepa_lewm.spatial_native import SpatialNativeLeWM
from vforge.ops import OpsConfig, initialize_ops


def model_kwargs(cfg, kind):
    kwargs = dict(cfg['model'])
    for key in list(kwargs):
        if key.startswith('renderer_') or key in ['beta', 'sigma0', 'rho', 'forecast_fraction', 'flow_logit_shift']:
            kwargs.pop(key)
    kwargs['predictor_kind'] = kind
    kwargs['spatial_predictor_seed'] = cfg['seed']
    return kwargs


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--output', required=True)
    args = parser.parse_args();torch.set_num_threads(2)
    initialize_ops(OpsConfig(backend='sdpa_math', allow_fallback=False), device=torch.device('cpu'), parallel_state=None)
    rows = []
    for task in ['pusht', 'tworoom', 'cube']:
        for kind in ['flat', 'spatial']:
            root = B / ('runs/unified_three_20261101' if kind == 'flat' else 'runs/unified_spatial_checkpoint600_20261101')
            ident = f'{task}-{"flat" if kind == "flat" else "field"}-seed1'
            cfg = read(root / ident / 'config.json');kwargs = model_kwargs(cfg, kind)
            torch.manual_seed(cfg['seed']);model = FixedForecastLeWM(**kwargs).eval()
            initial = torch.load(root / ident / 'initial-model/weights.pt', map_location='cpu', weights_only=True, mmap=True)
            current = model.state_dict()
            assert set(current) == {k for k in initial if not k.startswith('renderer.')}
            assert all(torch.equal(v, initial[k]) for k, v in current.items())
            weights = torch.load(root / ident / 'model-step0000600/weights.pt', map_location='cpu', weights_only=True, mmap=True)
            model.load_state_dict({k:v for k,v in weights.items() if not k.startswith('renderer.')}, strict=True)
            reference = (SpatialCarrierLeWM if kind == 'flat' else SpatialNativeLeWM)(**cfg['model']).eval()
            reference.load_state_dict(weights, strict=True)
            ds = dataset(cfg, 'train');batch = default_collate([ds[0]]);batch.pop('state', None)
            with torch.no_grad():
                px = model.normalize(batch['pixels'])
                left = model.encode(dict(pixels=px, action=batch['action']))
                right = reference.encode(dict(pixels=px, action=batch['action']))
                assert all(torch.equal(v, right[k]) for k, v in left.items())
                assert torch.equal(model.predict(left['emb'][:, :3], left['act_emb'][:, :3]),
                                   reference.predict(right['emb'][:, :3], right['act_emb'][:, :3]))
                before = torch.get_rng_state();one, _ = model.loss(batch, step=600)
                assert torch.equal(before, torch.get_rng_state())
                doubled = {k:torch.cat([v,v], 0) if torch.is_tensor(v) else v for k,v in batch.items()}
                two, _ = model.loss(doubled, step=600)
                torch.testing.assert_close(one, two, rtol=2e-5, atol=2e-6)
                calls = [];native_predict = model.predict
                def capture(emb, act):
                    calls.append((emb.clone(), act.clone()))
                    return native_predict(emb, act)
                model.predict = capture
                model.loss(batch, step=600);first = calls[:];calls.clear()
                changed = dict(batch);changed['pixels'] = batch['pixels'].clone();changed['pixels'][:, 3:] = 0
                model.loss(changed, step=600);second = calls[:]
                assert [x[0].shape[1] for x in first] == [3,4,5,6,7]
                assert len(second) == 5 and all(torch.equal(a, b) for x,y in zip(first,second) for a,b in zip(x,y))
                model.predict = native_predict
            del reference, initial, weights, left, right;gc.collect()
            model.train();loss, metrics = model.loss(batch, step=600);loss.backward()
            assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            assert all(p.grad is None for p in model.encoder.parameters())
            assert all(p.grad is None for p in model.pool.parameters())
            assert model.renderer is None
            rows.append(dict(task=task, predictor=kind, initial_nonrenderer_tensors_exact=len(current),
                             native_encode_and_predict_exact=True, future_observations_never_enter_forecast=True,
                             history_lengths=[3,4,5,6,7], sample_loss_separable=True, private_outer_RNG_exact=True,
                             actual_loss_finite=float(loss.detach()),
                             parameters=sum(p.numel() for p in model.parameters()),
                             trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                             frozen_observations=True, model_kwargs=kwargs))
            del model, loss, metrics, current, first, second, calls;gc.collect()
    assert not torch.cuda.is_initialized()
    write(args.output, dict(passed=True, CUDA_initialized=False, full_actual_models=rows,
          source_sha256=sha(Path(__file__).resolve().parents[2] / 'as_jepa_lewm/fixed_forecast.py'),
          scope='Six actual model CPU checks on one real train clip each. Same frozen local1024 carrier;'
                'train direct clean five-step forecasts from native history3,not FM or Gaussian posterior noise. '
                'No renderer,extra parameter,label or batch loss. Constructor RNG after dropping renderer differs '
                'from old FM runs;all common initial tensors are exact. Not a novel-method or task-success claim.'))
    print([(r['task'],r['predictor'],r['trainable_parameters'],r['actual_loss_finite']) for r in rows])


if __name__ == '__main__':main()
