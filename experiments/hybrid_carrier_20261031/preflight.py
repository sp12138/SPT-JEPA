"""Strict stop-gradient compatibility around the unchanged native preflight.

No trainer/model/optimizer update is implemented here. The native preflight
still owns checkpoint, RNG/cursor, initial/frozen tensors and objective backward.
Only its requires_grad-based interpretation of a frozen QueryPool is extended.
"""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from as_jepa_lewm import factory
from as_jepa_lewm.train import TrainModule
from experiments.ideas_validate_20261031 import preflight as native
from experiments.ideas_validate_20261031.carrier.calibrate import tensor_hash


def query_contract(model, cfg, checkpoint, initial):
    if cfg['model_class'] != 'hybrid_carrier' or model.hybrid_query not in {'fixed', 'adaptive'}:
        raise ValueError('This compatibility contract is exclusively for the paired hybrid gate')
    named = dict(model.named_parameters())
    query = {n: p for n, p in named.items() if n.startswith('pool.')}
    if len(query) != 9 or not all(p.requires_grad for p in query.values()):
        raise ValueError('Require the exact9 nominally differentiable original QueryPool tensors')
    configured = TrainModule.configure_optimizers(SimpleNamespace(model=model, cfg=cfg))
    optimizer = configured['optimizer']
    saved = checkpoint['optimizer_states']
    assert len(saved) == 1
    saved = saved[0]
    assert len(saved['param_groups']) == len(optimizer.param_groups)
    names = {id(p): n for n, p in named.items()}
    name_to_id = {}
    for current, stored in zip(optimizer.param_groups, saved['param_groups']):
        assert len(current['params']) == len(stored['params'])
        for parameter, state_id in zip(current['params'], stored['params']):
            name_to_id[names[id(parameter)]] = state_id
    assert len(set(name_to_id.values())) == len(name_to_id)
    rows = []
    for name, p in query.items():
        state_id = name_to_id[name]
        state = saved['state'].get(state_id)
        same = torch.equal(p.detach().cpu(), initial[name])
        gradient_finite = p.grad is not None and native.finite(p.grad)
        if model.hybrid_query == 'fixed':
            ok = same and p.grad is None and state is None
        else:
            ok = gradient_finite and state is not None and native.finite(state)
            if state is not None:
                ok = ok and all(state[k].shape == p.shape for k in ('exp_avg', 'exp_avg_sq'))
                ok = ok and float(state['step']) == checkpoint['global_step']
        rows.append(dict(name=name, optimizer_parameter_id=int(state_id), initial_equal=same,
            gradient_is_none=p.grad is None, gradient_finite=bool(gradient_finite),
            gradient_norm=None if p.grad is None else float(p.grad.detach().norm()),
            optimizer_state_present=state is not None, passed=bool(ok)))
    result = dict(arm=model.hybrid_query, tensors=9, mapping='Actual native optimizer group order to saved parameter IDs',
        nominal_requires_grad=True, effective_query_parameters=152464 if model.hybrid_query == 'adaptive' else 0,
        rows=rows, passed=all(row['passed'] for row in rows))
    del optimizer, configured
    return result


def query_numerics(model, initial, tokens):
    """Q-only repeated-input assay; no encoder call, data, labels or backward.

Use the initial Q on tokens already computed by the native audit.8 versus1024
repeats measures numerical shape/mode drift, not new data or feature quality.
The inherited native backward precision is not changed by this assay.
"""
    assert tokens.shape[0] >= 8
    pool = copy.deepcopy(model.pool).eval()
    pool.load_state_dict({n[5:]: v for n, v in initial.items() if n.startswith('pool.')}, strict=True)
    before = tensor_hash(pool.state_dict())
    actual_before = tensor_hash(model.pool.state_dict())
    gradients_before = {n: None if p.grad is None else p.grad.detach().clone() for n,p in model.pool.named_parameters()}
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(tokens.device).clone() if tokens.is_cuda else None
    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    rows, reference = [], None
    try:
        torch.set_float32_matmul_precision('high')
        base = tokens[:8].detach()
        for count, grad in ((8,False),(8,True),(1024,False),(1024,True)):
            expanded = base.repeat(count//8,1,1)
            with torch.set_grad_enabled(grad), torch.autocast(device_type=tokens.device.type, enabled=False):
                result = pool(expanded.float()).reshape(count,64)
                normalized = (result-model.hybrid_query_mean)/model.hybrid_query_scale
            value, z = result.detach(), normalized.detach()
            if reference is None:
                reference = value.clone(), z.clone()
            raw_error = value-reference[0].repeat(count//8,1)
            error = z-reference[1].repeat(count//8,1)
            assert native.finite(error) and native.finite(raw_error)
            rows.append(dict(frames=count, grad_enabled=grad,
                raw_max_abs=float(raw_error.abs().max()), raw_rms=float(raw_error.square().mean().sqrt()),
                normalized_max_abs=float(error.abs().max()), normalized_rms=float(error.square().mean().sqrt()),
                rms_over_sigma=float(error.square().mean().sqrt()/model.sigma0),
                max_abs_over_sigma=float(error.abs().max()/model.sigma0)))
            del expanded, result, normalized, value, z, raw_error, error
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = tf32
    assert tensor_hash(pool.state_dict()) == before and tensor_hash(model.pool.state_dict()) == actual_before
    for name, p in model.pool.named_parameters():
        old = gradients_before[name]
        assert (old is None and p.grad is None) or (old is not None and torch.equal(old, p.grad))
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if tokens.is_cuda: assert torch.equal(cuda_rng, torch.cuda.get_rng_state(tokens.device))
    return dict(rows=rows, query_scale=float(model.hybrid_query_scale), sigma=model.sigma0,
        encoder_calls=0, optimizer_updates=0, extra_input_source='First8 frozen tokens already computed by the native validation backward; repeat to1024.',
        pool='Initial same-seed QueryPool, isolated copy; no change to live checkpoint/gradients/RNG.',
        scope='Numerical-only; does not certify train/validation encoder precision equivalence or task information.',
        backend='Unchanged PyTorch MHA/SDPA; high FP32 matmul policy for production/calibration comparison.',
        state_grad_rng_and_global_flags_preserved=True, no_floor_or_tolerance_change=True)


def main():
    # Preserve the existing command line and all its native validation logic.
    argv = sys.argv[1:]
    root = Path(argv[argv.index('--run')+1]); out = Path(argv[argv.index('--output')+1])
    cfg = json.loads((root/'config.json').read_text())
    assert cfg['model_class'] == 'hybrid_carrier'
    required = {'hybrid_projection','hybrid_fixed_mean','hybrid_fixed_scale','hybrid_query_mean','hybrid_query_scale'}
    assert required <= set(cfg['audit_expectations']['initial_unchanged_prefixes'])
    initial = torch.load(root/'initial-model/weights.pt', map_location='cpu', weights_only=True)
    assert len(initial) == 555, 'All initial state tensors must be covered, not only named modules'
    checkpoint = torch.load(root/'last.ckpt', map_location='cpu', weights_only=False)
    old_build, old_gradient = factory.build_model, native.gradient_report
    capture, extra, handles = {}, {}, []

    def build(config):
        model = old_build(config)
        def tokens_hook(module, args):
            if 'tokens' not in capture: capture['tokens'] = args[0].detach()
        handles.append(model.pool.register_forward_pre_hook(tokens_hook))
        return model

    def gradients(model, expectations):
        assert expectations['pool'] == ('frozen' if model.hybrid_query == 'fixed' else 'active')
        rows = old_gradient(model, {k:v for k,v in expectations.items() if k != 'pool'})
        extra['query_stop_gradient'] = query_contract(model, cfg, checkpoint, initial)
        extra['query_numerics'] = query_numerics(model, initial, capture['tokens'])
        rows['pool'] = dict(expectation=expectations['pool'], interpretation='Explicit stop-gradient/AdamW-state contract',
            **extra['query_stop_gradient'])
        return rows

    try:
        factory.build_model, native.gradient_report = build, gradients
        native.main()
        native_report = json.loads((out/'preflight.json').read_text())
        assert native_report['passed'] and native_report['paired_initial_equal']['compared_tensors'] == 555
        extra['all_native_checks_passed'] = True
    finally:
        factory.build_model, native.gradient_report = old_build, old_gradient
        for handle in handles: handle.remove()
        out.mkdir(parents=True, exist_ok=True)
        extra['passed'] = bool(extra.get('all_native_checks_passed') and
            extra.get('query_stop_gradient', {}).get('passed') and extra.get('query_numerics'))
        (out/'hybrid-query-audit.json').write_text(json.dumps(extra, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
