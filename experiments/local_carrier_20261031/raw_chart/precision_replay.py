"""One full captured-shape verification of the repaired production adjoint."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from as_jepa_lewm.carrier import _contractive_solve, _precise_adjoint


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    import as_jepa_lewm.carrier as carrier
    source = json.loads(Path(args.source_record).read_text())
    manifest = json.loads(Path(source['manifest']).read_text())
    assert digest(carrier.__file__) == manifest['files']['as_jepa_lewm/carrier.py']
    assert Path(carrier.__file__).resolve() == (Path(source['source'])/'as_jepa_lewm/carrier.py').resolve()
    assert digest(__file__) == manifest['files']['experiments/local_carrier_20261031/raw_chart/precision_replay.py']
    cap = Path(args.capture_root); capture_path = cap/'capture/capture.pt'
    metadata = json.loads((cap/'capture/capture.json').read_text())
    assert digest(capture_path) == metadata['capture_sha256']
    saved = torch.load(capture_path, map_location='cpu', weights_only=False)
    values = {}
    for name in ['target', 'first', 'second', 'derivative']:
        item = saved['tensors'][name]; value = item['value'].to('cuda')
        assert list(value.shape) == item['shape'] and list(value.stride()) == item['stride']
        values[name] = value
    target, first, second, derivative = [values[k] for k in ['target', 'first', 'second', 'derivative']]
    bound, iterations, rtol = metadata['bound'], metadata['iterations'], metadata['rtol']
    assert metadata['atol'] == 0 and target.shape == (128, 4, 64, 16)
    old = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32
    rows = []
    try:
        torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
        t64, a64, b64, d64 = (x.double() for x in (target, first, second, derivative))
        matrix = bound*(b64[None]*d64.reshape(-1, 64)[:, None, :])@a64
        matrix += torch.eye(16, dtype=torch.float64, device=target.device)
        oracle = torch.linalg.solve(matrix.transpose(-1, -2), t64.reshape(-1, 16, 1)).reshape_as(t64)
        del matrix
        for mode in ['ieee_fp32_solver', 'local_fp64_production']:
            if mode == 'ieee_fp32_solver':
                torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
                def residual(x): return bound*F.linear(F.linear(x, second.T)*derivative, first.T)
                def execute(): return _contractive_solve(target, residual, bound, iterations, 0., rtol, 'IEEE diagnostic adjoint')
            else:
                torch.set_float32_matmul_precision(metadata['float32_matmul_precision'])
                torch.backends.cuda.matmul.allow_tf32 = metadata['allow_tf32']
                def execute(): return _precise_adjoint(target, first, second, derivative, bound, iterations, rtol)[0]
            policy = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32
            # One warm call then one timed call, not a throughput benchmark.
            warm = execute(); del warm; torch.cuda.synchronize()
            before = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
            result = execute(); torch.cuda.synchronize(); seconds = time.perf_counter()-start
            peak, increment = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_allocated()-before
            assert policy == (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
            norm = t64.norm(dim=-1); denominator = norm.clamp_min(torch.finfo(torch.float64).tiny)
            residual64 = result.double()+bound*F.linear(F.linear(result.double(), b64.T)*d64, a64.T)-t64
            actual_bound = residual64.norm(dim=-1)/(1-bound)
            assert bool((actual_bound <= rtol*norm).all())
            rows.append(dict(mode=mode, public_dtype=str(result.dtype), global_policy_before=list(policy),
                global_policy_unchanged=True, all_public_residual_certificates_pass=True,
                max_solution_error_over_rhs_norm=float(((result.double()-oracle).norm(dim=-1)/denominator).max()),
                max_public_residual_bound_over_rhs_norm=float((actual_bound/denominator).max()),
                wall_seconds_single_timed_call=seconds, peak_allocated_bytes=peak, peak_increment_bytes=increment,
                scope='Captured32768-slot solver only; one warm+one timed call, excludes whole model and is not a stable throughput measurement.'))
            del result
        # The same production helper must reject a deliberately unattainable
        # public-FP32 precision request; no hidden relaxation or swallowed error.
        rejected = False
        try: _precise_adjoint(target, first, second, derivative, bound, iterations, 1e-12)
        except RuntimeError as exc:
            rejected = True; failure_message = str(exc)
        assert rejected and (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32) == policy
    finally:
        torch.set_float32_matmul_precision(old[0]); torch.backends.cuda.matmul.allow_tf32 = old[1]
    report = dict(passed=True, source=source, helper_sha256=digest(__file__), capture_sha256=digest(capture_path),
        target_shape=list(target.shape), target_stride=list(target.stride()), unchanged_original_tolerance=dict(atol=0, rtol=rtol, iterations=iterations),
        rows=rows, strict_public_rounding_failure_propagates=rejected, strict_failure_message=failure_message,
        no_model_or_training=True, dynamic_readout_enabled=False,
        limitation='Only the captured adjoint operands run on GPU. Complete precision-repaired native training/memory beyond this solver remains untested.')
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle: handle.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source-record', required=True)
    parser.add_argument('--capture-root', required=True); parser.add_argument('--output', required=True)
    run(parser.parse_args())
