"""Capture one unchanged native solver failure, then replay its full operands."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback


PIN = '0c9ae74840cf28247ae74341d2656e2d9ddd457c'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    with path.open('x') as handle: handle.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def pinned(record):
    import as_jepa_lewm.carrier as carrier
    source = json.loads(Path(record).read_text())
    manifest = json.loads(Path(source['manifest']).read_text())
    assert source['source_commit'] == manifest['source_commit'] == PIN
    for name in ['as_jepa_lewm/carrier.py', 'as_jepa_lewm/model.py', 'as_jepa_lewm/spatial_carrier.py',
                 'as_jepa_lewm/train.py', 'as_jepa_tools/train_stage.py']:
        assert digest(Path(source['source'])/name) == manifest['files'][name]
    assert Path(carrier.__file__).resolve() == (Path(source['source'])/'as_jepa_lewm/carrier.py').resolve()
    return carrier, source


def tensor_record(value):
    return dict(value=value.detach().cpu(), shape=list(value.shape), stride=list(value.stride()),
                storage_offset=value.storage_offset(), dtype=str(value.dtype), device=str(value.device))


def capture(args):
    import torch
    carrier, source = pinned(args.source_record)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    assert not (out/'capture.pt').exists()
    original = carrier._contractive_solve
    def observing_solver(target, residual, bound, iterations, atol, rtol, label):
        try:
            return original(target, residual, bound, iterations, atol, rtol, label)
        except RuntimeError:
            # No altered arithmetic or pre-emptive tensor copies in normal
            # execution. Keep the full batch because GEMM kernels depend on M.
            closure = inspect.getclosurevars(residual).nonlocals
            caller = inspect.currentframe().f_back.f_locals
            tensors = {'target': tensor_record(target)}
            for name in ['first', 'second', 'derivative', 'bias_one', 'bias_two']:
                if torch.is_tensor(closure.get(name)): tensors[name] = tensor_record(closure[name])
            for name in ['hidden', 'value']:
                if torch.is_tensor(caller.get(name)): tensors['caller_'+name] = tensor_record(caller[name])
            metadata = dict(label=label, bound=bound, iterations=iterations, atol=atol, rtol=rtol,
                float32_matmul_precision=torch.get_float32_matmul_precision(), allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, autocast_enabled=torch.is_autocast_enabled('cuda'),
                torch_version=torch.__version__, device_name=torch.cuda.get_device_name(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), source=source,
                helper_sha256=digest(__file__), config_path=args.config, config_sha256=digest(args.config),
                caught_at=time.time(), pid=os.getpid(), traceback=traceback.format_exc())
            assert not (out/'capture.pt').exists(), 'Capture exactly the first failure'
            torch.save(dict(tensors=tensors, metadata=metadata), out/'capture.pt')
            metadata['capture_sha256'] = digest(out/'capture.pt')
            metadata['tensor_metadata'] = {k: {a: b for a, b in v.items() if a != 'value'} for k, v in tensors.items()}
            write(out/'capture.json', metadata)
            raise
    carrier._contractive_solve = observing_solver
    sys.argv = ['as_jepa_tools.train_stage', '--config', args.config, '--stop-step', '200']
    error = None
    try:
        runpy.run_module('as_jepa_tools.train_stage', run_name='__main__')
    except BaseException as exc:
        error = exc
        raise
    finally:
        write(out/'capture-exit.json', dict(finished_at=time.time(), pid=os.getpid(),
            returncode=0 if error is None else 1, captured=(out/'capture.pt').exists(),
            exception=None if error is None else repr(error), native_source_commit=PIN,
            scope='Unmodified native Trainer/solver; wrapper records only at first RuntimeError. Cap200, no restart.'))


def full_layout(record, device, dtype):
    import torch
    value = record['value'].to(device=device, dtype=dtype)
    expected = tuple(record['stride'])
    if value.stride() != expected:
        # Original leading storage offset does not change GEMM geometry. Keep
        # the actual shape/stride, including zero-stride broadcast dimensions.
        if all(s > 0 for s in expected):
            restored = torch.empty_strided(record['shape'], expected, device=device, dtype=dtype)
            restored.copy_(value); value = restored
        else:
            span = 1+sum((n-1)*s for n, s in zip(record['shape'], expected))
            indices = torch.zeros(record['shape'], dtype=torch.int64, device=device)
            for axis, (size, stride) in enumerate(zip(record['shape'], expected)):
                shape = [1]*len(expected); shape[axis] = size
                indices += torch.arange(size, device=device).reshape(shape)*stride
            storage = torch.empty(span, device=device, dtype=dtype)
            storage.scatter_(0, indices.reshape(-1), value.reshape(-1))
            value = torch.as_strided(storage, record['shape'], expected)
    assert tuple(value.shape) == tuple(record['shape']) and value.stride() == expected
    return value


def replay(args):
    import torch
    from torch.nn import functional as F
    carrier, source = pinned(args.source_record)
    out = Path(args.output); assert (out/'capture.pt').exists() and not (out/'replay.json').exists()
    captured = torch.load(out/'capture.pt', map_location='cpu', weights_only=False)
    metadata, data = captured['metadata'], captured['tensors']
    assert metadata['label'] == 'Chart inverse adjoint'
    assert all(k in data for k in ['target', 'first', 'second', 'derivative'])
    device = torch.device('cuda'); previous = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32
    results = []
    try:
        # A dense16x16 solve supplies an independent exact-arithmetic target.
        torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
        target64, first64, second64, derivative64 = [full_layout(data[k], device, torch.float64)
            for k in ['target', 'first', 'second', 'derivative']]
        width = target64.shape[-1]; assert width == 16
        jacobian = metadata['bound']*(second64[None]*derivative64.reshape(-1, derivative64.shape[-1])[:, None, :])@first64
        jacobian += torch.eye(width, dtype=torch.float64, device=device)
        oracle = torch.linalg.solve(jacobian.transpose(-1, -2), target64.reshape(-1, width, 1)).reshape_as(target64)
        del jacobian
        target_norm64 = target64.norm(dim=-1)
        for mode in ['captured', 'ieee_fp32', 'fp64']:
            if mode == 'captured':
                torch.set_float32_matmul_precision(metadata['float32_matmul_precision'])
                torch.backends.cuda.matmul.allow_tf32 = metadata['allow_tf32']
            else:
                torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
            dtype = torch.float64 if mode == 'fp64' else torch.float32
            target, first, second, derivative = [full_layout(data[k], device, dtype)
                for k in ['target', 'first', 'second', 'derivative']]
            def residual(vector):
                return metadata['bound']*F.linear(F.linear(vector, second.T)*derivative, first.T)
            native_pass, error = True, None
            try:
                solved = carrier._contractive_solve(target, residual, metadata['bound'], metadata['iterations'],
                    metadata['atol'], metadata['rtol'], metadata['label'])
            except RuntimeError as exc:
                native_pass, error, solved = False, str(exc), None
            # Exact original update sequence, with additional read-only
            # measurements at the same4-step certificate points.
            x = target.clone(); trajectory = []
            tolerance = metadata['atol']+metadata['rtol']*target.norm(dim=-1)
            for step in range(metadata['iterations']):
                x = target-residual(x)
                if (step+1)%4 == 0 or step+1 == metadata['iterations']:
                    measured = (x+residual(x)-target).norm(dim=-1)/(1-metadata['bound'])
                    bad = measured > tolerance
                    trajectory.append(dict(iteration=step+1, violations=int(bad.sum()),
                        measured_error_bound_max=float(measured.max()), tolerance_min=float(tolerance.min())))
                    if bool(torch.isfinite(measured).all() and not bad.any()): break
            final = solved if solved is not None else x
            true_residual = final.double()+metadata['bound']*F.linear(
                F.linear(final.double(), second64.T)*derivative64, first64.T)-target64
            denom = target_norm64.clamp_min(torch.finfo(torch.float64).tiny)
            oracle_relative = (final.double()-oracle).norm(dim=-1)/denom
            residual_relative = true_residual.norm(dim=-1)/denom
            results.append(dict(mode=mode, passed=native_pass, error=error,
                float32_matmul_precision=torch.get_float32_matmul_precision(), allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                dtype=str(dtype), target_shape=list(target.shape), target_stride=list(target.stride()),
                norm_underflow_rows=int(((target.norm(dim=-1)==0)&(target_norm64>0)).sum()),
                nonzero_target_rows=int((target_norm64>0).sum()),
                oracle_relative_error_max=float(oracle_relative.max()), oracle_relative_error_mean=float(oracle_relative.mean()),
                fp64_relative_residual_max=float(residual_relative.max()), trajectory=trajectory))
    finally:
        torch.set_float32_matmul_precision(previous[0]); torch.backends.cuda.matmul.allow_tf32 = previous[1]
    write(out/'replay.json', dict(source=source, helper_sha256=digest(__file__), capture_sha256=digest(out/'capture.pt'),
        modes=results, oracle='Batched FP64 direct solve of the16x16 adjoint Jacobian for every original batch/slot; no sampling.',
        scope='Frozen full operands only; no model/trainer or optimizer. Same64-iteration cap and captured tolerance. No production precision/tolerance change.'))
    print(json.dumps(dict(modes=[dict(mode=x['mode'],passed=x['passed'],relative_error=x['oracle_relative_error_max']) for x in results])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['capture', 'replay'])
    parser.add_argument('--source-record', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--config'); args = parser.parse_args()
    if args.phase == 'capture':
        assert args.config; capture(args)
    else: replay(args)
