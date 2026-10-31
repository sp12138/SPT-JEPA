"""Observe only the first public-rounding certificate failure of pinned1060."""
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


PIN = '1060dde0e36ba4fb32a54e4a141c6e1d2cbd7a98'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def tensor_record(value):
    return dict(value=value.detach().cpu(), shape=list(value.shape), stride=list(value.stride()),
        storage_offset=value.storage_offset(), dtype=str(value.dtype), device=str(value.device))


def main(args):
    import torch
    import as_jepa_lewm.carrier as carrier
    source = json.loads(Path(args.source_record).read_text())
    manifest = json.loads(Path(source['manifest']).read_text())
    assert source['source_commit'] == manifest['source_commit'] == PIN
    for name, digest in manifest['files'].items(): assert sha(Path(source['source'])/name) == digest, name
    assert Path(carrier.__file__).resolve() == (Path(source['source'])/'as_jepa_lewm/carrier.py').resolve()
    cfg = json.loads(Path(args.config).read_text()); expected = json.loads(Path(args.reference_status).read_text())
    assert cfg['model']['spatial_state_interface'] == 'raw' and cfg['model']['spatial_chart'] == 'bounded'
    assert cfg['model'].get('spatial_readout_mode', 'none') == 'none'
    assert cfg['seed'] == 1 and cfg['batch_size'] == 128 and cfg['checkpoint_every'] == 200
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    assert not (out/'capture.pt').exists() and not (out/'capture-exit.json').exists()
    original = carrier._certify_rounded_solution

    def observing_certificate(result, target, residual, bound, atol, rtol, label):
        try:
            return original(result, target, residual, bound, atol, rtol, label)
        except RuntimeError:
            # No extra tensors, arithmetic, RNG calls or graph changes before
            # the original certificate has already failed. Retain full shapes.
            caller = inspect.currentframe().f_back.f_locals
            closure = inspect.getclosurevars(residual).nonlocals
            solved = caller['solved']; assert solved.dtype == torch.float64
            tensors = {name: tensor_record(value) for name, value in
                dict(public=result, target=target, solved64=solved).items()}
            for name, value in closure.items():
                if torch.is_tensor(value): tensors['closure_'+name] = tensor_record(value)
            public_error = (result.double()+residual(result.double())-target.double()).norm(dim=-1)/(1-bound)
            solved_error = (solved+residual(solved)-target.double()).norm(dim=-1)/(1-bound)
            tolerance = atol+rtol*target.double().norm(dim=-1)
            for name, value in dict(public_error_bound=public_error, solved_error_bound=solved_error,
                                    tolerance=tolerance, target_norm=target.double().norm(dim=-1)).items():
                tensors[name] = tensor_record(value)
            metadata = dict(label=label, bound=bound, atol=atol, rtol=rtol,
                iterations=caller.get('iterations', 64),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                autocast_enabled=torch.is_autocast_enabled('cuda'), torch_version=torch.__version__,
                device_name=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                source=source, helper_sha256=sha(__file__), config_sha256=sha(args.config),
                reference_status_sha256=sha(args.reference_status), caught_at=time.time(), pid=os.getpid(),
                traceback=traceback.format_exc(), violations=int((public_error>tolerance).sum()),
                solved_violations=int((solved_error>tolerance).sum()),
                tensor_metadata={key: {k:v for k,v in record.items() if k!='value'} for key,record in tensors.items()})
            assert not (out/'capture.pt').exists()
            torch.save(dict(tensors=tensors, metadata=metadata), out/'capture.pt')
            metadata['capture_sha256'] = sha(out/'capture.pt'); write(out/'capture.json', metadata)
            raise

    carrier._certify_rounded_solution = observing_certificate
    sys.argv = ['as_jepa_tools.train_stage', '--config', args.config, '--stop-step', '600', '--resume']
    error = None
    try:
        runpy.run_module('as_jepa_tools.train_stage', run_name='__main__')
    except BaseException as exc:
        error = exc
        raise
    finally:
        status_path = Path(cfg['output'])/'status.json'
        actual = json.loads(status_path.read_text()) if status_path.exists() else {}
        keys = ['step', 'loss', 'fm', 'rate', 'entry', 'surprise', 'feature_std']
        comparison = {key: dict(expected=expected.get(key), actual=actual.get(key),
                               exact=actual.get(key)==expected.get(key)) for key in keys}
        write(out/'capture-exit.json', dict(finished_at=time.time(), pid=os.getpid(),
            returncode=0 if error is None else 1, captured=(out/'capture.pt').exists(),
            exception=None if error is None else repr(error), original_failure_exact=all(x['exact'] for x in comparison.values()),
            comparison=comparison, native_source_commit=PIN,
            scope='Original native400 full-state resume, cap600, math/tolerances unchanged. Only first actual public-round failure is captured. No automatic retry.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source-record', required=True)
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--reference-status', required=True)
    main(parser.parse_args())
