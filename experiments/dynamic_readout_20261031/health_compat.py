"""Strict adapter for four exact-name diagnostics; native health body unchanged."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

MODEL_COMMIT = 'd6d0a9f85127a8af3e65bd31b29c161c3584867c'
SELF = 'experiments/dynamic_readout_20261031/health_compat.py'
PREFIX = 'dynamic_readout_module.'
SHAPES = {'input.weight': [64, 18], 'input.bias': [64],
          'output.weight': [64, 64], 'output.bias': [64]}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_evidence(run, fullstate, source):
    run, fullstate = Path(run), Path(fullstate)
    native = read(fullstate / 'preflight.json')
    detail = read(fullstate / 'readout-audit.json')
    dispatch = read(fullstate / 'dispatch.json')
    exit_record = read(fullstate.parent / 'exits' / (fullstate.name + '.json'))
    assert source['source_commit'] == MODEL_COMMIT and dispatch['source'] == source
    assert exit_record['returncode'] == 0 and exit_record['id'] == dispatch['id'] == fullstate.name
    assert native['run'] == str(run) and native['passed'] and detail['passed']
    assert native['checkpoint']['step'] == 600 and native['checkpoint']['checks']
    assert all(native['checkpoint']['checks'].values())
    assert detail['step'] == 600 and detail['original_initial_tensors_exact'] == 553
    assert detail['frozen_tensors_unchanged'] == 210 and detail['new_tensors'] == 4
    assert detail['new_parameters'] == 5376 and detail['head_gradient']['passed']
    assert native['all_parameter_gradients_finite'] and native['loss_finite']
    cfg = read(run / 'config.json')
    assert cfg['model']['spatial_readout_mode'] == detail['mode']
    assert detail['mode'] in {'static', 'state'}
    assert cfg['model'].get('spatial_chart', 'none') == 'none'
    hashes = dict(config=sha(run / 'config.json'), checkpoint=sha(run / 'last.ckpt'),
                  weights=sha(run / 'model-step0000600/weights.pt'),
                  initial=sha(run / 'initial-model/weights.pt'))
    assert hashes['config'] == native['config_sha256']
    assert hashes['checkpoint'] == native['checkpoint_sha256'] == detail['checkpoint_sha256']
    assert hashes['weights'] == native['weights_sha256'] == detail['weights_sha256']
    assert hashes['initial'] == native['initial_weights']['sha256'] == detail['initial_sha256']
    command = dispatch['command']
    assert command[command.index('--run') + 1] == str(run)
    assert command[command.index('--expected-step') + 1] == '600'
    reference = command[command.index('--reference-initial') + 1]
    assert sha(reference) == detail['reference_initial_sha256']
    parameters = {}
    for short, shape in SHAPES.items():
        key = PREFIX + short
        assert detail['new_parameter_shapes'][key] == shape
        gradient = detail['head_gradient']['per_tensor'][short]
        maximum = float(gradient['max_abs'])
        assert gradient['finite'] is True and math.isfinite(maximum) and maximum > 0
        assert native['gradients'][key]['passed'] and native['gradients'][key]['gradients_finite']
        parameters[key] = dict(shape=shape, actual_fp32_gradient_max=maximum)
    return dict(parameters=parameters, hashes=hashes, fullstate=str(fullstate),
                fullstate_report_sha256=sha(fullstate / 'preflight.json'),
                readout_report_sha256=sha(fullstate / 'readout-audit.json'),
                fullstate_dispatch_sha256=sha(fullstate / 'dispatch.json'),
                fullstate_exit_sha256=sha(fullstate.parent / 'exits' / (fullstate.name + '.json')))


def verified_module_expectations(model, expectations, evidence):
    """Exact keys already independently checked; keep every module rule intact."""
    named = list(model.named_parameters())
    for key, value in evidence['parameters'].items():
        matches = [parameter for name, parameter in named if name == key]
        assert len(matches) == 1 and list(matches[0].shape) == value['shape']
        assert matches[0].requires_grad and expectations[key] == 'active'
    assert expectations['dynamic_readout_module'] == 'active'
    return {key: value for key, value in expectations.items() if key not in evidence['parameters']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cases', type=int, default=16)
    parser.add_argument('--fullstate', required=True)
    parser.add_argument('--source-record', required=True)
    parser.add_argument('--helper-manifest', required=True)
    args = parser.parse_args()
    assert args.cases == 16
    helper_manifest = read(args.helper_manifest)
    assert sha(__file__) == helper_manifest['files'][SELF]
    source = read(args.source_record)
    manifest = read(source['manifest'])
    relative = 'as_jepa_tools/benchmark_health.py'
    original = Path(source['source']) / relative
    assert manifest['source_commit'] == source['source_commit'] == MODEL_COMMIT
    assert sha(original) == manifest['files'][relative]
    evidence = verify_evidence(args.run, args.fullstate, source)
    spec = importlib.util.spec_from_file_location('immutable_native_benchmark_health', original)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    original_expectations = native.gradient_expectations
    called = []
    def expectations(model, cfg):
        full = original_expectations(model, cfg)
        modules = verified_module_expectations(model, full, evidence)
        called.append(dict(original=full, native_module_rules=modules))
        return modules
    native.gradient_expectations = expectations
    sys.argv = [str(original), '--run', args.run, '--output', args.output, '--cases', str(args.cases)]
    try:
        native.main()
    finally:
        native.gradient_expectations = original_expectations
    assert len(called) == 1
    sidecar = dict(passed=True, evidence=evidence, expectations=called[0], source=source,
        helper_source_commit=helper_manifest['source_commit'], helper_sha256=sha(__file__),
        original_health_sha256=sha(original),
        scope='Four exact-name checks use checksum-matched successful native600 FP32 p.grad evidence. All module training-trace checks and the original16-case health body/noise/renderer remain unchanged. Training trace artificial zeros are preserved and are not represented as passing positive gradients. No additional backward, model training, cases or budget.')
    with (Path(args.output) / 'exact-parameter-evidence.json').open('x') as stream:
        json.dump(sidecar, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
