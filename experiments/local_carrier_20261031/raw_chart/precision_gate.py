"""Prepare/review one fresh native200 numerical gate; never implement training."""
import argparse
import json
from pathlib import Path

from experiments.local_carrier_20261031.prepare_basis import B, PY, read, sha, write

ROOT = B/'runs/raw_chart_precision_gate200_20261031'
REFERENCE = B/'runs/local_raw_chart600_20261031/tworoom-raw-chart-bounded-seed1'
SELF = 'experiments/local_carrier_20261031/raw_chart/precision_gate.py'
MODEL_COMMIT = '1060dde0e36ba4fb32a54e4a141c6e1d2cbd7a98'


def verify(args):
    model, helper = read(args.model_source_record), read(args.helper_source_record)
    for source in [model, helper]:
        manifest = read(source['manifest']); assert manifest['source_commit'] == source['source_commit']
        for name, digest in manifest['files'].items(): assert sha(Path(source['source'])/name) == digest, name
    assert model['source_commit'] == MODEL_COMMIT
    assert sha(__file__) == read(helper['manifest'])['files'][SELF]
    proof = read(Path(model['source'])/'experiments/local_carrier_20261031/raw_chart/precision_cpu/contracts.json')
    assert proof['passed'] and proof['tests'] == 6
    assert read(Path(model['source'])/'experiments/local_carrier_20261031/raw_chart/precision_cpu/legacy-contracts.json')['passed']
    gpu = read(B/'runs/raw_chart_precision_verify_20261031/replay.json')
    assert gpu['passed'] and gpu['source'] == model
    assert gpu['strict_public_rounding_failure_propagates']
    return model, helper


def prepare(args):
    model, helper = verify(args); assert not ROOT.exists()
    ROOT.mkdir(); run = ROOT/'tworoom-seed1'
    cfg = read(REFERENCE/'config.json'); cfg.update(output=str(run), checkpoint_every=25, retain_steps=list(range(25, 201, 25)))
    assert cfg['seed'] == 1 and cfg['batch_size'] == 128 and cfg['max_steps'] == 124910
    assert cfg['record_initial_weights'] and not cfg.get('initial_weights') and not cfg.get('initial_encoder_checkpoint')
    assert cfg['model']['spatial_state_interface'] == 'raw' and cfg['model']['spatial_chart'] == 'bounded'
    assert cfg['model'].get('spatial_readout_mode', 'none') == 'none'
    cal = cfg['spatial_calibration']; assert sha(cal['path']) == cal['sha256']
    write(run/'config.json', cfg); write(ROOT/'source-record.json', model); write(ROOT/'helper-source-record.json', helper)
    common = dict(host='dev13', gpu=0, source=model, task='tworoom', training_seed=1, family='spatial_carrier', arm='raw-local-fp64')
    jobs = [dict(**common, id='train200-tworoom-seed1', phase='train200', output=str(run), depends_on=[],
        config=str(run/'config.json'), config_sha256=sha(run/'config.json'),
        command=[PY, '-B', '-u', '-m', 'as_jepa_tools.train_stage', '--config', str(run/'config.json'), '--stop-step', '200']),
        dict(**common, id='fullstate-tworoom-seed1', phase='fullstate', output=str(ROOT/'fullstate-tworoom-seed1'),
        depends_on=['train200-tworoom-seed1'], command=[PY, '-B', '-u', str(Path(model['source'])/'experiments/ideas_validate_20261031/preflight.py'),
        '--run', str(run), '--output', str(ROOT/'fullstate-tworoom-seed1'), '--expected-step', '200',
        '--reference-initial', str(REFERENCE/'initial-model/weights.pt')])]
    write(ROOT/'candidate-plan.json', dict(root=str(ROOT), deadline=None, environment=read(REFERENCE.parent/'plan.json')['environment'], jobs=jobs))
    write(ROOT/'protocol.json', dict(model_source=model, helper_source=helper, cap=200, fresh=True, slots=[dict(host='dev13', gpu=0)],
        reference_initial=str(REFERENCE/'initial-model/weights.pt'), reference_initial_sha256=sha(REFERENCE/'initial-model/weights.pt'),
        authorization='Root approved one fresh TwoRoomseed1 nativebatch128 numerical gate only; no planning, no automatic>200 or four-run restart.',
        purpose='Cross the original112 adjoint failure under consistent localFP64 from initialization; judge actual state/gradients/numerics/cost, not task success.',
        checkpoints='Native atomic last.ckpt every25; retain portable25..200. Final200 fullstate audited; does not claim eight archived optimizer states.',
        unchanged='Original557 initial tensors, fixedcalibration/data/sampler/RNG/LR horizon/nativeoptimizer/loss/planner. Dynamic disabled. Old100 fullstate and all holds remain immutable.',
        candidate_only=True))
    assert not (ROOT/'plan.json').exists(); print(json.dumps(dict(root=str(ROOT), jobs=2, launched=False)))


def review(args):
    import torch
    from as_jepa_lewm.carrier import ResidualCarrierChart
    torch.set_num_threads(2); model, helper = verify(args)
    assert read(ROOT/'source-record.json') == model and read(ROOT/'helper-source-record.json') == helper
    for job in read(ROOT/'plan.json')['jobs']: assert read(ROOT/'exits'/(job['id']+'.json'))['returncode'] == 0
    run = ROOT/'tworoom-seed1'; report = read(ROOT/'fullstate-tworoom-seed1/preflight.json')
    assert report['passed'] and all(report['checkpoint']['checks'].values()) and report['optimizer_parameter_count']['passed']
    assert report['frozen_tensors_unchanged']['equal'] and report['frozen_tensors_unchanged']['compared_tensors'] == 210
    original = torch.load(REFERENCE/'initial-model/weights.pt', map_location='cpu', weights_only=True)
    initial = torch.load(run/'initial-model/weights.pt', map_location='cpu', weights_only=True)
    assert len(initial) == len(original) == 557 and set(initial) == set(original)
    assert all(torch.equal(value, initial[key]) for key, value in original.items())
    weights = torch.load(run/'model-step0000200/weights.pt', map_location='cpu', weights_only=True)
    chart = ResidualCarrierChart(16, 64, 'bounded', .5).enable_raw_precision()
    chart.load_state_dict({key.removeprefix('carrier_chart_module.'): value for key, value in weights.items() if key.startswith('carrier_chart_module.')}, strict=True)
    assert report['gradients']['carrier_chart_module']['gradient_norm'] > 0
    status = read(run/'status.json'); assert status['state'] == 'finished' and status['step'] == 200
    steps = sorted(int(path.name.removeprefix('model-step')) for path in run.glob('model-step*'))
    assert steps == list(range(25, 201, 25))
    rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    write(ROOT/'review.json', dict(passed=True, model_source=model, helper_source=helper, original_initial_tensors_equal=557,
        frozen=report['frozen_tensors_unchanged'], checkpoint=report['checkpoint'], optimizer=report['optimizer_parameter_count'],
        gradients=report['gradients'], chart_certificate=chart.norm_certificate(), native_metrics=rows, final_status=status,
        portable_steps=steps, fullstate_report_sha256=sha(ROOT/'fullstate-tworoom-seed1/preflight.json'),
        scope='One fresh0->200 numerical gate only; no planning/task-performance claim or automatic continuation.'))
    print(json.dumps(dict(passed=True, step=200, peak_gib=status['peak_gib'], chart_gradient=report['gradients']['carrier_chart_module']['gradient_norm'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'review'])
    p.add_argument('--model-source-record', required=True); p.add_argument('--helper-source-record', required=True)
    a = p.parse_args(); prepare(a) if a.phase == 'prepare' else review(a)
