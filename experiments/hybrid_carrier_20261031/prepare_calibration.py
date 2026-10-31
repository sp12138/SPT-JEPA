"""Prepare four calibration candidates only; no activation/launcher/training."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

B = Path('/efs/prescottsun/SPT-JEPA')
PY = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'
SELF = 'experiments/hybrid_carrier_20261031/prepare_calibration.py'


def read(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def jobs(root, src):
    result = []
    for gpu, (task, seed) in enumerate(itertools.product(('tworoom', 'cube'), (1, 2))):
        name = f'calibrate-{task}-seed{seed}'
        reference = B/'runs/ideas_validate_calibration_20261031'/name/'calibration.json'
        out = root/name
        result.append(dict(id=name, host='dev13', gpu=gpu, task=task, training_seed=seed,
            phase='initial-training-frame-calibration', source=src, output=str(out),
            command=[PY, '-B', '-u', str(Path(src['source'])/'experiments/hybrid_carrier_20261031/calibrate.py'),
                '--reference-calibration', str(reference), '--output', str(out/'calibration.json'),
                '--source-commit', src['source_commit'], '--device', 'cuda']))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-record', required=True)
    p.add_argument('--root', default=str(B/'runs/hybrid_carrier_calibration_20261031'))
    args = p.parse_args()
    root = Path(args.root)
    if root.exists(): raise FileExistsError(root)
    src = read(args.source_record)
    source = Path(src['source']); manifest = read(src['manifest'])
    assert Path(manifest['source']) == source and manifest['source_commit'] == src['source_commit']
    for name, expected in manifest['files'].items():
        assert digest(source/name) == expected, name
    assert digest(__file__) == manifest['files'][SELF], 'Run only the frozen helper'
    contract = read(source/'experiments/hybrid_carrier_20261031/cpu/contracts.json')
    assert contract['passed'] and contract['tests'] == 4
    for name in ('as_jepa_lewm/hybrid_carrier.py', 'as_jepa_lewm/factory.py',
                 'experiments/hybrid_carrier_20261031/calibrate.py'):
        assert contract['source_hashes'][name] == manifest['files'][name], name
    planned = jobs(root, src)
    for j in planned:
        ref = read(j['command'][j['command'].index('--reference-calibration')+1])
        assert ref['measured_frames'] == 128 and ref['selection_seed'] == 1042
    root.mkdir()
    for j in planned: Path(j['output']).mkdir()
    write(root/'source-record.json', src)
    environment = read(B/'runs/ideas_validate600_20261031/plan.json')['environment']
    write(root/'candidate-plan.json', dict(root=str(root), deadline=None, environment=environment, jobs=planned))
    write(root/'protocol.json', dict(source_commit=src['source_commit'], candidate_only=True,
        requires_previous48_closed_hold=True, requires_actual_free_gpus=['dev13:0','dev13:1','dev13:2','dev13:3'],
        tasks=['tworoom','cube'], seeds=[1,2], frames_per_job=128, batch_size=8,
        total_frames=512, total_encoder_calls=64, optimizer_updates=0,
        normalization='per-branch64 coordinate means plus one population global RMS',
        calibration_selection='Exact original seed1042 initial training-frame manifests; no new cohort.',
        next_phase='No two-step or600 training starts without separate review.'))
    (root/'HOLD').write_text('Candidate calibration only. Activate after source/CPU/previous-campaign-close/actual-GPU checks.\n')
    print(json.dumps(dict(root=str(root), candidate_jobs=4, active_plan=False, source_commit=src['source_commit'])))


if __name__ == '__main__': main()
