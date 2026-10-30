"""Merge fresh-case and metric evaluations into one noncompeting 12-GPU queue."""
import copy
import hashlib
import json
from pathlib import Path
import time

B = Path('/efs/prescottsun/SPT-JEPA')
SHARED = B/'runs/carrier_progress_20261031'
ROOT = B/'runs/carrier_diagnostics_20261031'
FRESH = B/'runs/carrier_freshcases_20261031'
CUBE_FRESH = B/'runs/carrier_cube_freshcases_20261031'
METRIC = B/'runs/carrier_metric_20261031'
CAL = B/'runs/carrier_metric_calibration_20261031'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    source = read(SHARED/'source-record.json')
    calibration_plan = read(CAL/'plan.json')
    assert len(calibration_plan['jobs']) == 4
    calibration = []
    for job in calibration_plan['jobs']:
        assert read(CAL/'exits'/(job['id']+'.json'))['returncode'] == 0
        path = Path(job['output'])/'calibration.json'
        report = read(path)
        assert report['training_clips'] == 128 and report['step'] == 600
        assert report['source']['source_commit'] == source['source_commit']
        assert digest(Path(job['output'])/'matrices.npz') == report['matrices_sha256']
        assert all(x['eigenvalue_min'] > 0 and x['effective_condition_number'] > 0
                   for x in report['fit'].values())
        calibration.append(dict(path=str(path), sha256=digest(path), fit=report['fit']))
    fresh = read(FRESH/'fresh-jobs.json')['jobs']
    metric = read(METRIC/'plan.json')['jobs']
    assert len(fresh) == 7 and len(metric) == 12
    cube_baselines = read(CUBE_FRESH/'fresh-jobs.json')['jobs']
    assert len(cube_baselines) == 3
    cube_metrics = []
    for original in metric:
        job = copy.deepcopy(original)
        job['id'] = 'fresh-' + job['id']
        job['output'] = str(CUBE_FRESH/job['id'])
        command = job['command']
        command[command.index('--manifest')+1] = str(CUBE_FRESH/'fresh-manifest.json')
        command[command.index('--output')+1] = job['output']
        job['phase'] = 'fresh-metric-planning'
        cube_metrics.append(job)
    slots = [('dev13', g) for g in range(4, 8)] + [('dev15', g) for g in range(8)]
    jobs = []
    for i, original in enumerate(fresh + metric + cube_baselines + cube_metrics):
        job = copy.deepcopy(original)
        host, gpu = slots[i % len(slots)]
        job.update(host=host, gpu=gpu, source=source,
                   environment={**job.get('environment', {}), 'MUJOCO_EGL_DEVICE_ID': str(gpu)})
        job.pop('depends_on', None)
        if 'checkpoint_weights_sha256' in job:
            assert digest(Path(job['checkpoint'])/'weights.pt') == job['checkpoint_weights_sha256']
            assert digest(Path(job['checkpoint'])/'config.json') == job['checkpoint_config_sha256']
        jobs.append(job)
    assert len({j['id'] for j in jobs}) == 34
    ROOT.mkdir()
    plan = dict(root=str(ROOT), deadline=None, environment=read(SHARED/'environment.json'), jobs=jobs)
    (ROOT/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    protocol = dict(created_at=time.time(), source_commit=source['source_commit'], jobs=34,
        fresh_tworoom_jobs=7, primary_cube_metric_jobs=12, fresh_cube_baselines=3,
        fresh_cube_metric_jobs=12, slots=slots,
        calibration_review=calibration, fresh_manifest_sha256=digest(FRESH/'fresh-manifest.json'),
        fresh_cube_manifest_sha256=digest(CUBE_FRESH/'fresh-manifest.json'),
        source_inputs={str(p):digest(p) for p in [FRESH/'fresh-jobs.json', CUBE_FRESH/'fresh-jobs.json',
                                               METRIC/'plan.json', CAL/'plan.json']},
        authoritative_exits=str(ROOT/'exits'),
        scheduling='One native dispatcher per host for these12slots; never launch the unmerged metric plan.',
        scope='No training. Fresh validation and explicit goal-metric interventions remain distinct comparisons.')
    (ROOT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps(dict(root=str(ROOT), jobs=len(jobs), slots=len(slots))))


if __name__ == '__main__':
    main()
