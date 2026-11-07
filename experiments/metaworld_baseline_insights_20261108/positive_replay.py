"""CPU replay of first two hard-task positives and matched blind controls.

Selection is outcome-conditioned and diagnostic,never a performance cohort.
No learned input,proposal,NN,rendering,or new policy is introduced.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
BASE = Path('/efs/prescottsun/SPT-JEPA')
sys.path[:0] = [str(BASE / p) for p in [
    'reports/metaworld_baselines_20261107/runtime-scaling-v1',
    'reports/metaworld_improve_20261104/helper-698ae41',
    'reports/metaworld_improve_20261104/runtime-7b78f5d',
    'environments/metaworld-sanity-20261103/lib/python3.12/site-packages',
    'environments/completion-overlay', 'environments/overlay',
    'runs/mechanism_cubeaudit_20261030/source-v1/runtime',
    'sources/stable-worldmodel', 'reports/metaworld_eval_20261103/source-b574b4f']]
os.environ['MUJOCO_GL'] = 'disable'
os.environ['LD_LIBRARY_PATH'] = str(BASE / 'environments/egl-lib/usr/lib/x86_64-linux-gnu') + ':' + os.environ.get('LD_LIBRARY_PATH', '')
spec = importlib.util.spec_from_file_location('replay_helper', BASE / 'reports/metaworld_baseline_insights_20261108/helper-v3/mechanism.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def main():
    root = BASE / 'runs/metaworld_baseline_insights_20261108/positive-replay-v3'
    root.mkdir(exist_ok=False)
    bank = {(r['task'], r['seed']): r for r in helper.read(BASE / 'runs/metaworld_input_inverse_20261107_v1/followup/cases/manifest.json')['rows']}
    hard = {'bin-picking-v3', 'door-open-v3', 'drawer-open-v3', 'peg-insert-side-v3', 'pick-place-v3', 'push-v3'}
    positives = []
    for shard in [0, 1]:
        d = helper.read(BASE / f'runs/metaworld_baselines_native_20261108/training/lewm-seed2/evaluation-0058600-{shard}/metrics.json')
        assert d['passed'] and d['model_state_before'] == d['model_state_after']
        positives.extend(r for r in d['episodes'] if r['task'] in hard and r['success'])
    keys = {(r['task'], r['seed']) for r in positives}
    assert keys == {('drawer-open-v3', 6050), ('drawer-open-v3', 7047)}
    jobs = []

    def add(source, r):
        case = bank[r['task'], r['seed']]
        assert case['sha256'] == r['case_sha256']
        jobs.append(dict(source=source, case=case, trace=r['trace'], trace_sha256=r['trace_sha256'], steps=r['steps'], success=r['success']))

    for r in positives:
        add('lewm-seed2-epoch100', r)
    for shard in range(16):
        for r in helper.read(BASE / f'runs/metaworld_baselines_20261107/floors/shard{shard:02}/summary.json')['rows']:
            if (r['task'], r['seed']) in keys:
                add('floor-' + r['policy'], r)
    assert len(jobs) == 8 and len({(r['source'], r['case']['task'], r['case']['seed']) for r in jobs}) == 8
    helper.save(root / 'plan.json', dict(jobs=jobs, outcome_conditioned_selection=True))
    rows = [helper.replay(j) for j in jobs]
    d = dict(passed=True, rows=rows, episodes=8, unique_physical_cases=2, model_updates=0, neural_models_loaded=0,
             rendering_calls=0, new_policies=0, exact_all_success_sequences=True, utc=helper.now(),
             prior_failures='v1 lacked environment overlay;v2 imported unavailable EGL. Both failed before physical/NN steps. Original plans retained. v3 disables GL for CPU-only replay.',
             scope='Selected positive-case attribution,not an independent success-rate estimate. State/contact diagnostic only. No requirement that every successful task has finger contact.')
    helper.save(root / 'summary.json', d)
    print(json.dumps(d))


if __name__ == '__main__':
    main()
