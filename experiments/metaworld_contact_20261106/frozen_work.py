"""Paired target/horizon controls on frozen parents; no additional SGD."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.metaworld_contact_20261106.bridge_campaign import command, TASKS
from experiments.metaworld_contact_20261106.launch import PYTHON, environment
from experiments.metaworld_contact_20261106.diagnose import save, digest


def run(a):
    plan = json.loads((a.root/'bridge-plan.json').read_text())
    recipe = next(r for r in plan['recipes'] if r['host'] == a.host)
    variants = [('ap-global', 6), ('ap-local', 6), ('cem', 12), ('cem', 24)]
    mode, horizon = variants[a.gpu-4]
    directory = a.root/'frozen'/f'{mode}-h{horizon}-seed{recipe["seed"]}'
    directory.mkdir(parents=True, exist_ok=False)
    env = environment(a.source, a.gpu)
    script = a.source/'experiments/metaworld_contact_20261106/evaluate.py'
    common = [PYTHON, '-B', '-u', str(script), '--parent', recipe['parent'], '--data', plan['data'], '--manifest', plan['manifest']]
    preflight = directory/'preflight'
    command(common+['--mode', 'cem', '--horizon', '6', '--budget', '20', '--seeds', '42', '--tasks', 'bin-picking-v3',
                    '--output', str(preflight)], directory/'preflight.log', env, a.source, plan['deadline_unix'])
    observed = np.load(preflight/'bin-picking-v3-seed42.npz')['actions']
    previous = json.loads((a.root.parent/'metaworld_improve_20261104/results/summary.json').read_text())
    original = [o for g in previous['groups'] if g['step'] == 16480 and g['arm'] == 'tf-control' and g['task'] == 'bin-picking-v3'
                for o in g['observations'] if o['train_seed'] == recipe['seed'] and o['seed'] == 42]
    assert len(original) == 1 and digest(original[0]['trace']) == original[0]['trace_sha256']
    expected = np.load(original[0]['trace'])['actions'][:20]
    error = float(np.max(np.abs(observed-expected)))
    save(directory/'parity.json', dict(passed=bool(np.array_equal(observed, expected)), exact_actions=bool(np.array_equal(observed, expected)),
                                      max_abs=error, original_trace_sha256=original[0]['trace_sha256']))
    assert np.array_equal(observed, expected), ('New MPC control differs from archived native CEM', error)
    feature_cache = a.root/'cache-views'/f'tf-control-seed{recipe["seed"]}'
    command(common+['--mode', mode, '--horizon', str(horizon), '--feature-cache', str(feature_cache),
                    '--seeds', '42', '43', '1042', '1043', '--tasks']+TASKS+['--output', str(directory/'evaluation')],
            directory/'evaluation.log', env, a.source, plan['deadline_unix'])
    assert json.loads((directory/'evaluation/summary.json').read_text())['passed']
    save(directory/'complete.json', dict(passed=True, model_updates=0))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--host', required=True); p.add_argument('--gpu', type=int, required=True); run(p.parse_args())
