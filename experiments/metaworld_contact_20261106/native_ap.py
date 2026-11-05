"""Native LeWM AP comparator: diagnosis only,Native has batch regularization."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.metaworld_contact_20261106.bridge_campaign import command, TASKS
from experiments.metaworld_contact_20261106.launch import BASE, PYTHON, environment
from experiments.metaworld_contact_20261106.diagnose import save, digest


def run(a):
    plan = json.loads((a.root/'bridge-plan.json').read_text())
    seed = 1 if a.host == 'dev13' else 2
    old = json.loads((BASE/'runs/metaworld_proposal_20261104_v3/plan.json').read_text())
    parent = next(r['checkpoint'] for r in old['recipes'] if r['arm'] == 'native-control' and r['seed'] == seed)
    mode = 'ap-global' if a.gpu == 4 else 'ap-local'
    out = a.root/'native-ap'/f'{mode}-seed{seed}'; out.mkdir(parents=True, exist_ok=False)
    env = environment(a.source, a.gpu)
    common = [PYTHON, '-B', '-u', str(a.source/'experiments/metaworld_contact_20261106/evaluate.py'),
              '--data', plan['data'], '--manifest', plan['manifest'], '--parent', parent]
    preflight = out/'preflight'
    command(common+['--mode', 'cem', '--horizon', '6', '--budget', '20', '--seeds', '42', '--tasks', 'bin-picking-v3', '--output', str(preflight)],
            out/'preflight.log', env, a.source, plan['deadline_unix'])
    summary = json.loads((BASE/'runs/metaworld_improve_20261104/results/summary.json').read_text())
    original = [o for g in summary['groups'] if g['step'] == 16480 and g['arm'] == 'native-control' and g['task'] == 'bin-picking-v3'
                for o in g['observations'] if o['train_seed'] == seed and o['seed'] == 42]
    assert len(original) == 1 and digest(original[0]['trace']) == original[0]['trace_sha256']
    observed = np.load(preflight/'bin-picking-v3-seed42.npz')['actions']
    expected = np.load(original[0]['trace'])['actions'][:20]
    save(out/'parity.json', dict(passed=bool(np.array_equal(observed, expected)), max_abs=float(np.abs(observed-expected).max())))
    assert np.array_equal(observed, expected)
    command(common+['--mode', mode, '--feature-cache', str(a.root/'cache-views'/f'native-control-seed{seed}'),
                    '--seeds', '42', '43', '1042', '1043', '--tasks']+TASKS+['--output', str(out/'evaluation')],
            out/'evaluation.log', env, a.source, plan['deadline_unix'])
    save(out/'complete.json', dict(passed=True, comparator_only=True, scope='Native BN/SIGReg comparator,not a sample-level candidate.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--host', choices=['dev13', 'dev15'], required=True); p.add_argument('--gpu', type=int, choices=[4, 5], required=True)
    run(p.parse_args())
