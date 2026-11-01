"""Completed-only comparison of a six-run revision with both matched parents."""
import argparse
from pathlib import Path

import numpy as np

from common import read, sha, write


def interval(values):
    x = np.asarray(values, dtype=float)
    ids = np.random.default_rng(20261101).integers(len(x), size=(10000, len(x)))
    return dict(mean_pp=float(x.mean()), ci95_pp=np.quantile(x[ids].mean(1), [.025, .975]).tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--parent', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, parent = Path(a.root), Path(a.parent)
    protocol = read(root / 'protocol.json')
    assert len(protocol['runs']) == 6
    rows, paired, bindings = [], [], {}
    for task in ['pusht', 'tworoom', 'cube']:
        for seed in [1, 2]:
            ident = f'{task}-field-seed{seed}'
            for arm, base, name in [('field', root, 'direct'), ('field', parent, 'indirect'), ('flat', parent, 'flat')]:
                runid = f'{task}-{arm}-seed{seed}'
                terminal = base / 'worker-records' / f'{runid}-gate.json'
                t = read(terminal)
                assert t['passed'] and len(t['results']) == 3 and all(x['returncode'] == 0 for x in t['results'])
                audit_path = base / 'audits' / f'{runid}-step600/audit.json'
                audit = read(audit_path)
                assert audit['passed'] and audit['native_cpu_model_optimizer_scheduler_restore']
                assert audit['state_before'] == audit['state_after']
                archive = base / runid / 'full-archives/step0000600.ckpt'
                assert sha(archive) == audit['full_checkpoint_sha256']
                metrics_path = base / 'evaluations' / runid / 'metrics.json'
                m = read(metrics_path)
                values = m['metrics']['episode_successes']
                assert len(values) == 50 and all(isinstance(x, bool) for x in values)
                pair = m['pairs'] if 'pairs' in m else dict(episodes=m['episodes'], start_steps=m['start_steps'])
                diagnostic = read(audit_path.with_name('state-audit.json'))
                assert diagnostic['step'] == 600 and len(diagnostic['cohort']) == 64
                if name != 'direct':
                    with np.load(root / 'audits' / f'{ident}-step600/state-audit.npz') as left, np.load(audit_path.with_name('state-audit.npz')) as right:
                        assert np.array_equal(left['observed'], right['observed'])
                        assert np.array_equal(left['actions'], right['actions'])
                rows.append(dict(task=task, seed=seed, arm=name, success_percent=2 * sum(values),
                                 success_vector=values, pairs=pair, diagnostics=diagnostic['results'],
                                 audit_sha256=sha(audit_path), metrics_sha256=sha(metrics_path)))
                bindings[str(terminal)] = sha(terminal)
        for control in ['indirect', 'flat']:
            differences = []
            for seed in [1, 2]:
                find = lambda arm: next(x for x in rows if (x['task'], x['seed'], x['arm']) == (task, seed, arm))
                direct, old = find('direct'), find(control)
                assert direct['pairs'] == old['pairs']
                delta = 100 * (np.asarray(direct['success_vector'], float) - np.asarray(old['success_vector'], float))
                differences.append(delta)
                paired.append(dict(task=task, control=control, seed=seed, **interval(delta)))
            paired.append(dict(task=task, control=control, seed='two_fixed_models', **interval(np.mean(differences, axis=0))))
    write(a.output, dict(passed=True, step=600, root=str(root), parent=str(parent), rows=rows,
          direct_minus_control=paired, terminal_bindings=bindings, observed_and_actions_exact_all6=True,
          script_sha256=sha(__file__),
          scope='All six revision gates plus matched six indirect and six flat controls;native predicates retained. '
                'Direct has98304 extra predictor parameters versus indirect. Case intervals condition on two weights '
                'and omit simulator-repeat/training-seed uncertainty. No three-task success or novelty follows from technical passes.'))
    print([(x['task'], x['arm'], x['seed'], x['success_percent']) for x in rows])
    print([x for x in paired if x['seed'] == 'two_fixed_models'])


if __name__ == '__main__':
    main()
