"""All six local-query600 outcomes versus their matched original TF controls."""
import argparse
from pathlib import Path

import numpy as np

from common import B, read, sha, write


def interval(values):
    values = np.asarray(values, dtype=float)
    indices = np.random.default_rng(20261101).integers(len(values), size=(10000, len(values)))
    return dict(mean_pp=float(values.mean()), ci95_pp=np.quantile(values[indices].mean(1), [.025, .975]).tolist())


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--output', required=True)
    args = parser.parse_args()
    roots = [('original_tf', B / 'runs/unified_three_20261101'),
             ('local_query', B / 'runs/unified_quadrant600_20261101')]
    rows, bindings = [], {}
    for task in ['pusht', 'tworoom', 'cube']:
        for seed in [1, 2]:
            actions = cohort = None
            for arm, root in roots:
                ident = f'{task}-tf-seed{seed}'
                terminal = root / 'worker-records' / (ident + '-gate.json')
                done = read(terminal)
                assert done['passed'] and len(done['results']) == 3 and all(x['returncode'] == 0 for x in done['results'])
                cfg = read(root / ident / 'config.json')
                assert cfg['model_class'] == ('robust' if arm == 'original_tf' else 'quadrant_render')
                assert not cfg['model'].get('sample_forecast', False) and cfg['model'].get('flow_logit_shift', 0.) == 0.
                ap = root / 'audits' / (ident + '-step600') / 'audit.json';audit = read(ap)
                assert audit['passed'] and audit['native_cpu_model_optimizer_scheduler_restore']
                assert audit['state_before'] == audit['state_after']
                assert sha(root / ident / 'full-archives/step0000600.ckpt') == audit['full_checkpoint_sha256']
                diagnostic = read(ap.with_name('state-audit.json'))
                with np.load(ap.with_name('state-audit.npz')) as data:
                    if actions is None:actions, cohort = data['actions'].copy(), diagnostic['cohort']
                    else:assert np.array_equal(actions, data['actions']) and cohort == diagnostic['cohort']
                mp = root / 'evaluations' / ident / 'metrics.json';metrics = read(mp)
                values = metrics['metrics']['episode_successes']
                assert len(values) == 50 and all(isinstance(v, bool) for v in values)
                pairs = metrics['pairs'] if 'pairs' in metrics else dict(episodes=metrics['episodes'], start_steps=metrics['start_steps'])
                rows.append(dict(task=task, seed=seed, arm=arm, success_percent=2 * sum(values),
                                 success_vector=values, pairs=pairs, diagnostics=diagnostic['results'],
                                 audit_sha256=sha(ap), metrics_sha256=sha(mp)))
                bindings[str(terminal)] = sha(terminal)
    paired = []
    for task in ['pusht', 'tworoom', 'cube']:
        deltas = []
        for seed in [1, 2]:
            left, right = [next(x for x in rows if (x['task'], x['seed'], x['arm']) == (task, seed, arm))
                           for arm in ['local_query', 'original_tf']]
            assert left['pairs'] == right['pairs']
            delta = 100 * (np.asarray(left['success_vector'], float) - np.asarray(right['success_vector'], float))
            deltas.append(delta);paired.append(dict(task=task, seed=seed, **interval(delta)))
        paired.append(dict(task=task, seed='two_fixed_models', **interval(np.mean(deltas, axis=0))))
    write(args.output, dict(passed=True, step=600, rows=rows, query_minus_original_tf=paired,
          terminal_bindings=bindings, actions_and_cohorts_exact_all6=True, extra_renderer_parameters=1152,
          collector_sha256=sha(__file__),
          scope='Complete3 tasks×2seeds×2arms;all native cases/predicates and planner budgets retained. '
                'Global224px observations existed in both arms;local queried targets/anchors and known address differ. '
                'All550 common initial tensors and all saved2-step RNG exact;new query parameters1152. '
                'Learned latent coordinates differ,so their MSEs are not directly comparable. '
                'Renderer targets differ,so FM/energy numbers are not cross-arm likelihood improvements. '
                'Conditional case CIs omit seed-population and simulator-repeat uncertainty. '
                '600 updates are inside warmup;no convergence/novelty/three-task success follows.'))
    print([(x['task'], x['arm'], x['seed'], x['success_percent'], x['diagnostics']['observed_state_std_mean']) for x in rows])
    print([x for x in paired if x['seed'] == 'two_fixed_models'])


if __name__ == '__main__':main()
