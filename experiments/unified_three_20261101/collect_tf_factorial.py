"""Completed-only TF prior-law x training-noise factorial on all three tasks."""
import argparse
from pathlib import Path

import numpy as np

from common import B, read, sha, write


def interval(values):
    values = np.asarray(values, dtype=float)
    indices = np.random.default_rng(20261101).integers(len(values), size=(10000, len(values)))
    return dict(mean_pp=float(values.mean()), ci95_pp=np.quantile(values[indices].mean(1), [.025, .975]).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--sampling-only', action='store_true')
    args = parser.parse_args()
    cells = [(False, 0., 'unified_three_20261101'), (True, 0., 'unified_prior_sample600_20261101')]
    if not args.sampling_only:
        cells += [(False, .8, 'unified_tf_highnoise600_20261101'), (True, .8, 'unified_prior_highnoise600_20261101')]
    rows, bindings = [], {}
    for task in ['pusht', 'tworoom', 'cube']:
        for seed in [1, 2]:
            actions = cohort = None
            for sampled, shift, name in cells:
                root = B / 'runs' / name
                ident = f'{task}-tf-seed{seed}'
                terminal = root / 'worker-records' / (ident + '-gate.json')
                done = read(terminal)
                assert done['passed'] and len(done['results']) == 3
                assert all(r['returncode'] == 0 for r in done['results'])
                cfg = read(root / ident / 'config.json')
                assert cfg['model_class'] == 'robust'
                assert cfg['model'].get('sample_forecast', False) == sampled
                assert cfg['model'].get('flow_logit_shift', 0.) == shift
                ap = root / 'audits' / (ident + '-step600') / 'audit.json'
                audit = read(ap)
                assert audit['passed'] and audit['native_cpu_model_optimizer_scheduler_restore']
                assert audit['state_before'] == audit['state_after']
                assert sha(root / ident / 'full-archives/step0000600.ckpt') == audit['full_checkpoint_sha256']
                diagnostics = read(ap.with_name('state-audit.json'))
                with np.load(ap.with_name('state-audit.npz')) as data:
                    if actions is None:
                        actions, cohort = data['actions'].copy(), diagnostics['cohort']
                    else:
                        assert np.array_equal(actions, data['actions']) and cohort == diagnostics['cohort']
                mp = root / 'evaluations' / ident / 'metrics.json'
                metrics = read(mp); values = metrics['metrics']['episode_successes']
                assert len(values) == 50 and all(isinstance(v, bool) for v in values)
                pairs = metrics['pairs'] if 'pairs' in metrics else dict(episodes=metrics['episodes'], start_steps=metrics['start_steps'])
                rows.append(dict(task=task, seed=seed, sampled=sampled, noise_logit_shift=shift,
                                 success_percent=2 * sum(values), success_vector=values, pairs=pairs,
                                 diagnostics=diagnostics['results'], audit_sha256=sha(ap), metrics_sha256=sha(mp)))
                bindings[str(terminal)] = sha(terminal)
    contrasts = [('sampling_ordinary', [(True, 0., 1), (False, 0., -1)])]
    if not args.sampling_only:
        contrasts += [('sampling_highnoise', [(True, .8, 1), (False, .8, -1)]),
                      ('highnoise_mean', [(False, .8, 1), (False, 0., -1)]),
                      ('highnoise_sampled', [(True, .8, 1), (True, 0., -1)]),
                      ('interaction', [(True, .8, 1), (True, 0., -1), (False, .8, -1), (False, 0., 1)])]
    paired = []
    for task in ['pusht', 'tworoom', 'cube']:
        for name, terms in contrasts:
            deltas = []
            for seed in [1, 2]:
                selected = [(next(x for x in rows if (x['task'], x['seed'], x['sampled'], x['noise_logit_shift']) ==
                                  (task, seed, sampled, shift)), sign) for sampled, shift, sign in terms]
                assert all(x['pairs'] == selected[0][0]['pairs'] for x, sign in selected)
                delta = sum(sign * 100 * np.asarray(x['success_vector'], float) for x, sign in selected)
                deltas.append(delta)
                paired.append(dict(task=task, contrast=name, seed=seed, **interval(delta)))
            paired.append(dict(task=task, contrast=name, seed='two_fixed_models', **interval(np.mean(deltas, axis=0))))
    write(args.output, dict(passed=True, step=600, rows=rows, paired=paired, terminal_bindings=bindings,
          all_fixed_actions_and_cohorts_exact=True, collector_sha256=sha(__file__),
          scope='All requested task/seed/cell endpoints complete;fixed native evaluation/data/batch/schedule. '
                'All550 initial tensors and saved2-step RNG match;no new parameter/label. Learned coordinate '
                'systems differ,so latent MSEs are not comparable across models. Flow target times differ in '
                'high-noise cells,so their training FM losses are not comparable. Conditional case CIs omit '
                'seed-population/simulator-repeat uncertainty. Six hundred updates do not establish convergence.'))
    print([(x['task'], x['sampled'], x['noise_logit_shift'], x['seed'], x['success_percent']) for x in rows])
    print([x for x in paired if x['seed'] == 'two_fixed_models'])


if __name__ == '__main__':
    main()
