"""Complete3-task spatial exposure stages with both unchanged matched parents."""
import argparse

import numpy as np

from common import B, read, sha, write


def interval(values):
    values = np.asarray(values, dtype=float)
    indices = np.random.default_rng(20261101).integers(len(values), size=(10000, len(values)))
    return dict(mean_pp=float(values.mean()), ci95_pp=np.quantile(values[indices].mean(1), [.025, .975]).tolist())


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--step', type=int, choices=[1200, 2000], required=True)
    parser.add_argument('--output', required=True);args = parser.parse_args()
    original = B / 'runs/unified_three_20261101'
    spatial = B / 'runs/unified_spatial2000_20261101'
    definitions = [('flat', original, 'flat'), ('indirect_head', original, 'field'), ('spatial', spatial, 'field')]
    cases = ['primary'] if args.step == 1200 else ['primary', 'fresh']
    rows, diagnostics, bindings = [], [], {}
    for task in ['pusht', 'tworoom', 'cube']:
        for seed in [1, 2]:
            observed = actions = cohort = None
            for label, root, arm in definitions:
                ident = f'{task}-{arm}-seed{seed}'; run = root / ident
                if args.step == 2000 or label != 'spatial':
                    terminal = root / 'worker-records' / (ident + '-maturity2000.json');done = read(terminal)
                    assert done['passed'] and all(x['returncode'] == 0 for x in done['results'])
                else:
                    # The frozen sequential worker starts2000 only after all
                    #1200 train/audit/L2/NGC children returned0. No partial score.
                    terminal = root / 'processes' / (ident + '-maturity-train2000.json')
                    done = read(terminal)
                    assert done['command'][-1] == '2000'
                bindings[str(terminal)] = sha(terminal)
                ap = root / 'audits' / f'{ident}-step{args.step}' / 'audit.json';audit = read(ap)
                assert audit['passed'] and audit['native_cpu_model_optimizer_scheduler_restore']
                assert audit['state_before'] == audit['state_after']
                assert sha(run / 'full-archives' / f'step{args.step:07d}.ckpt') == audit['full_checkpoint_sha256']
                d = read(ap.with_name('state-audit.json'))
                with np.load(ap.with_name('state-audit.npz')) as data:
                    if observed is None:observed, actions, cohort = data['observed'].copy(), data['actions'].copy(), d['cohort']
                    else:
                        assert np.array_equal(observed, data['observed']) and np.array_equal(actions, data['actions'])
                        assert cohort == d['cohort']
                diagnostics.append(dict(task=task, seed=seed, arm=label, values=d['results'], audit_sha256=sha(ap)))
                for case in cases:
                    for mode in ['l2', 'ngc']:
                        suffix = '' if case == 'primary' else '-fresh'
                        out = root / 'graph-evaluations' / f'{ident}-step{args.step}-{mode}{suffix}'
                        graph = read(out / 'graph-audit.json');assert graph['passed']
                        assert graph['model_audit']['state_before'] == graph['state_after']
                        metrics = read(out / 'metrics.json');values = metrics['metrics']['episode_successes']
                        assert len(values) == 50 and all(isinstance(v, bool) for v in values)
                        pairs = metrics['pairs'] if 'pairs' in metrics else dict(episodes=metrics['episodes'], start_steps=metrics['start_steps'])
                        rows.append(dict(task=task, seed=seed, arm=label, case_set=case, mode=mode,
                                         success_percent=2 * sum(values), success_vector=values, pairs=pairs,
                                         metrics_sha256=sha(out / 'metrics.json'), graph_audit_sha256=sha(out / 'graph-audit.json')))
    contrasts = []
    for task in ['pusht', 'tworoom', 'cube']:
        for case in cases:
            for mode in ['l2', 'ngc']:
                for control in ['flat', 'indirect_head']:
                    differences = []
                    for seed in [1, 2]:
                        left, right = [next(x for x in rows if (x['task'], x['seed'], x['arm'], x['mode'], x['case_set']) ==
                                            (task, seed, arm, mode, case)) for arm in ['spatial', control]]
                        assert left['pairs'] == right['pairs']
                        delta = 100 * (np.asarray(left['success_vector'], float) - np.asarray(right['success_vector'], float))
                        differences.append(delta)
                        contrasts.append(dict(task=task, case_set=case, mode=mode, control=control, seed=seed, **interval(delta)))
                    contrasts.append(dict(task=task, case_set=case, mode=mode, control=control,
                                          seed='two_fixed_models', **interval(np.mean(differences, axis=0))))
    write(args.output, dict(passed=True, step=args.step, rows=rows, diagnostics=diagnostics,
          spatial_minus_control=contrasts, phase_bindings=bindings, observed_actions_cohorts_all_exact=True,
          collector_sha256=sha(__file__),
          scope='Complete stage,all3tasks/2seeds/2controllers/case sets. Identical frozen local1024 observations. '
                'Spatial predictor11001104 versus flat11389184 parameters;higher spatial compute. '
                'Six actual post-copy CPU restores plus bound original600 audits;held parents unchanged. '
                'Conditional case CIs omit seed-population/repeat uncertainty and multiplicity adjustment. '
                'Fixed2000 cap,no automatic extension/convergence/novelty claim.'))
    print([(x['task'], x['arm'], x['seed'], x['case_set'], x['mode'], x['success_percent']) for x in rows])


if __name__ == '__main__':main()
