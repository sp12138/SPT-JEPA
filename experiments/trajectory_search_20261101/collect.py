"""Completed-only paired results; all models, both seeds, both case sets retained."""
import argparse
from pathlib import Path
import numpy as np
from common import read, sha, write


def interval(values):
    values = np.asarray(values, dtype=float)
    indices = np.random.default_rng(20261101).integers(len(values), size=(10000, len(values)))
    return dict(mean=float(values.mean()), ci95=np.quantile(values[indices].mean(1), [.025, .975]).tolist(), n=len(values))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, output = Path(a.root), Path(a.output)
    plan = read(root / 'candidate-plan.json')
    assert len(plan['jobs']) in (12, 24)
    rows, reports = [], {}
    immutable = {}
    for host in sorted({j['host'] for j in plan['jobs']}):
        close = read(root / (host + '-closure.json'))
        assert close.get('all_jobs_pass', close.get('all_eight_pass'))
        assert set(close['completed']) == {j['id'] for j in plan['jobs'] if j['host'] == host}
    for job in plan['jobs']:
        exit_path = root / 'exits' / (job['id'] + '.json')
        assert read(exit_path)['returncode'] == 0
        spec = read(job['spec'])
        assert sha(job['spec']) == job['spec_sha256']
        out = Path(job['output'])
        metrics, audit = read(out / 'metrics.json'), read(out / 'trajectory-search.json')
        assert audit['passed'] and audit['state_before'] == audit['state_after']
        assert audit['predict_calls'] == 5 * audit['cost_calls']
        assert len(audit['injections']) * 30 == audit['cost_calls']
        assert len(metrics['pairs']) == 50 and len(metrics['metrics']['episode_successes']) == 50
        for path, expected in spec['immutable_inputs'].items():
            if path not in immutable:
                immutable[path] = sha(path)
            assert immutable[path] == expected
        success = np.asarray(metrics['metrics']['episode_successes'], dtype=float)
        assert np.isin(success, [0, 1]).all()
        reports[job['id']] = (spec, metrics, audit, success)
        rows.append(dict(id=job['id'], family=spec['family'], seed=spec['seed'], cases=spec['case_set'],
              mode=spec['mode'], success_percent=100 * float(success.mean()), success_vector=success.tolist(),
              seconds=metrics['seconds'], cost_calls=audit['cost_calls'], predict_calls=audit['predict_calls'],
              report_sha256=sha(out / 'metrics.json'), audit_sha256=sha(out / 'trajectory-search.json'),
              actual_exit_sha256=sha(exit_path), historical_original_success_vector_equal=audit['historical_original_success_vector_equal']))
    contrasts = []
    for family in ('native', 'tf'):
        for case in sorted({r['cases'] for r in rows}):
            for treatment, control in [('random', 'original'), ('retrieval', 'original'), ('retrieval', 'random')]:
                diffs = []
                for seed in (1, 2):
                    name = f'{family}-seed{seed}-{case}-'
                    s, m, d, values = reports[name + treatment]
                    _, ref, _, baseline = reports[name + control]
                    assert m['pairs'] == ref['pairs']
                    for key in ('checkpoint', 'solver', 'plan', 'normalization', 'protocol', 'checkpoint_audit'):
                        assert m[key] == ref[key]
                    delta = 100 * (values - baseline)
                    diffs.append(delta)
                    contrasts.append(dict(family=family, cases=case, treatment=treatment, control=control,
                          seed=seed, **interval(delta), gained=int(((values == 1) & (baseline == 0)).sum()),
                          lost=int(((values == 0) & (baseline == 1)).sum())))
                contrasts.append(dict(family=family, cases=case, treatment=treatment, control=control,
                                      seed='mean_of_fixed_two', **interval(np.mean(diffs, axis=0))))
    # First provided dataset images/goals are common across modes/models. Later
    # states and banks legitimately diverge after their different actions.
    first_bank_proof = []
    for case in sorted({r['cases'] for r in rows}):
        entries = [reports[f'{f}-seed{s}-{case}-{mode}'][2]['injections'][0]
                   for f in ('native', 'tf') for s in (1, 2) for mode in ('original', 'random', 'retrieval')]
        assert len({x['before_sha256'] for x in entries}) == 1
        retrieved = [reports[f'{f}-seed{s}-{case}-retrieval'][2]['injections'][0]
                     for f in ('native', 'tf') for s in (1, 2)]
        assert all(x['ids'] == retrieved[0]['ids'] and x['query_sha256'] == retrieved[0]['query_sha256'] for x in retrieved)
        first_bank_proof.append(dict(cases=case, original_bank_sha256=entries[0]['before_sha256'],
                                     same_retrieved_ids_all4_models=True, query_sha256=retrieved[0]['query_sha256']))
    result = dict(passed=True, endpoints=len(plan['jobs']), rows=rows, contrasts=contrasts, first_case_bank_proof=first_bank_proof,
          immutable_inputs_checked=len(immutable), collector_sha256=sha(__file__),
          limitation='Fixed two models/seeds,50 cases/set; case bootstrap excludes simulator repeats and seed-population uncertainty. Controller control, no learned-method novelty claim. All modes/controllers retained.')
    write(output / 'results.json', result)
    print(f'All{len(plan["jobs"])} exits, paired cases, model state and first-bank contracts pass.', flush=True)
    print('\n'.join(f"{x['id']}: {x['success_percent']:.1f}%" for x in rows), flush=True)


if __name__ == '__main__':
    main()
