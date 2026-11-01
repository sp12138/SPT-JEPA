"""Completed-only ranking/optimism audit on eight unchanged-pool rescores."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np

from common import B, read, sha, write


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True);p.add_argument('--output', required=True)
    a = p.parse_args();root, out = Path(a.root), Path(a.output)
    protocol = read(root / 'protocol.json');assert protocol['passed'] and len(protocol['jobs']) == 8
    helper_path = Path(__file__).resolve().parents[1] / 'next_mechanism_20261101/maturity/pool_collect.py'
    spec = importlib.util.spec_from_file_location('fixed_pool_formulas', helper_path)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    rows, reports, bindings = [], [], {}
    for job in protocol['jobs']:
        ep = root / 'exits' / (job['id'] + '.json');exit_record = read(ep)
        assert exit_record['passed'] and exit_record['returncode'] == 0 and sha(job['spec']) == job['spec_sha256']
        rp = Path(job['output']) / 'result.json';report = read(rp)
        assert report['passed'] and report['spec'] == read(job['spec'])
        assert report['state_before'] == report['state_after'] and report['rng_before'] == report['rng_after']
        assert report['actual_calls'] == report['spec']['budget']
        reference_path = B / f'runs/cube_pool_reuse2000_20261031/local6000-seed{report["spec"]["seed"]}/result.json'
        reference = read(reference_path);assert reference['passed']
        result = helper.summarize(report, reference)
        oldcases = {x['case']:x for x in reference['cases']}
        for rec in report['cases']:
            with np.load(rec['file']) as f, np.load(oldcases[rec['case']]['file']) as old:
                for key in ['initial', 'goal', 'true_endpoint', 'encoded_cost']:
                    assert np.array_equal(f[key], old[key]), key
                for group, selected in [('nonoracle', np.flatnonzero(np.isin(f['family'], ['zero','gaussian']) | np.char.startswith(f['family'],'cem_'))),
                                        ('oracle_neighborhood', np.flatnonzero(np.char.startswith(f['family'],'recorded'))),
                                        ('all', np.arange(39))]:
                    entry = next(x for x in result['cases'][group] if x['case'] == rec['case'])
                    chosen = int(selected[np.argmin(f['predicted_cost'][selected])])
                    errors = np.square(f['predicted_endpoint'].astype(np.float64) - f['true_endpoint']).mean(-1)
                    entry.update(endpoint_state_mse=float(errors[selected].mean()),
                                 selected_endpoint_mse=float(errors[chosen]),
                                 goal_cost_optimism=float(f['encoded_cost'][chosen]-f['predicted_cost'][chosen]))
        for group in helper.GROUPS:
            for metric in ['endpoint_state_mse','selected_endpoint_mse','goal_cost_optimism']:
                result['summary'][group][metric] = helper.interval([x[metric] for x in result['cases'][group]])
        result.update(report=str(rp), report_sha256=sha(rp), reference_report_sha256=sha(reference_path))
        rows.append(result);reports.append(report);bindings[str(ep)] = sha(ep)
    totals = {k:sum(r['actual_calls'][k] for r in reports) for k in reports[0]['actual_calls']}
    assert totals == protocol['total_calls']
    paired = []
    metrics = ['predicted_hold_success','predicted_joint_pose_pass','encoded_predicted_rank','physical_predicted_rank',
               'encoded_selection_regret_relative_range','endpoint_state_mse','selected_endpoint_mse','goal_cost_optimism']
    for candidate, control in protocol['matched_contrasts']:
        for group in helper.GROUPS:
            for metric in metrics:
                deltas = []
                for seed in [1,2]:
                    left, right = [next(r for r in rows if r['id'] == f'{name}-seed{seed}')['cases'][group] for name in [candidate,control]]
                    assert [x['case'] for x in left] == [x['case'] for x in right]
                    delta = [None if x[metric] is None or y[metric] is None else x[metric]-y[metric] for x,y in zip(left,right)]
                    deltas.append(delta)
                    paired.append(dict(candidate=candidate,control=control,group=group,metric=metric,seed=seed,**helper.interval(delta)))
                both = [None if x is None or y is None else .5*(x+y) for x,y in zip(*deltas)]
                paired.append(dict(candidate=candidate,control=control,group=group,metric=metric,seed='two_fixed_models',**helper.interval(both)))
    full = dict(passed=True, rows=rows, paired=paired, actual_calls=totals, exit_bindings=bindings,
                helper_sha256=sha(helper_path), collector_sha256=sha(__file__), protocol_sha256=sha(root/'protocol.json'),
                all_frozen_input_and_truth_arrays_exact=True, all8_weights_buffers_RNG_unchanged=True,
                scope='Same old50x39 pool,H1/T5;nonoracle22 primary,recorded17 secondary. No new simulation/encoding/fit. '
                      'Not current CEM/warmhistory or50-step task performance. Intervals condition on two weights and one simulator realization. '
                      'Pair600 only with600,2000 only with2000. Optimism is a diagnostic,not a proven causal mediator.')
    write(out/'results.json', full)
    write(out/'summary.json', dict(full, rows=[{k:v for k,v in r.items() if k not in ['cases','zero']} for r in rows]))
    print([(r['id'], {k:r['summary']['nonoracle'][k]['mean'] for k in ['predicted_hold_success','encoded_predicted_rank','endpoint_state_mse','selected_endpoint_mse','goal_cost_optimism']}) for r in rows])
    print([x for x in paired if x['group']=='nonoracle' and x['seed']=='two_fixed_models'])


if __name__ == '__main__':main()
