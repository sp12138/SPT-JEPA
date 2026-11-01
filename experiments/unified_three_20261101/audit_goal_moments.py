"""One CPU-only realized-second-moment substitution, plus exact witnesses.

This is an oracle diagnostic, not a deployable cost or uncertainty estimate.
No model, encoder, simulator, new feature fitting, or physical-label fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

B = Path('/efs/prescottsun/SPT-JEPA')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    draws = np.random.default_rng(1042).integers(len(values), size=(5000, len(values)))
    return dict(mean=float(values.mean()), ci95=np.quantile(values[draws].mean(1), [.025, .975]).tolist())


def witnesses():
    goal = np.array([1., 0.])
    risky = np.array([[.5, np.sqrt(.75)], [.5, -np.sqrt(.75)]])
    safe = np.array([[.6, .8], [.6, .8]])
    rows = []
    for name, outcomes in [('risky', risky), ('safe', safe)]:
        mu = outcomes.mean(0)
        m2 = np.square(outcomes).sum(1).mean()
        plugin = np.square(mu - goal).sum()
        actual = np.square(outcomes - goal).sum(1).mean()
        proper = m2 - 2 * mu @ goal + goal @ goal
        assert np.allclose(np.square(outcomes).sum(1), 1.)
        assert abs(actual - proper) < 1e-12
        rows.append(dict(name=name, plugin=float(plugin), expected=float(actual), variance=float(m2 - mu @ mu)))
    assert rows[0]['plugin'] < rows[1]['plugin']
    assert rows[0]['expected'] > rows[1]['expected']
    # Exact nonlinear propagation does not commute with an observation mean.
    states = np.array([-1., 1.])
    assert np.square(states.mean()) == 0 and np.square(states).mean() == 1
    return dict(unit_norm_goal_witness=rows, nonlinear_mean_closure=dict(f_of_mean=0., mean_of_f=1.),
                scope='Standard squared-risk/Jensen identities,not new theorems or evidence about current models.')


def main(output):
    output = Path(output)
    assert not output.exists()
    rows, provenance, inputs = [], [], []
    shared_truth = {}
    for kind in ['local6000', 'native23342', 'tf23342']:
        root = B / ('runs/cube_pool_reuse2000_20261031' if kind == 'local6000'
                    else 'runs/cube_pool_maturity23342_20261101')
        for seed in [1, 2]:
            ident = f'{kind}-seed{seed}'
            path = root / ident / 'result.json'
            result = read(path)
            assert result['passed'] and result['state_before'] == result['state_after']
            assert result['rng_before'] == result['rng_after'] and len(result['cases']) == 50
            provenance.append(dict(id=ident, path=str(path), sha256=sha(path)))
            for rec in result['cases']:
                p = Path(rec['file']); assert sha(p) == rec['sha256']
                inputs.append(dict(path=str(p), sha256=rec['sha256']))
                with np.load(p) as data:
                    true = data['true_endpoint'].astype(np.float64)
                    pred = data['predicted_endpoint'].astype(np.float64)
                    goal = data['goal'].astype(np.float64)
                    family = data['family']
                    physical = data['physical_cost'][:, -1]
                    position = data['position_error_m'][:, -1]
                    angle = data['orientation_error_deg'][:, -1]
                    hold = data['success'].any(1)
                    ids = np.flatnonzero(np.isin(family, ['zero', 'gaussian']) | np.char.startswith(family, 'cem_'))
                    assert len(ids) == 22 and true.shape == pred.shape and true.shape[0] == 39
                    truth = [physical, position, angle, hold, family, data['raw_actions']]
                    if rec['case'] not in shared_truth:
                        shared_truth[rec['case']] = [v.copy() for v in truth]
                    else:
                        assert all(np.array_equal(x, y) for x, y in zip(truth, shared_truth[rec['case']]))
                    m2_true = np.square(true).sum(1)
                    m2_pred = np.square(pred).sum(1)
                    plugin = np.square(pred - goal).sum(1)
                    old = data['predicted_cost']
                    assert np.allclose(plugin, old, rtol=3e-6, atol=1e-8)
                    substitute = m2_true - 2 * pred @ goal + goal @ goal
                    assert np.allclose(substitute - plugin, m2_true - m2_pred, rtol=1e-8, atol=1e-8)
                    assert np.isfinite(substitute).all()
                    for group, selected in [('nonoracle', ids), ('all39', np.arange(39))]:
                        original = int(selected[np.argmin(plugin[selected])])
                        assert original == int(selected[np.argmin(old[selected])])
                        changed = int(selected[np.argmin(substitute[selected])])
                        entry = dict(id=ident, kind=kind, seed=seed, case=rec['case'], group=group,
                                     original_index=original, substituted_index=changed,
                                     original_physical_rank=float(spearmanr(plugin[selected], physical[selected]).statistic),
                                     substituted_physical_rank=float(spearmanr(substitute[selected], physical[selected]).statistic),
                                     negative_substitute_fraction=float((substitute[selected] < 0).mean()),
                                     mean_norm_gap=float((m2_true - m2_pred)[selected].mean()),
                                     selected_norm_gap=float(m2_true[original] - m2_pred[original]))
                        for label, index in [('original', original), ('substituted', changed)]:
                            entry.update({f'{label}_pose_cost': float(physical[index]),
                                          f'{label}_hold': float(hold[index]),
                                          f'{label}_joint': float(position[index] <= .03 and angle[index] <= 15)})
                        rows.append(entry)
    assert len(rows) == 600 and len(shared_truth) == 50
    summaries = []
    for kind in ['local6000', 'native23342', 'tf23342']:
        for group in ['nonoracle', 'all39']:
            by_seed = []
            for seed in [1, 2]:
                rr = sorted([r for r in rows if (r['kind'], r['seed'], r['group']) == (kind, seed, group)], key=lambda r: r['case'])
                assert len(rr) == 50
                by_seed.append(rr)
                metrics = ['pose_cost', 'hold', 'joint', 'physical_rank']
                summaries.append(dict(kind=kind, seed=seed, group=group,
                                      changed=sum(r['original_index'] != r['substituted_index'] for r in rr),
                                      original={k: interval([r['original_' + k] for r in rr]) for k in metrics},
                                      substituted={k: interval([r['substituted_' + k] for r in rr]) for k in metrics},
                                      delta={k: interval([r['substituted_' + k] - r['original_' + k] for r in rr]) for k in metrics}))
            assert [r['case'] for r in by_seed[0]] == [r['case'] for r in by_seed[1]]
            summaries.append(dict(kind=kind, seed='two_fixed_models', group=group,
                                  delta={k: interval(np.mean([[r['substituted_' + k] - r['original_' + k] for r in rr] for rr in by_seed], axis=0)) for k in metrics}))
    payload = dict(passed=True, calls=dict(model=0, encoder=0, renderer=0, simulator=0, fit=0, GPU=0),
                   witnesses=witnesses(), provenance=provenance, inputs=inputs, rows=rows, summaries=summaries,
                   script_sha256=sha(__file__), all300_input_hashes_pass=True, all_physical_truth_and_actions_exact=True,
                   limitations=['Realized future norm is privileged and not a deployable moment estimate.',
                                'A realized squared norm is not an oracle conditional expectation under partial observability.',
                                'Current predictions need not be conditional means;the missing-variance identity alone does not identify their failure.',
                                'Six fixed checkpoints,old50-case pool22nonoracle primary;all39 includes17recorded-action neighbors.',
                                '25-step first segment,not current50-step MPC;no fresh/OOD/repeat or learned-head validation.',
                                'Conditional case intervals omit seed-population/repeated-environment variance and multiplicity.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(payload, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps([r for r in summaries if r['group'] == 'nonoracle'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
