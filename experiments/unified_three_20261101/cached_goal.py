"""Read-only goal-geometry audit on the complete old Cube candidate cache.

Candidate indices are locked by latent costs before physical diagnostic fields
are read. Encoded endpoints diagnose geometry separately from prediction; this
is a25-step candidate-pool diagnostic, not a bound on50-step receding MPC.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from common import B, read, sha, write
from graph import encoding_state


def interval(x):
    x = np.asarray(x, dtype=float)
    indices = np.random.default_rng(1042).integers(len(x), size=(5000, len(x)))
    return dict(mean=float(x.mean()), ci95=np.quantile(x[indices].mean(1), [.025, .975]).tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, out = Path(a.root), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    public = Path(__file__).resolve().parents[1] / 'public_ngc_20261031'
    sys.path.insert(0, str(public))
    from stable_cost import StableNGCCost
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    rows, costs, provenance = [], {}, []
    for seed in [1, 2]:
        path = B / f'runs/cube_pool_reuse2000_20261031/local6000-seed{seed}/result.json'
        old = read(path)
        assert old['passed'] and old['state_before'] == old['state_after']
        assert old['rng_before'] == old['rng_after'] and len(old['cases']) == 50
        gp = root / f'graphs/cube-seed{seed}/graph.json'
        graph = read(gp)
        assert graph['passed'] and sha(graph['graph']) == graph['graph_sha256']
        assert graph['reference']['encoding_tensors_exact'] > 0
        assert sha(old['spec']['weights']) == old['spec']['weights_sha256']
        cached_weights = torch.load(old['spec']['weights'], map_location='cpu', weights_only=True, mmap=True)
        graph_weights = torch.load(graph['weights'], map_location='cpu', weights_only=True, mmap=True)
        left, right = encoding_state(cached_weights), encoding_state(graph_weights)
        assert left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)
        encoding_tensors_exact = len(left)
        del cached_weights, graph_weights, left, right
        with np.load(graph['graph']) as f:
            objective = StableNGCCost(f['nodes'].copy(), f['graph_distance'].copy(), 15, 4)
        provenance.append(dict(seed=seed, cache_report=str(path), cache_sha256=sha(path),
                               cache_spec=old['spec'], graph_report=str(gp), graph_sha256=sha(gp),
                               encoding_tensors_exact=encoding_tensors_exact))
        for rec in old['cases']:
            assert sha(rec['file']) == rec['sha256']
            with np.load(rec['file']) as f:
                goal, family = f['goal'], f['family']
                ids = np.flatnonzero(np.isin(family, ['zero', 'gaussian']) | np.char.startswith(family, 'cem_'))
                assert len(ids) == 22 and goal.shape == (1024,)
                locked = []
                for endpoint, key, oldkey in [('encoded', 'true_endpoint', 'encoded_cost'),
                                               ('predicted', 'predicted_endpoint', 'predicted_cost')]:
                    z = f[key]
                    assert z.shape == (39, 1024) and z.dtype == goal.dtype == np.float32
                    l2 = np.square(z - goal).sum(-1)
                    np.testing.assert_allclose(l2, f[oldkey], rtol=2e-6, atol=1e-8)
                    with torch.inference_mode():
                        ngc = objective(dict(predicted_emb=torch.from_numpy(z)[None, :, None],
                                             goal_emb=torch.from_numpy(goal)[None, None]))[0].numpy()
                    assert np.isfinite(ngc).all()
                    costs[f'seed{seed}_case{rec["case"]:04d}_{endpoint}'] = np.stack([l2, ngc])
                    for group, indices in [('nonoracle', ids), ('all_secondary', np.arange(39))]:
                        locked.append((endpoint, group, indices, int(indices[np.argmin(l2[indices])]),
                                       int(indices[np.argmin(ngc[indices])])))
                # Physical fields enter only after all choices are locked.
                pos, angle = f['position_error_m'][:, -1], f['orientation_error_deg'][:, -1]
                hold, physical = f['success'].any(1), f['physical_cost'][:, -1]
                for endpoint, group, indices, left, right in locked:
                    row = dict(seed=seed, case=rec['case'], endpoint=endpoint, group=group,
                               l2_index=left, ngc_index=right, changed=left != right,
                               coverage_hold=float(hold[indices].any()))
                    for mode, index in [('l2', left), ('ngc', right)]:
                        for k, v in dict(hold=hold[index], joint=pos[index] <= .03 and angle[index] <= 15,
                                         position_cm=100 * pos[index], angle_deg=angle[index],
                                         pose_cost=physical[index]).items():
                            row[f'{mode}_{k}'] = float(v)
                        row[f'{mode}_family'] = str(family[index])
                    rows.append(row)
    assert len(rows) == 400
    metrics = ['hold', 'joint', 'position_cm', 'angle_deg', 'pose_cost']
    summaries, paired = [], []
    for endpoint in ['encoded', 'predicted']:
        for group in ['nonoracle', 'all_secondary']:
            both = []
            for seed in [1, 2]:
                rr = [x for x in rows if (x['endpoint'], x['group'], x['seed']) == (endpoint, group, seed)]
                assert len(rr) == 50
                both.append(rr)
                summaries.append(dict(seed=seed, endpoint=endpoint, group=group,
                    changed=sum(x['changed'] for x in rr), coverage_hold=float(np.mean([x['coverage_hold'] for x in rr])),
                    **{mode: {k: interval([x[f'{mode}_{k}'] for x in rr]) for k in metrics} for mode in ['l2', 'ngc']}))
            assert [x['case'] for x in both[0]] == [x['case'] for x in both[1]]
            paired.append(dict(endpoint=endpoint, group=group, ngc_minus_l2={k: interval(np.mean([
                [x[f'ngc_{k}'] - x[f'l2_{k}'] for x in rr] for rr in both], axis=0)) for k in metrics}))
    np.savez_compressed(out / 'costs.npz', **costs)
    write(out / 'results.json', dict(passed=True, script_sha256=sha(__file__), rows=rows,
          provenance=provenance, summaries=summaries, paired=paired, new_model_calls=0,
          new_simulator_calls=0, optimizer_steps=0, physical_labels_used_for_selection=False,
          costs_sha256=sha(out / 'costs.npz'),
          scope='CPU FP32 public NGC replay on fixed39-candidate caches;22 nonoracle candidates primary. '
                'True endpoints separate geometry from prediction. This is not an actual CUDA planning replay, '
                'a50-step MPC ceiling, a fitted metric or a new learner. Intervals condition on two frozen models.'))
    print([(x['seed'], x['endpoint'], x['l2']['hold']['mean'], x['ngc']['hold']['mean'])
           for x in summaries if x['group'] == 'nonoracle'], flush=True)


if __name__ == '__main__':
    main()
