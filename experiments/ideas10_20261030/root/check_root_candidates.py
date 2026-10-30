"""Exact identities, failure controls, and logged-action support for R1/R2.

No world-model training or task-performance claim. All HDF5 use is read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def checks():
    rng = np.random.default_rng(20261030)
    e = rng.normal(size=(8, 13))
    pair = sum(float(np.square(e[i] - e[j]).sum()) for i in range(8) for j in range(i))
    centered = 8 * float(np.square(e - e.mean(0)).sum())
    assert abs(pair - centered) < 1e-10
    a = np.linspace(-1., 1., 200001)
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    v = (1 - a*a)**2
    vp = -4 * a * (1 - a*a)
    xs = np.linspace(-.6, .9, 11)
    rows = []
    for x in xs:
        mean = .45 * np.tanh(2 * x)
        sigma = .35
        p = np.exp(-.5*((a - mean)/sigma)**2)
        p /= integrate(p, a)
        score = -(a - mean) / sigma**2
        stein = vp + v * score
        norm = integrate(v * p, a)
        affine_y = x + 1.7*a
        affine = -integrate(affine_y * stein * p, a) / norm
        clipped_y = np.minimum(x + a, .6)
        lhs = -integrate(clipped_y * stein * p, a) / norm
        rhs = integrate((x + a < .6) * v * p, a) / norm
        assert abs(affine - 1.7) < 1e-8
        assert abs(lhs - rhs) < 2e-5
        rows.append(dict(x=float(x), action_mean=float(mean), affine_coordinate=float(affine),
                         contact_coordinate=float(lhs), derivative_integral=float(rhs), mass=float(norm)))
    contact_var = float(np.var([r['contact_coordinate'] for r in rows]))
    assert contact_var > .03
    # State-independent affine response cannot distinguish different locations.
    affine_var = float(np.var([r['affine_coordinate'] for r in rows]))
    assert affine_var < 1e-16
    return dict(r1_equivalence_error=abs(pair-centered), r2_rows=rows,
                r2_contact_constant_state_mse_floor=contact_var,
                r2_free_space_location_counterexample_variance=affine_var,
                endpoint_atoms_have_zero_v_and_vprime=bool(v[0]==v[-1]==vp[0]==vp[-1]==0),
                scope='Numerical checks of analytic identities; no learned-model or benchmark result.')


def support(base):
    import h5py
    import hdf5plugin  # noqa: F401, read supplied compression filters
    result = {}
    for task in ['tworoom', 'cube']:
        root = Path(base) / 'data/hardtasks-v1' / (task+'-metadata')
        card = json.loads((root/'dataset_card.json').read_text())
        offsets, lengths, ids, train = [np.load(root/(name+'.npy')) for name in ['offsets','lengths','episode_ids','train_episodes']]
        eligible = np.flatnonzero(np.isin(ids, train) & (lengths > 26))
        rng = np.random.default_rng(1042)
        selected = rng.choice(eligible, min(256,len(eligible)), replace=False)
        blocks, metadata = [], []
        with h5py.File(card['h5_path'], 'r', swmr=True) as f:
            for ep in selected:
                start = int(rng.integers(0, lengths[ep]-25))
                row = int(offsets[ep]+start)
                blocks.append(np.asarray(f['action'][row:row+25], dtype=np.float64))
                metadata.append([int(ids[ep]), start])
        actions = np.concatenate(blocks)
        assert np.isfinite(actions).all()
        endpoint = np.isclose(np.abs(actions), 1., atol=1e-7, rtol=0.)
        interior = (1-actions**2)**2
        dimensions = []
        for d in range(actions.shape[1]):
            values, counts = np.unique(np.round(actions[:,d],7), return_counts=True)
            order = np.argsort(counts)[-3:][::-1]
            dimensions.append(dict(min=float(actions[:,d].min()), max=float(actions[:,d].max()),
                endpoint_fraction=float(endpoint[:,d].mean()),
                mean_boundary_safe_weight=float(interior[:,d].mean()),
                unique_rounded_values=len(values),
                top_rounded_values=[dict(value=float(values[k]),fraction=float(counts[k]/len(actions)))for k in order]))
        result[task] = dict(rows=len(actions), episodes=metadata, dimensions=dimensions,
            interpretation='Endpoint mass or few discrete values invalidates a naive continuous-density score target; absence of atoms does not establish conditional density accuracy or causal sufficiency.')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--base');p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    r=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),theory=checks())
    if a.base:r['logged_action_support']=support(a.base)
    out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(r1_error=r['theory']['r1_equivalence_error'],r2_contact_variance=r['theory']['r2_contact_constant_state_mse_floor'],
        endpoints={k:[v['endpoint_fraction']for v in x['dimensions']]for k,x in r.get('logged_action_support',{}).items()})))


if __name__=='__main__':main()
