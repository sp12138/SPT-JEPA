"""D9 synchronized-memory geometry: exact toys and optional frozen CPU audit.

No GPU, environment stepping, model loading, or learning. Physical fields in
the optional old pool audit are diagnostic only, never proposed loss targets.
"""
import argparse
import json
import math
from pathlib import Path


def synthetic():
    n, offset = 64, 7
    points = [(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n))
              for k in range(n)]
    sq = lambda a, b: sum((v - w) ** 2 for v, w in zip(a, b))
    sync = sum(sq(points[u], points[(u + offset) % n]) for u in range(n)) / n
    indep = sum(sq(x, y) for x in points for y in points) / n ** 2
    assert abs(sync - 4 * math.sin(math.pi * offset / n) ** 2) < 1e-12
    assert abs(indep - 2) < 1e-12
    # Both initial conditions have exactly the same uniform endpoint marginal.
    marginal_histogram_l1 = sum(abs(1/n - 1/n) for _ in range(n))
    linear = []
    for alpha in [0, .5, .9, 1]:
        for h in [1, 2, 4, 8]:
            # Goal (r,b)=(0,1); endpoints A=(0,0), B=(2,1).
            costs = [1, 4 * alpha ** (2 * h)]
            linear.append(dict(alpha=alpha, horizon=h, costs=costs,
                               unique_choice=(None if costs[0] == costs[1] else costs.index(min(costs)))))
    contact = []
    alpha, h = .5, 4
    for coupling in [0, .1, .2, .4, 1]:
        # Differences obey delta_r_next=alpha*delta_r,
        # delta_b_next=delta_b+coupling*delta_r under identical inputs.
        dr, db = 2., 0.
        for _ in range(h):
            dr, db = alpha * dr, db + coupling * dr
        costs = [1, dr ** 2 + db ** 2]
        contact.append(dict(coupling=coupling, horizon=h, final_difference=[dr, db],
                            costs=costs, choice=costs.index(min(costs))))
    # A bounded navigation coordinate, pushed right until both hit the wall.
    x, y = .2, .6
    initial_sq = (x-y) ** 2
    for _ in range(10):
        x, y = min(1, x+.1), min(1, y+.1)
    # Independent repeated endpoint noise: debias within the paired unit.
    # Mean difference .3, each environment's noise is independently +/-1.
    delta = .3
    differences = [delta + a - b for a in [-1., 1.] for b in [-1., 1.]]
    naive_noisy_target = sum(v*v for v in differences) / len(differences)
    corrected_target = sum(v*w for v in differences for w in differences) / len(differences)**2
    assert abs(corrected_target-delta**2)<1e-12
    # With a partial observable psi(a,b)=a and swap dynamics, H=1 endpoint
    # equality is not a Markov equivalence: hidden distinctions can reappear.
    s,t=(0,0),(1,0)
    swap=lambda v:(v[1],v[0])
    one_step=(swap(s)[0]-swap(t)[0])**2
    two_step=(swap(swap(s))[0]-swap(swap(t))[0])**2
    assert one_step==0 and two_step==1
    return dict(
        circular_navigation=dict(n_states=n, initial_offset=offset,
            marginal_endpoint_histogram_l1=marginal_histogram_l1,
            synchronous_chord_squared=sync,
            independent_chord_squared=indep,
            synchronous_self_distance=0,
            independent_self_expected_squared=indep,
            reduced_independent_distance=indep-.5*indep-.5*indep),
        direct_control_linear_examples=linear,
        mediated_initial_condition_leakage=contact,
        wall_erases_navigation=dict(initial_squared_difference=initial_sq,
                                     final_squared_difference=(x-y)**2),
        independent_noise_debiasing=dict(true_mean_squared_difference=delta**2,
            naive_expected_squared_difference=naive_noisy_target,
            paired_repeat_inner_product_expectation=corrected_target,
            minimum_single_target=min(v*w for v in differences for w in differences)),
        partial_observation_non_markov=dict(initial_states=[s,t],
            observable="first coordinate", dynamics="swap coordinates",
            H1_distance_squared=one_step,H2_distance_squared=two_step),
        psd_property="s -> (A -> psi(F_A(s))) is an explicit L2 feature map; d_H is its Hilbert distance",
        different_future_policies_warning="conditional policies depending on each initial state are not common actions",
    )


def pool_audit(root):
    import numpy as np
    files = sorted(Path(root).glob("case[0-9][0-9][0-9][0-9].npz"))
    assert len(files) == 50, len(files)
    eff, block = [], []
    for p in files:
        with np.load(p, allow_pickle=False) as q:
            family = q["family"].astype(str)
            indices = np.flatnonzero(family == "zero")
            assert len(family) == 39 and len(indices) == 1
            idx = int(indices[0])
            assert np.array_equal(q["actions"][idx], np.zeros((25,5)))
            eff.append(q["effector_positions"][idx].astype(float))
            block.append(q["positions"][idx].astype(float))
    out = {}
    i, j = np.triu_indices(50, 1)
    for name, xx in [("effector", np.stack(eff)), ("block", np.stack(block))]:
        assert xx.shape == (50,26,3)
        d0 = np.linalg.norm(xx[i,0]-xx[j,0], axis=1)
        rows=[]
        for h in [1,5,10,25]:
            dh = np.linalg.norm(xx[i,h]-xx[j,h], axis=1)
            changed = np.linalg.norm(xx[:,h]-xx[:,0], axis=1)
            keep=d0>1e-6
            rows.append(dict(raw_steps=h, rms_pair_distance_ratio=float(np.sqrt(np.mean(dh**2)/np.mean(d0**2))),
                             median_pair_distance_ratio=float(np.median(dh[keep]/d0[keep])),
                             initial_final_pair_distance_correlation=float(np.corrcoef(d0,dh)[0,1]),
                             mean_individual_displacement_m=float(changed.mean()),
                             max_individual_displacement_m=float(changed.max())))
        out[name]=rows
    return dict(source=str(root), n_cases=50, candidates_per_case=39,
                raw_action="all five controls exactly zero at every one of 25 steps",
                n_cross_case_pairs=1225, diagnostic_coordinates_only=True,
                caveat="read-only old validation bank; supports no training/evaluation gain claim; only zero action tested",
                results=out)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pool-root")
    p.add_argument("--output", type=Path, default=Path(__file__).with_name("fourth-cpu.json"))
    a=p.parse_args()
    out={"synthetic":synthetic()}
    if a.pool_root:
        out["frozen_cube_zero_action"]=pool_audit(a.pool_root)
    a.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"output":str(a.output),"circle":out["synthetic"]["circular_navigation"],
                      "cube":out.get("frozen_cube_zero_action",{}).get("results",{})}))


if __name__=="__main__":
    main()
