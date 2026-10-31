"""Exact CPU witnesses for a rejected path-budget learning construction.

Candidate: d_phi(s,g)=max(0,max_j(phi_j(g)-phi_j(s))); factual-edge
upper-bound penalty plus endpoint expansion. This reduces to QRL's mechanism,
not a new method: https://proceedings.mlr.press/v202/wang23al.html
No model/data/GPU call, training, or benchmark-performance claim.
"""
from collections import deque
from fractions import Fraction
from pathlib import Path
import hashlib
import json


def shortest(edges):
    n = len(edges)
    result = []
    for source in range(n):
        d = [None] * n
        d[source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in edges[u]:
                if d[v] is None:
                    d[v] = d[u] + 1
                    queue.append(v)
        assert all(v is not None for v in d)
        result.append(d)
    return result


def main():
    # Logged action a moves around a directed cycle. Action b is never logged.
    logged = [[1], [2], [3], [0]]
    distance = shortest(logged)
    # Exact potential construction phi_j(v)=distance(j,v).
    represented = [[max(0, max(row[g] - row[s] for row in distance))
                    for g in range(4)] for s in range(4)]
    assert represented == distance
    assert all(represented[u][v] <= 1 for u, vs in enumerate(logged) for v in vs)
    assert all(represented[u][v] <= represented[u][k] + represented[k][v]
               for u in range(4) for v in range(4) for k in range(4))
    assert represented[0][1] == 1 and represented[1][0] == 3

    # Two deterministic worlds agree on every logged (state,a,next-state).
    # In world B only, the unlogged action b from state0 reaches goal3.
    world_a = [vs + [i] for i, vs in enumerate(logged)]
    world_b = [list(vs) for vs in world_a]
    world_b[0][-1] = 3
    assert all(world_a[s][0] == world_b[s][0] for s in range(4))
    da, db = shortest(world_a)[0][3], shortest(world_b)[0][3]
    assert (da, db) == (3, 1)

    # With a stochastic action, observing an edge is only possibility evidence.
    p = Fraction(1, 10)
    horizon = 5
    success = 1 - (1 - p) ** horizon
    expected_hitting_time = 1 / p
    assert success == Fraction(40951, 100000) and expected_hitting_time == 10
    result = dict(
        passed=True, admitted_new_method=False,
        graph_distance=distance, potential_quasimetric=represented,
        all_16_directed_pairs_recovered=True,
        asymmetry=dict(forward=1, reverse=3,
                       consequence="No symmetric Euclidean encoding can equal both distances."),
        missing_action_witness=dict(logged_transitions_identical=True,
                                   learned_logged_graph_cost=3,
                                   true_cost_world_A=da, true_cost_world_B=db,
                                   one_step_goal_reachable_A=False,
                                   one_step_goal_reachable_B=True),
        stochastic_witness=dict(one_step_success_probability=float(p),
                                horizon=horizon, success_within_horizon=float(success),
                                expected_hitting_time=int(expected_hitting_time),
                                successful_observed_edge_cost_bound=1),
        interpretation="Directed path geometry is distinguishable from visual similarity, "
                       "but this construction is QRL-like, needs transition support, and a "
                       "successful stochastic edge does not certify expected hitting cost.",
        scope="stdlib exact finite-state algebra only; no data, model calls, optimization or GPU",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output = Path(__file__).with_name("path-budget-rejection.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(passed=True, output=str(output), admitted_new_method=False)))


if __name__ == "__main__":
    main()
