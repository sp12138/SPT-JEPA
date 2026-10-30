"""CPU-only exact counterexamples for D7/D8; no benchmark or model loading.

Run with Python stdlib. All constants are declared here; this does not tune to
the Cube/TwoRoom evaluation bank. Physical coordinates below are synthetic
mathematical examples, not training targets for a proposed visual method.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

N = NormalDist()


def gaussian_tv(distance, sigma):
    return 2 * N.cdf(distance / (2 * sigma)) - 1


def required_distance(delta, sigma, per_state_tv_error):
    target = max(0.0, delta - 2 * per_state_tv_error)
    return 2 * sigma * N.inv_cdf((1 + target) / 2), target


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def matvec(a, x):
    return [dot(row, x) for row in a]


def response_order(a, b, c, max_horizon=8):
    response = b[:]
    for order in range(1, max_horizon + 1):
        if abs(dot(c, response)) > 1e-12:
            return order
        response = matvec(a, response)
    return None


def d7():
    # r_next=u; b_next=b+r; c_next=c. c is an uncontrollable distractor.
    a = [[0, 0, 0], [1, 1, 0], [0, 0, 1]]
    b = [1, 0, 0]
    axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    goal = [0, 1, 0]
    controls = [[0, 0], [1, 2]]
    endpoints = []
    for seq in controls:
        x = [0, 0, 0]
        for u in seq:
            x = [v + bi * u for v, bi in zip(matvec(a, x), b)]
        endpoints.append(x)
    euc = [sum((v - g) ** 2 for v, g in zip(x, goal)) for x in endpoints]
    deep = [(x[1] - goal[1]) ** 2 for x in endpoints]
    # Horizon-2 controllability Gramian on (r,b) is exactly identity.
    ab = matvec(a, b)
    gramian = [[b[i] * b[j] + ab[i] * ab[j] for j in range(2)] for i in range(2)]
    assert gramian == [[1, 0], [0, 1]]
    # Unit linear observable: minimize immediate response, maximize delayed.
    # The constant distractor has zero delayed response and is excluded.
    objective = []
    for w in axes:
        immediate, delayed = dot(w, b) ** 2, dot(w, ab) ** 2
        objective.append({"head": w, "early_energy": immediate,
                          "delayed_energy": delayed,
                          "admissible_depth2": immediate == 0 and delayed > 0})
    assert response_order(a, b, axes[0]) == 1
    assert response_order(a, b, axes[1]) == 2
    assert response_order(a, b, axes[2]) is None
    assert euc.index(min(euc)) == 0 and deep.index(min(deep)) == 1
    # Navigation has only x_next=x+u: the deepest controllable head remains x.
    nav_order = response_order([[1]], [1], [1])
    assert nav_order == 1
    # Tiny mediated effects retain exact order but are statistically unstable.
    weak = []
    for eps in [1.0, 0.1, 0.01, 0.001]:
        aw = [[0, 0], [eps, 1]]
        order = response_order(aw, [1, 0], [0, 1])
        weak.append({"coupling": eps, "order": order,
                     "delayed_squared_response": eps ** 2})
    # One frame can span two primitive control steps. Held-constant input then
    # reaches both coordinates in that frame, removing coordinate b's delay.
    coarse_b = [bi + abi for bi, abi in zip(b, ab)]
    coarse_b_coordinate_order = 1 if dot(axes[1], coarse_b) else None
    # A *different* deepest combination b-r still exists at coarse scale;
    # the old object coordinate itself is no longer the deepest output.
    coarse_deep_head = [-1, 1, 0]
    assert dot(coarse_deep_head, coarse_b) == 0
    return {
        "system": {"A": a, "B": b, "observables": ["actuator", "mediated", "background"]},
        "primitive_relative_orders": [response_order(a, b, c) for c in axes],
        "learnable_linear_head_contrast": objective,
        "navigation_order": nav_order,
        "goal": goal, "bounded_control_sequences": controls, "endpoints": endpoints,
        "euclidean_costs": euc, "horizon2_gramian": gramian,
        "inverse_gramian_costs": euc, "deep_output_costs": deep,
        "choice_euclidean_or_gramian": 0, "choice_deep": 1,
        "object_task_errors_of_choices": {"euclidean": 1, "deep": 0},
        "actuator_task_errors_of_choices": {"euclidean": 0, "deep": 4},
        "same_observations_different_goal_semantics": "object-only and actuator-only tasks require opposite choices",
        "weak_coupling": weak,
        "two_primitive_steps_per_observation": {
            "held_input_B": coarse_b,
            "old_mediated_coordinate_order": coarse_b_coordinate_order,
            "new_early_invariant_head": coarse_deep_head,
            "warning": "sampling/action aggregation changes the preferred output, despite identical underlying mechanics"},
        "fixed_head_response_energy_bound": {
            "n_independent_validation_pairs": 128, "norm_delta_phi_bound": 1,
            "alpha": 0.05, "hoeffding_radius": math.sqrt(math.log(40) / 256)},
    }


def d8():
    sigma, allowed_error = 0.1, 0.02
    rows = []
    for label, delta in [("large_future_difference", 0.8),
                         ("small_future_difference", 0.1),
                         ("recurrent_identical_state", 0.0)]:
        dist, target = required_distance(delta, sigma, allowed_error)
        tau = gaussian_tv(dist, sigma)
        # Two Bernoulli experiments with probabilities (1+-delta)/2.
        # The threshold test between Gaussian codes has contrast tau, and a
        # shared randomized threshold decoder can achieve every contrast <=tau.
        exact_best_worst_error = max(0.0, delta - tau) / 2
        assert abs(tau - target) < 1e-12
        rows.append({"pair": label, "future_bernoulli_gap": delta,
                     "target_channel_tv": target, "minimum_mean_distance": dist,
                     "mean_dependent_kl_min_nats": dist ** 2 / 8,
                     "achieved_channel_tv": tau,
                     "exact_best_worst_future_tv_error": exact_best_worst_error})
    fixed_dist = rows[0]["minimum_mean_distance"]
    adaptive_kl = sum(x["mean_dependent_kl_min_nats"] for x in rows[:2])
    fixed_kl = 2 * fixed_dist ** 2 / 8
    # A clean-code scale can be arbitrarily informative to noiseless probes,
    # while Gaussian channel discrimination remains almost chance.
    toy_clean_std, toy_dim = 0.00015, 64
    toy_distance = math.sqrt(2 * toy_dim) * toy_clean_std
    toy_tv = gaussian_tv(toy_distance, sigma)
    # A fixed time-gap cosine target conflicts with exactly recurrent images.
    time_gap, temporal_scale = 4, 2
    temporal_target = math.exp(-time_gap / temporal_scale)
    best_same_cosine = (1 + temporal_target) / 2
    unavoidable_two_term_mean_mse = (1 - temporal_target) ** 2 / 4
    # With arbitrary side information this bound cannot constrain q alone.
    bypass = {"latent_channel_tv": 0.0, "side_information_identifies_state": True,
              "achievable_future_tv": 1.0, "future_prediction_error": 0.0}
    # If future Bernoulli probabilities are estimated independently from n
    # repetitions per initial state, Hoeffding+union gives a conservative LCB.
    n, alpha = 2048, 0.05
    probability_radius = math.sqrt(math.log(4 / alpha) / (2 * n))
    finite_sample = []
    for observed_gap in [0.8, 0.1]:
        lower = max(0, observed_gap - 2 * probability_radius)
        dist, target = required_distance(lower, sigma, allowed_error)
        finite_sample.append({"observed_gap": observed_gap, "gap_lcb": lower,
                              "target_tv": target, "distance": dist})
    # The bound is exact for a Bernoulli future and a Gaussian threshold code.
    exhaustive_error = 0.0
    for delta_int in range(101):
        delta = delta_int / 100
        for d_int in range(81):
            distance = d_int / 100
            tau = gaussian_tv(distance, sigma)
            contrast = min(delta, tau)
            predicted_probs = [(1 - contrast) / 2, (1 + contrast) / 2]
            true_probs = [(1 - delta) / 2, (1 + delta) / 2]
            achieved = max(abs(p - q) for p, q in zip(predicted_probs, true_probs))
            bound = max(0, delta - tau) / 2
            exhaustive_error = max(exhaustive_error, abs(achieved - bound))
    return {
        "sigma": sigma, "uniform_per_state_tv_tolerance": allowed_error,
        "adaptive_pair_constraints": rows,
        "two_distinct_pairs_fixed_strongest_margin_mean_kl_sum": fixed_kl,
        "two_distinct_pairs_controlled_future_margin_mean_kl_sum": adaptive_kl,
        "capacity_ratio_adaptive_over_fixed": adaptive_kl / fixed_kl,
        "low_clean_scale_example_only": {"assumed_coordinate_std": toy_clean_std,
            "dimension": toy_dim, "representative_distance_sqrt_2D_std": toy_distance,
            "channel_tv": toy_tv, "optimal_equal_prior_accuracy": (1 + toy_tv) / 2,
            "future_gap_assumed": 0.8, "necessary_worst_future_error_lower_bound": (0.8 - toy_tv) / 2},
        "tdwm_exact_recurrence_counterexample": {"time_gap": time_gap,
            "temporal_scale": temporal_scale, "self_target": 1,
            "distant_same_image_target": temporal_target,
            "best_shared_cosine": best_same_cosine,
            "unavoidable_mean_two_term_squared_error": unavoidable_two_term_mean_mse,
            "d8_gap_for_same_state_same_action": 0},
        "side_information_counterexample": bypass,
        "finite_sample_lcb": {"n_per_state": n, "alpha": alpha,
            "per_probability_hoeffding_radius": probability_radius,
            "rows": finite_sample, "warning": "large repeat cost; replace by justified pooled estimator or admit infeasibility"},
        "tightness_grid": {"delta_count": 101, "distance_count": 81,
                            "max_bound_vs_achievable_error_difference": exhaustive_error},
        "novelty_degenerate_case": "deterministic distinct futures with uniform tolerance reduce to a fixed contrastive distance margin",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path(__file__).with_name("third-cpu.json"))
    args = p.parse_args()
    result = {"kind": "synthetic exact checks only", "D7": d7(), "D8": d8()}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output),
        "D7_costs": [result["D7"]["euclidean_costs"], result["D7"]["deep_output_costs"]],
        "D8_capacity_ratio": result["D8"]["capacity_ratio_adaptive_over_fixed"],
        "D8_tightness_max_error": result["D8"]["tightness_grid"]["max_bound_vs_achievable_error_difference"]}))


if __name__ == "__main__":
    main()
