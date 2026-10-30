"""CPU constructive separation: fixed features, a bounded learned chart,
semantic-identity adaptation, and a predictive but information-losing encoder.
This is a mathematical control, not a world-model benchmark.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


def h(x):
    # Globally 2-Lipschitz, agrees with x^2 on [-1,1].
    return np.where(abs(x) <= 1, x*x, 2*abs(x)-1)


def chart(u, theta):
    z = u.copy(); z[:, 1] -= theta*h(u[:, 0])
    return z


def data(rng, n):
    s = rng.uniform(-1, 1, (n, 2))
    a = rng.uniform(-1, 1, (n, 1))
    target = np.stack([.7*s[:, 0]+.25*a[:, 0], .85*s[:, 1]+.3*s[:, 0]], -1)
    u, future = s.copy(), target.copy()
    u[:, 1] += .1*h(s[:, 0]); future[:, 1] += .1*h(target[:, 0])
    return u, a, future, s, target


def linear_predictor(train, test, theta):
    u, a, future, _, _ = train
    tu, ta, tfuture, _, _ = test
    x = np.concatenate([chart(u, theta), a, np.ones((len(a), 1))], -1)
    y = chart(future, theta)
    weight = np.linalg.lstsq(x, y, rcond=None)[0]
    tx = np.concatenate([chart(tu, theta), ta, np.ones((len(ta), 1))], -1)
    error = tx @ weight-chart(tfuture, theta)
    return float(np.mean(error**2)), weight


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); args = p.parse_args()
    rng = np.random.default_rng(20261031)
    train, test = data(rng, 8192), data(rng, 8192)
    # Every candidate chart has global residual Lipschitz constant <.5.
    optimized = minimize_scalar(lambda theta: linear_predictor(train, train, theta)[0],
                                bounds=(-.249, .249), method="bounded", options={"xatol": 1e-12})
    theta = float(optimized.x)
    frozen_error, _ = linear_predictor(train, test, 0.)
    learned_error, learned_weight = linear_predictor(train, test, theta)
    assert abs(theta-.1) < 1e-6 and learned_error < 1e-14 and frozen_error > 1e-6
    u, _, _, _, _ = train
    x, y = rng.uniform(-2, 2, (20000, 2)), rng.uniform(-2, 2, (20000, 2))
    c = 2*abs(theta)
    ratio = np.linalg.norm(chart(x, theta)-chart(y, theta), axis=1)/np.linalg.norm(x-y, axis=1)
    assert ratio.min() >= 1-c-1e-12 and ratio.max() <= 1+c+1e-12
    z = chart(u, theta); recovered = z.copy(); recovered[:, 1] += theta*h(z[:, 0])
    assert np.max(abs(recovered-u)) < 1e-12
    # Identity semantic teacher prefers theta=0, unlike the dynamics objective.
    identity_teacher_error = float(np.mean((chart(u, theta)-u)**2))
    assert identity_teacher_error > 0
    # Dropping coordinate 2 gives exact latent prediction, while losing s2.
    tu, ta, tfuture, true_state, _ = test
    collapsed_prediction = .7*tu[:, 0]+.25*ta[:, 0]
    collapsed_error = float(np.mean((collapsed_prediction-tfuture[:, 0])**2))
    assert collapsed_error < 1e-25
    # Explicit two-state distinction disappears only in the lossy encoder.
    pair = np.array([[.2, -.4], [.2, .4]])
    assert np.linalg.norm(chart(pair, theta)[0]-chart(pair, theta)[1]) > .79
    assert pair[0, 0] == pair[1, 0]
    result = dict(scope="Constructive mathematical control, CPU only, no benchmark or novelty proof",
                  learned_theta=theta, known_inverse_shear_theta=.1,
                  frozen_carrier_linear_predictor_test_mse=frozen_error,
                  learned_chart_linear_predictor_test_mse=learned_error,
                  learned_predictor=learned_weight.tolist(),
                  identity_semantic_teacher_mse_at_dynamics_optimum=identity_teacher_error,
                  lossy_one_coordinate_latent_prediction_mse=collapsed_error,
                  lossy_missing_coordinate_variance=float(np.var(true_state[:, 1])),
                  global_residual_lipschitz_bound=c, tested_pair_distance_ratio_range=[float(ratio.min()), float(ratio.max())],
                  exact_recovery_error=float(np.max(abs(recovered-u))))
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
