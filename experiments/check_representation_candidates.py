"""Deterministic NumPy checks of claims AND failure cases; no task benchmark.

Run with Python/NumPy on CPU. No torch, checkpoints, network or dataset needed.
The algebraic properties are not asserted to be novel.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def response_coefficients():
    xs, actions = np.array([-1., 1.]), np.array([-1., 1.])
    # A command-only factor plus an initial-state-dependent response.
    y = np.array([[[2*a, x+a*x] for a in actions] for x in xs])
    baseline = y.mean(axis=0)
    residual = y - baseline[None]
    e0, e1 = residual.mean(1), (residual*actions[None, :, None]).mean(1)
    assert np.allclose(e0[:, 1], xs) and np.allclose(e1[:, 1], xs)
    assert np.allclose(e0[:, 0], 0) and np.allclose(e1[:, 0], 0)
    # A free action-conditioned decoder can absorb subtracting B(a) exactly.
    prediction = .3 * y + .7
    ordinary = np.mean((prediction-y)**2)
    shifted = np.mean(((prediction-baseline[None])-residual)**2)
    assert abs(ordinary-shifted) < 1e-14
    # Confounding A=X makes every target predictable from A: residual is zero.
    confounded_y = np.stack([2*xs, xs], -1)
    confounded_baseline = confounded_y.copy()
    assert np.max(abs(confounded_y-confounded_baseline)) == 0
    return dict(zero_order=e0.tolist(), first_order=e1.tolist(),
                collapsed_coefficient_risk=float(np.mean(np.concatenate([e0, e1], -1)**2)),
                decoder_reparameterization_loss_gap=float(abs(ordinary-shifted)),
                confounded_residual_norm=0.)


def observable_closure():
    # Three hidden coordinates enter the observed coordinate at successive times.
    a = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    c = np.array([[1., 0., 0.]])
    bank = c.copy()
    ranks = [int(np.linalg.matrix_rank(bank))]
    for _ in range(2):
        pullback = bank @ a
        residual = pullback - pullback @ np.linalg.pinv(bank) @ bank
        _, singular, right = np.linalg.svd(residual, full_matrices=False)
        bank = np.concatenate([bank, right[singular > 1e-10]], 0)
        ranks.append(int(np.linalg.matrix_rank(bank)))
    assert ranks == [1, 2, 3]
    # A constant dictionary closes perfectly but fails the fixed observable.
    x = np.eye(3)
    constant_state = np.zeros((3, 1))
    fixed_observable_error = np.mean((x @ c.T-(x @ c.T).mean(0))**2)
    assert np.linalg.norm(constant_state-constant_state) == 0 and fixed_observable_error > 0
    return dict(observable_ranks=ranks, constant_dictionary_closure_error=0.,
                constant_dictionary_observable_error=float(fixed_observable_error),
                novelty_status="Known Krylov/model-error basis construction; not a new algorithm.")


def bridge_curvature():
    lam, sigma, beta, eta = 1., .1, .25, .4
    def objective(e, w, coupling):
        return 1 - 2*coupling*e*w + lam*e*e*w*w + sigma*sigma*w*w + .5*beta*lam*e*e
    h = np.array([[beta*lam, -2*eta], [-2*eta, 2*sigma*sigma]])
    values, vectors = np.linalg.eigh(h)
    step = .1*vectors[:, 0]
    before, after = objective(0, 0, eta), objective(*step, eta)
    assert values[0] < 0 and after < before
    assert objective(step[0], 0, eta) >= before and objective(0, step[1], eta) >= before
    stable_eta = .001
    stable_h = np.array([[beta*lam, -2*stable_eta], [-2*stable_eta, 2*sigma*sigma]])
    assert np.linalg.eigvalsh(stable_h)[0] > 0
    # Congruence preserves existence, not the coordinate-specific chosen vector.
    change = np.diag([100., .01])
    assert np.linalg.eigvalsh(change.T @ h @ change)[0] < 0
    return dict(eigenvalues=values.tolist(), original_objective_before=before,
                paired_step_objective=after, encoder_only=objective(step[0], 0, eta),
                readout_only=objective(0, step[1], eta),
                stable_case_eigenvalues=np.linalg.eigvalsh(stable_h).tolist())


def moments(nodes, weights, count):
    return (weights[:, None] * nodes[:, None]**np.arange(count)[None]).sum(0)


def recover_nodes(moment, n):
    h0 = np.array([[moment[i+j] for j in range(n)] for i in range(n)])
    h1 = np.array([[moment[i+j+1] for j in range(n)] for i in range(n)])
    return np.linalg.eigvals(np.linalg.solve(h0, h1)), np.linalg.cond(h0)


def match_error(nodes, recovered):
    return min(float(np.max(abs(nodes-recovered[list(p)]))) for p in itertools.permutations(range(len(nodes))))


def spectral_address():
    nodes = np.exp(1j*np.array([.3, 1.4]))
    coefficient = np.poly(nodes)
    errors, conditions, noisy_errors = [], [], []
    for small_mass in [1., .1, 1e-4, 1e-8]:
        moment = moments(nodes, np.array([1., small_mass]), 4)
        recovered, condition = recover_nodes(moment, 2)
        error = match_error(nodes, recovered)
        assert error < 1e-6
        perturbed = moment.copy()
        perturbed[-1] += 1e-6*(1+1j)
        noisy, _ = recover_nodes(perturbed, 2)
        errors.append(error); conditions.append(float(condition)); noisy_errors.append(match_error(nodes, noisy))
    assert conditions[-1] > 1e6*conditions[0]
    assert noisy_errors[-1] > 1e5*noisy_errors[0]
    delta = .27
    moved = np.poly(nodes*np.exp(1j*delta))
    assert np.max(abs(moved-coefficient*np.exp(1j*delta*np.arange(3)))) < 1e-14
    # Both atoms have equal coefficient sensitivity, regardless of image mass.
    eps = 1e-6
    changes = []
    weighted_changes = []
    for j in range(2):
        changed = nodes.copy(); changed[j] *= np.exp(1j*eps)
        changes.append(float(np.linalg.norm(np.poly(changed)-coefficient)/eps))
        weighted_changes.append(float(np.linalg.norm(moments(changed, np.array([1., .01]), 4)
                                                     - moments(nodes, np.array([1., .01]), 4))/eps))
    assert abs(changes[0]/changes[1]-1) < 1e-6
    assert abs(weighted_changes[0]/weighted_changes[1]-100) < 1e-4
    # Marginal x/y moments cannot identify 2D point pairing.
    diagonal, cross = np.array([[0., 0.], [1., 1.]]), np.array([[0., 1.], [1., 0.]])
    marginals = lambda points: np.concatenate([np.exp(1j*points[:, 0]).sum()[None], np.exp(1j*points[:, 1]).sum()[None]])
    assert np.allclose(marginals(diagonal), marginals(cross))
    mixed_gap = abs(np.exp(1j*diagonal.sum(1)).sum()-np.exp(1j*cross.sum(1)).sum())
    assert mixed_gap > .1
    return dict(noiseless_node_errors=errors, hankel_condition_numbers=conditions,
                perturbed_node_errors=noisy_errors, amplitude_free_coefficient_sensitivity=changes,
                weighted_moment_sensitivity=weighted_changes, marginal_2d_ambiguity=True,
                mixed_moment_distinction=float(mixed_gap),
                novelty_status="Prony/FRI algebra is existing; world-state application is an unvalidated hypothesis.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = dict(scope="Algebra and counterexample checks only; not world-model training or benchmark evidence.",
                  A=response_coefficients(), B=observable_closure(), C=bridge_curvature(), D=spectral_address())
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
