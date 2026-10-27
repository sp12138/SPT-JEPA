"""Analytical prerequisites for candidate methods; not world-model benchmarks."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import expit


def gaussian_kl(mean_q, covariance_q, mean_p, covariance_p):
    difference = mean_q - mean_p
    return .5 * (np.trace(np.linalg.solve(covariance_p, covariance_q))
                 + difference @ np.linalg.solve(covariance_p, difference)
                 - len(mean_q) + np.linalg.slogdet(covariance_p)[1]
                 - np.linalg.slogdet(covariance_q)[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261028)
    report = dict(scope="Mathematical prerequisites and negative controls only; zero candidate world-model experiments completed.")
    variances = np.geomspace(.001, 1000, 81)
    uniform = np.array([quad(lambda t: v / ((1-t)**2*v+t*t), 0, 1,
                             epsabs=1e-10, epsrel=1e-10)[0] for v in variances])
    exact = math.pi * np.sqrt(variances) / 2
    assert np.max(np.abs(uniform-exact)/exact) < 1e-9
    nodes, weights = np.polynomial.hermite.hermgauss(128)
    shift = 2.667 * math.sqrt(256/600)
    t = expit(.8 + .8*math.sqrt(2)*nodes + math.log(shift))
    density = weights/math.sqrt(math.pi)
    weighted = np.array([np.sum(density*v/((1-t)**2*v+t*t)) for v in variances])
    weighted_slopes = np.array([2*v*np.sum(density*t*t/((1-t)**2*v+t*t)**2) for v in variances])
    report["I01"] = dict(uniform_quadrature_relative_error=float(np.max(np.abs(uniform-exact)/exact)),
                         current_sampler_gaussian_toy_slope_ratio_min=float(weighted_slopes.min()),
                         current_sampler_gaussian_toy_slope_ratio_max=float(weighted_slopes.max()),
                         flow_sampler=dict(mean=.8,std=.8,shift=shift),
                         interpretation="FM optimum is not a constant multiple of Gaussian entropy over varying residual variance. This does not estimate the real model's likelihood or prove a calibrated replacement.")

    size = 16
    z = rng.normal(size=size)
    matrix = rng.normal(size=(size,size))
    covariance = matrix@matrix.T + np.eye(size)
    matrix2 = rng.normal(size=(size,size))
    prior_covariance = matrix2@matrix2.T + np.eye(size)
    prior_mean = rng.normal(size=size)
    rotation = np.linalg.qr(rng.normal(size=(size,size)))[0]
    transform = rotation @ np.diag(np.geomspace(.2,5,size))
    original = gaussian_kl(z,covariance,prior_mean,prior_covariance)
    transformed = gaussian_kl(transform@z,transform@covariance@transform.T,
                              transform@prior_mean,transform@prior_covariance@transform.T)
    error = abs(original-transformed)
    assert error < 1e-8
    report["I02"] = dict(gaussian_kl_coordinate_transform_abs_error=error,
                         condition="Both posterior and prior, including entry reference distribution, must transform. A fixed unit prior is not invariant under arbitrary linear scaling.")

    length = 8
    sigma = .1
    rho = .8
    coefficient = .9
    mu = np.cumsum(rng.normal(scale=.1,size=length))
    bias = rng.normal(scale=.1,size=length-1)
    q_cov = sigma*sigma*rho**np.abs(np.arange(length)[:,None]-np.arange(length)[None,:])
    lower = np.eye(length)
    for i in range(1,length):
        lower[i,i-1] = -coefficient
    innovation_cov = np.diag([1]+[sigma*sigma]*(length-1))
    lower_inverse = np.linalg.inv(lower)
    p_cov = lower_inverse@innovation_cov@lower_inverse.T
    p_mean = lower_inverse@np.r_[0,bias]
    joint = gaussian_kl(mu,q_cov,p_mean,p_cov)
    entry = .5*(sigma*sigma+mu[0]**2-1-math.log(sigma*sigma))
    chain = entry
    naive = entry
    for i in range(1,length):
        q_variance = sigma*sigma*(1-rho*rho)
        p_variance = sigma*sigma
        residual_mean = mu[i]-coefficient*mu[i-1]-bias[i-1]
        residual_variance = (rho-coefficient)**2*sigma*sigma
        chain += .5*(math.log(p_variance/q_variance)
                     +(q_variance+residual_mean**2+residual_variance)/p_variance-1)
        naive += .5*(residual_mean**2+coefficient**2*sigma*sigma)/p_variance
    assert abs(joint-chain) < 1e-9
    assert abs(joint-naive) > .1
    report["I03"] = dict(full_joint_kl=joint,correct_chain_kl=chain,
                         abs_error=abs(joint-chain),incorrect_independent_formula=naive,
                         interpretation="Correlated posterior noise requires a new conditional KL; keeping the old independent formula misprices the same Gaussian chain.")

    generator = rng.normal(size=(8,3))
    style = rng.normal(size=(8,2))
    projection = np.eye(8)-style@np.linalg.solve(style.T@style,style.T)
    original_state = rng.normal(size=3)
    original_style = rng.normal(size=2)
    other_style = rng.normal(size=2)
    style_difference = np.linalg.norm(projection@(generator@original_state+style@original_style)
                                      -projection@(generator@original_state+style@other_style))
    assert style_difference < 1e-10
    assert np.linalg.matrix_rank(projection@generator)==3
    report["I04"] = dict(linear_style_removal_error=float(style_difference),
                         retained_dynamics_rank=int(np.linalg.matrix_rank(projection@generator)),
                         caveat="Requires separable subspaces in this toy; arbitrary anchor subtraction can remove absolute geometry. No image canonicalizer has been validated.")

    fm_gradient = np.array([1.,0.])
    rate_gradient = np.array([-10.,0.])
    projected = rate_gradient-min(0.,float(rate_gradient@fm_gradient))/(fm_gradient@fm_gradient)*fm_gradient
    direction = -(fm_gradient+projected)
    fm_change = float(fm_gradient@direction)
    original_objective_change = float((fm_gradient+rate_gradient)@direction)
    assert fm_change < 0 and original_objective_change > 0
    report["I06"] = dict(first_order_fm_change=fm_change,
                         first_order_original_priced_objective_change=original_objective_change,
                         interpretation="The constrained update can improve FM while increasing the original priced objective. It must be evaluated as a changed optimization rule.")

    expected_squared_residual = 4.
    optimum = expected_squared_residual
    grid = np.geomspace(.01,10000,10001)
    proper = expected_squared_residual/(2*grid)+.5*np.log(grid)
    improper = expected_squared_residual/(2*grid)
    assert abs(math.log(grid[np.argmin(proper)]/optimum)) < .002
    assert np.argmin(improper)==len(grid)-1
    report["I07"] = dict(analytic_optimum_variance=optimum,
                         numeric_optimum_variance=float(grid[np.argmin(proper)]),
                         no_log_penalty_optimum="infinity",
                         caveat="Variance predictor must not read target pixels; this regression identity does not make weighted FM an endpoint NLL.")

    report["I08"] = dict(latent_transition_gain=.95,encode_decode_gain=1.2,
                         pixel_feedback_gain=.95*1.2,
                         condition="A toy can have stable latent recurrence and unstable encode/decode feedback. If E*D=I, the two recurrences are equivalent. Neither condition is established for trained Native by this toy.")

    dimensions = 16
    action_dimensions = 9
    jacobian = rng.normal(size=(dimensions,action_dimensions))
    matrix = rng.normal(size=(dimensions,dimensions))
    training_cov = matrix@matrix.T+np.eye(dimensions)
    difference = rng.normal(size=dimensions)
    rotation = np.linalg.qr(rng.normal(size=(dimensions,dimensions)))[0]
    transform = rotation@np.diag(np.geomspace(.1,10,dimensions))
    regularizer = .1
    gramian = jacobian@jacobian.T+regularizer*training_cov
    cost = float(difference@np.linalg.solve(gramian,difference))
    transformed_jacobian = transform@jacobian
    transformed_gramian = transformed_jacobian@transformed_jacobian.T + regularizer*transform@training_cov@transform.T
    transformed_difference = transform@difference
    other = float(transformed_difference@np.linalg.solve(transformed_gramian,transformed_difference))
    assert abs(cost-other) < 1e-7
    report["I10"] = dict(control_energy_original=cost,control_energy_transformed=other,
                         invariant_absolute_error=abs(cost-other),
                         euclidean_cost_original=float(difference@difference),
                         euclidean_cost_transformed=float(transformed_difference@transformed_difference),
                         caveat="Ridge must use covariant C, not a fixed identity. Local control Jacobians and PushT benefit remain untested.")

    report["not_checked_empirically"] = ["I01","I02","I03","I04","I05","I06","I07","I08","I09","I10"]
    (out/"mathematical-checks.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    axes[0].loglog(variances,uniform,label="Optimal FM, uniform time")
    axes[0].loglog(variances,weighted,label="Optimal FM, current time sampler")
    axes[0].set(xlabel="Conditional Gaussian variance",ylabel="Bayes velocity MSE",title="A Gaussian counterexample to fixed units")
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].semilogx(variances,weighted_slopes)
    axes[1].set(xlabel="Conditional Gaussian variance",ylabel="d(FM risk) / d(Gaussian entropy)",title="Conversion factor depends on variance")
    fig.supxlabel("Analytical toy; not measured world-model likelihood or empirical method performance.",fontsize=9)
    for suffix in ["png","svg"]:
        fig.savefig(out/("flow-risk-units."+suffix),dpi=160,bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(dict(checks=list(report),output=str(out),world_model_experiments=0),indent=2))


if __name__=="__main__":
    main()
