"""Analytic scalar-coupling bias and a transport instability counterexample."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    rng=np.random.default_rng(1042);n=400000;sigma=.1;rho=.8
    a=np.array([[0.,-1.],[1.,0.]])
    noise=sigma*rng.standard_normal((n,2))
    mismatch=noise@a.T-rho*noise
    actual=float(np.square(mismatch).sum(1).mean())
    formula=float(sigma*sigma*np.square(a-rho*np.eye(2)).sum())
    assert abs(actual-formula)<1e-4
    # Full expected training objective: physical fit plus scalar coupling noise.
    # At zero reconstruction error the optimal linear predictor is still biased.
    signal_variance=.04
    optimum=(signal_variance*a+sigma*sigma*rho*np.eye(2))/(signal_variance+sigma*sigma)
    error=float(np.square(optimum-a).sum())
    assert error>.1
    mu=rng.normal(size=(128,2));following=np.tanh(mu@a.T)+.1
    prev=mu+.2*rng.normal(size=mu.shape)
    f=lambda z:np.tanh(z@a.T)+.1
    correction=following-f(mu)
    conditional=following+f(prev)-f(mu)
    identity=float(np.max(np.abs(conditional-f(prev)-correction)))
    assert identity<1e-14
    # Correct clean dynamics at fixed point 0 does not imply stable noise.
    # F(z)=2z+z^3 expands even infinitesimal perturbations; no innovation needed.
    z=np.array([.1]);amplitudes=[float(z[0])]
    for _ in range(4):
        z=2*z+z**3;amplitudes.append(float(z[0]))
    assert amplitudes[-1]>2
    result=dict(rotation_scalar_noise_energy_mc=actual,rotation_scalar_noise_energy_formula=formula,
        correct_dynamics_transport_conditional_kl=0.,scalar_objective_predictor=optimum.tolist(),
        scalar_objective_predictor_frobenius_error=error,nonlinear_mean_difference_identity_error=identity,
        unstable_transport_amplitudes=amplitudes,
        scope='Population linear-risk optimum and exact nonlinear identity; instability is an explicit counterexample. No evidence of task superiority or priority.')
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
