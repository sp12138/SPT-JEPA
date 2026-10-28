import unittest
from types import SimpleNamespace
import numpy as np
import torch
from as_jepa_tools.probes import add_posterior_noise, noisy_channel_bounds
from as_jepa_tools.collapse_audit import variable_dimension_rate


class NoiseEvaluationContracts(unittest.TestCase):
    def test_probe_noise_uses_the_declared_anisotropic_posterior(self):
        torch.manual_seed(41)
        model=SimpleNamespace(sigma0=.1,noise_diagonal=torch.tensor([.25,1.,4.]))
        draws=add_posterior_noise(torch.zeros(131072,3),model)
        torch.testing.assert_close(draws.var(0),model.noise_diagonal*.01,rtol=.02,atol=1e-5)

    def test_channel_bound_matches_full_gaussian_logdet(self):
        matrix=np.array([[1.,.3,-.2],[.3,2.,.1],[-.2,.1,.7]])
        diagonal=np.array([.25,1.,4.]);sigma=.1
        result=noisy_channel_bounds(matrix,sigma,diagonal)
        sign,logdet=np.linalg.slogdet(np.eye(3)+np.diag(1/(sigma*sigma*diagonal))@matrix)
        self.assertEqual(sign,1);self.assertAlmostEqual(result["covariance_bound"],.5*logdet,places=11)
        self.assertGreaterEqual(result["diagonal_bound"],result["covariance_bound"])

    def test_irreducible_markov_variance_kl_does_not_make_a_zero_code_active(self):
        rho=.8;sigma=.1
        q=torch.distributions.Normal(torch.zeros(2,7,64),torch.full((2,7,64),sigma*(1-rho*rho)**.5))
        p=torch.distributions.Normal(torch.zeros(2,7,64),torch.full((2,7,64),sigma))
        exact=torch.distributions.kl_divergence(q,p)
        constant=.5*(-rho*rho-np.log1p(-rho*rho))
        rate=variable_dimension_rate(torch.zeros(2,8,64),exact,constant)
        torch.testing.assert_close(rate,torch.zeros_like(rate),rtol=0,atol=1e-6)


if __name__=="__main__":
    unittest.main()
