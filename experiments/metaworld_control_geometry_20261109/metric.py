"""Goal discrepancy in the world model's own control geometry.

For an affine endpoint map x=A u+b, unit Gaussian control gives endpoint
covariance A A.T. Its inverse measures minimum squared control effort, rather
than pixel area. Here a finite rollout bank estimates nonlinear covariance.
An isotropic floor is measured from factual TRAIN prediction residuals only.
This is an empirical planning metric, not a proven nonlinear control distance.
"""
import math
import torch


class ControlGeometry:
    def __init__(self,reference,variance):
        flat=reference.flatten(1).double()
        assert len(flat)>1 and variance>0
        self.deviations=(flat-flat.mean(0))/math.sqrt(len(flat)-1)
        self.variance=float(variance);self.dimensions=flat.shape[1]
        gram=self.deviations@self.deviations.T
        self.factor=torch.linalg.cholesky(gram+self.variance*torch.eye(len(flat),device=flat.device,dtype=flat.dtype))

    def __call__(self,future,goal):
        difference=(future-goal).flatten(1).double()
        projected=torch.linalg.solve_triangular(self.factor,self.deviations@difference.T,upper=False)
        squared=difference.square().sum(1)-projected.square().sum(0)
        return squared.clamp_min(0)/(self.variance*self.dimensions)


def cpu_audit():
    assert not torch.cuda.is_available()
    generator=torch.Generator().manual_seed(20261109)
    reference=torch.randn(7,6,generator=generator,dtype=torch.float64)
    future=torch.randn(4,6,generator=generator,dtype=torch.float64)
    goal=torch.randn(1,6,generator=generator,dtype=torch.float64)
    metric=ControlGeometry(reference,.03)
    difference=future-goal
    covariance=metric.deviations.T@metric.deviations+.03*torch.eye(6,dtype=torch.float64)
    direct=(difference*torch.linalg.solve(covariance,difference.T).T).mean(1)
    assert torch.allclose(metric(future,goal),direct,atol=1e-11,rtol=1e-11)
    shifted=ControlGeometry(reference*3+11,.03*9)
    assert torch.allclose(metric(future,goal),shifted(future*3+11,goal*3+11),atol=1e-11,rtol=1e-11)
    # Large directly-actuated coordinate versus a small passive coordinate.
    reference=torch.tensor([[a,b] for a in [-.5,.5] for b in [-.01,.01]],dtype=torch.float64)
    candidates=torch.tensor([[0.,.05],[.1,0.]],dtype=torch.float64)
    zero=torch.zeros(1,2,dtype=torch.float64)
    pixel=candidates.square().mean(1)
    intrinsic=ControlGeometry(reference,.0001)(candidates,zero)
    assert pixel[0]<pixel[1] and intrinsic[1]<intrinsic[0]
    # If no output changes with controls, the metric retains ordinary RGB ranks.
    constant=ControlGeometry(torch.zeros_like(reference),.03)
    assert torch.allclose(constant(candidates,zero),pixel/.03)
    return dict(passed=True,explicit_full_covariance_inverse=True,affine_unit_invariance=True,
        passive_vs_actuated_toy=dict(pixel_cost=pixel.tolist(),intrinsic_cost=intrinsic.tolist()),
        zero_control_covariance_retains_pixel_ranks=True,model_updates=0,
        limits='Cannot infer an object response that the world model never predicts;cannot remove finite-horizon unreachability or guarantee task success.')
