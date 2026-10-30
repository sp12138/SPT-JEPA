"""CPU checks for the final two admission attempts; no world-model training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def signature(path):
    increments=path[1:]-path[:-1]
    total=torch.zeros(path.shape[-1],dtype=path.dtype)
    area=torch.zeros(path.shape[-1],path.shape[-1],dtype=path.dtype)
    for increment in increments:
        area=area+.5*(torch.outer(total,increment)-torch.outer(increment,total))
        total=total+increment
    return total,area


def transition_constraint():
    rng=np.random.default_rng(20261031)
    m=.5
    # The midpoint quantization is a deterministic approximation to U[0,1].
    x=(np.arange(100000,dtype=float)+.5)/100000
    true_next=np.ones_like(x)
    prediction=1+m*(x-.5)
    variance=np.var(x)
    mse=np.mean((prediction-true_next)**2)
    bound=m*m*variance
    assert abs(mse-bound)<1e-14
    # A generic norm-bounded residual also meets the lower distance bound.
    c=.5
    matrix=np.array([[.1,-.2],[.25,.05]])
    matrix=matrix*(c/np.linalg.svd(matrix,compute_uv=False)[0])
    left,right=rng.normal(size=(2,10000,2))
    moved=(left-right)@(np.eye(2)+matrix).T
    margin=np.linalg.norm(moved,axis=-1)-(1-c)*np.linalg.norm(left-right,axis=-1)
    assert margin.min()>=-1e-12
    return dict(co_lipschitz_m=m,uniform_variance=float(variance),best_centered_linear_contact_mse=float(mse),
                discrete_variance_lower_bound=float(bound),continuous_lower_bound=m*m/12,
                norm_bounded_residual_min_distance_margin=float(margin.min()),
                two_equal_goals_min_pair_average_mse=m*m/4,
                interpretation='A non-erasing transition conflicts with a many-to-one observed contact map under the same fully-read state/goal contract.')


def path_area():
    torch.manual_seed(20261031)
    max_error=0.
    for _ in range(100):
        increments=torch.randn(8,3,dtype=torch.float64)
        path=torch.cat([torch.zeros(1,3,dtype=torch.float64),increments.cumsum(0)])
        split=4
        r,a=signature(path);r1,a1=signature(path[:split+1]);r2,a2=signature(path[split:])
        expected=a1+a2+.5*(torch.outer(r1,r2)-torch.outer(r2,r1))
        max_error=max(max_error,float((r-r1-r2).abs().max()),float((a-expected).abs().max()))
    left=torch.tensor([[0.,0.],[1.,0.],[1.,1.]],dtype=torch.float64)
    right=torch.tensor([[0.,0.],[0.,1.],[1.,1.]],dtype=torch.float64)
    lr,la=signature(left);rr,ra=signature(right)
    torch.testing.assert_close(lr,rr)
    target=torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[0.,0.]],dtype=torch.float64)
    tr,ta=signature(target)
    predicted=torch.zeros_like(target,requires_grad=True)
    pr,pa=signature(predicted)
    loss=(pr-tr).square().sum()+(pa[0,1]-ta[0,1]).square()
    gradient=torch.autograd.grad(loss,predicted)[0]
    direct=predicted.square().sum()*0+(predicted-target).square().mean()
    direct_gradient=torch.autograd.grad(direct,predicted)[0]
    assert max_error<1e-12 and float(loss)==1 and float(gradient.norm())==0
    assert float(direct_gradient.norm())>0
    # Tiny nonconstant forecasts open area gradients only proportionally to
    # perturbation amplitude; the zero-gradient example is not global inability.
    perturb=torch.randn_like(target)
    perturb[0]=0;perturb[-1]=0
    rows=[]
    for epsilon in [1e-2,1e-3,1e-4,1e-5]:
        guess=(epsilon*perturb).requires_grad_()
        dr,da=signature(guess)
        objective=(dr-tr).square().sum()+(da[0,1]-ta[0,1]).square()
        norm=torch.autograd.grad(objective,guess)[0].norm()
        rows.append(dict(epsilon=epsilon,loss=float(objective.detach()),gradient_norm=float(norm)))
    return dict(chen_identity_max_error=max_error,same_endpoint=lr.tolist(),
                opposite_path_area=[float(la[0,1]),float(ra[0,1])],
                closed_loop_area=float(ta[0,1]),carry_signature_loss=float(loss.detach()),
                carry_signature_gradient_norm=float(gradient.norm()),
                carry_pointwise_loss=float(direct.detach()),carry_pointwise_gradient_norm=float(direct_gradient.norm()),
                small_perturbations=rows,
                interpretation='A genuine path distinction does not guarantee a learning gradient at carry initialization; quadratic area supervision is exactly stationary for this closed-loop example.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    result=dict(transition_constraint=transition_constraint(),path_area=path_area(),scope='Constructive CPU checks only; no training, task success, novelty or universal impossibility claim.')
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
