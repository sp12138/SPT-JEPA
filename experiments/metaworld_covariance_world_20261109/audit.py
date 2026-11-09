"""Actual four-GPU equivalence and exact Gaussian math gates, zero SGD."""
import torch
from torch.utils.data import default_collate
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,now
from experiments.metaworld_joint_flow_20261109.data import EndpointWindows,add_endpoint
from experiments.metaworld_boundary_world_20261109.model import random_terms
from .train import setup,gradients
from .model import gaussian_nll


def run():
    root=BASE/'runs/metaworld_covariance_world_20261109';p=read(root/'plan.json')
    cfg=read(next(j['config'] for j in p['jobs'] if j['mode']=='lowrank'))
    device,model,_,_=setup(cfg)
    torch.manual_seed(1040001)
    r=torch.randn(2,3,3,2,device=device,dtype=torch.float64)*.2
    d=torch.rand_like(r)*.1+.01;u=torch.randn(2,18,4,device=device,dtype=torch.float64)*.1
    exact=torch.distributions.LowRankMultivariateNormal(torch.zeros(2,18,device=device,dtype=torch.float64),u,d.flatten(1))
    ref=-exact.log_prob(r.flatten(1))/18;actual=gaussian_nll(r,d,u)
    error=float((actual-ref).abs().max());assert error<1e-5
    # Diagonal limit and rank-column rotation leave the same Gaussian density.
    zero=gaussian_nll(r,d,torch.zeros_like(u));diag=gaussian_nll(r,d)
    assert torch.allclose(zero,diag,atol=1e-6,rtol=1e-6)
    rotation=torch.linalg.qr(torch.randn(4,4,device=device,dtype=torch.float64))[0]
    rotated=gaussian_nll(r,d,u@rotation);assert torch.allclose(rotated,actual,atol=1e-6,rtol=1e-6)
    data=EndpointWindows(cfg['data'],'train',maximum_horizon=60)
    batch={k:v.to(device) for k,v in default_collate([data[i*193] for i in range(64)]).items()}
    terms=random_terms(batch);add_endpoint(data,batch,terms[0],device)
    loss,_=model(batch,terms);loss.backward();gradients(model,positive=True)
    one=torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad and p.grad is not None]).clone()
    model.zero_grad(set_to_none=True);parallel=torch.nn.DataParallel(model,device_ids=[0,1,2,3]);loss4,_=parallel(batch,terms);loss4=loss4.mean();loss4.backward()
    four=torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad and p.grad is not None])
    relative=float((one-four).norm()/one.norm());assert relative<1e-4
    assert all(p.grad is None for p in model.parent.parameters())
    save(root/'FOUR_GPU_AUDIT.json',dict(passed=True,utc=now(),Gaussian_exact_FP64_reference_max_abs_error=error,
        diagonal_limit_pass=True,rank_rotation_invariance_pass=True,four_gpu_FP32_gradient_relative_L2=relative,
        frozen_all_mean_parent_gradients_absent=True,optimizer_updates=0,source='SamplelevelconditionalcovarianceheadONLY. No goals/actionlabels/batchstats.'))


if __name__=='__main__':run()
