"""Actual four-GPU gradient equivalence and analytic Gaussian-unit check."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import default_collate
from torch.distributions import Normal,kl_divergence
from experiments.metaworld_transition_only_20261109.common import read,save,tensor_sha,now
from experiments.metaworld_transition_only_20261109.data import FactualWindows
from .train import setup
from .model import random_terms,PIXELS


def run(config,output):
    cfg=read(config);device,model,_,_=setup(cfg);model.train();assert torch.cuda.device_count()==4
    data=FactualWindows(cfg['data'],'train',maximum_horizon=60)
    batch={k:v.to(device) for k,v in default_collate([data[i] for i in range(8)]).items()}
    terms=list(random_terms(batch));terms[0]=terms[0].clamp_max(3);terms=tuple(terms)
    before=tensor_sha(model);loss,_=model(batch,terms);loss.backward()
    single={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    parallel=torch.nn.DataParallel(model,device_ids=[0,1,2,3]);other,_=parallel(batch,terms);other=other.mean();other.backward()
    absolute=sum(float((p.grad-single[n]).square().sum()) for n,p in model.named_parameters() if n in single)
    scale=sum(float(v.square().sum()) for v in single.values());relative=(absolute/scale)**.5
    assert torch.allclose(loss,other,rtol=2e-5,atol=1e-5),(loss,other)
    assert relative<3e-4,relative
    shape=(2,128,16,16);p=torch.randn(shape,device=device)*.1;q=torch.randn(shape,device=device)*.1
    lp=torch.randn(shape,device=device)*.1;lq=torch.randn(shape,device=device)*.1
    exact=kl_divergence(Normal(q,lq.exp()),Normal(p,lp.exp())).flatten(1).sum(1)/PIXELS
    found=model.kl(p,lp,q,lq);assert torch.allclose(exact,found,rtol=2e-6,atol=1e-6)
    assert tensor_sha(model)==before
    save(output,dict(passed=True,utc=now(),four_gpu_vs_single_gpu_FP32_loss=[float(loss),float(other)],
        gradient_relative_L2=relative,analytic_gaussian_KL_matches_torch=True,common_nats_per_RGB_dimension=True,
        actual_train_windows=8,maximum_native_future=3,model_unchanged=True,model_updates=0,physics_steps=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.config,a.output)
