"""Actual four-GPU gradient equivalence and analytic Gaussian-unit check."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import default_collate
from torch.distributions import Normal,kl_divergence
from experiments.metaworld_transition_only_20261109.common import read,save,tensor_sha,now
from experiments.metaworld_transition_only_20261109.data import FactualWindows
from .train import setup
from .model import random_terms
from .data import EndpointWindows,add_endpoint
PIXELS=3*128*128


def run(config,output):
    cfg=read(config);device,model,_,_=setup(cfg);model.train();assert torch.cuda.device_count()==4
    data=EndpointWindows(cfg['data'],'train',maximum_horizon=60)
    batch={k:v.to(device) for k,v in default_collate([data[i] for i in range(8)]).items()}
    terms=list(random_terms(batch));terms[0]=terms[0].clamp_max(3);terms=tuple(terms);add_endpoint(data,batch,terms[0],device)
    before=tensor_sha(model);loss,_=model(batch,terms);loss.backward()
    single={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    parallel=torch.nn.DataParallel(model,device_ids=[0,1,2,3]);other,_=parallel(batch,terms);other=other.mean();other.backward()
    absolute=sum(float((p.grad-single[n]).square().sum()) for n,p in model.named_parameters() if n in single)
    scale=sum(float(v.square().sum()) for v in single.values());relative=(absolute/scale)**.5
    assert torch.allclose(loss,other,rtol=2e-5,atol=1e-5),(loss,other)
    assert relative<3e-4,relative
    assert tensor_sha(model)==before
    save(output,dict(passed=True,utc=now(),four_gpu_vs_single_gpu_FP32_loss=[float(loss),float(other)],
        gradient_relative_L2=relative,
        actual_train_windows=8,maximum_native_future=3,model_unchanged=True,model_updates=0,physics_steps=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.config,a.output)
