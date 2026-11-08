"""Frozen factual world, unchanged objective/budget; fixed temporal covariance.

This tests a known planning principle, not a new learned algorithm or an iCEM
reproduction. No demonstrated controls or protected evaluation paths are fitted.
"""
import argparse
from pathlib import Path
import traceback
import torch
from experiments.metaworld_transition_only_20261109 import evaluate as original
from experiments.metaworld_transition_only_20261109.common import save,now
from experiments.metaworld_horizon_match_20261109.model import build

CORRELATED=False


def mixing(device):
    n=12
    k=torch.arange(n,device=device,dtype=torch.float64)
    t=torch.arange(n,device=device,dtype=torch.float64)+.5
    basis=(2/n)**.5*torch.cos(torch.pi*k[:,None]*t[None,:]/n)
    basis[0]/=2**.5
    covariance=basis.T@torch.diag(1/(1+k.square()))@basis
    diagonal=covariance.diag().sqrt()
    covariance=covariance/diagonal[:,None]/diagonal[None,:]
    return torch.linalg.cholesky(covariance).float()


@torch.no_grad()
def plan(model,hidden,image,goal,generator,horizon,warm=None,particles=64,iterations=4):
    assert horizon in [30,60] and horizon%5==0
    blocks=horizon//5
    mean=image.new_zeros(blocks,4) if warm is None else warm.reshape(blocks,5,4).mean(1)
    std=torch.ones_like(mean);best=None;best_score=float('inf');first_bank=None
    transform=mixing(image.device) if CORRELATED else None
    for iteration in range(iterations):
        raw=torch.randn(particles,12,4,device=image.device,generator=generator)
        if transform is not None:raw=torch.einsum('ij,nja->nia',transform,raw)
        candidates=(mean[None]+std[None]*raw[:,:blocks]).clamp(-1,1)
        candidates[0]=mean
        if best is not None:candidates[1]=best
        if iteration==0:first_bank=__import__('hashlib').sha256(candidates[:,:6].cpu().numpy().tobytes()).hexdigest()
        state=hidden.expand(particles,-1,-1,-1).clone();future=image.expand(particles,-1,-1,-1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for t in range(horizon):state,future=model.predict(state,future,candidates[:,t//5])
        costs=(future-goal).square().mean((1,2,3))
        assert torch.isfinite(costs).all()
        values,indices=costs.topk(min(10,particles),largest=False)
        elite=candidates[indices]
        if float(values[0])<best_score:best_score=float(values[0]);best=elite[0].clone()
        mean=elite.mean(0);std=elite.std(0,unbiased=False).clamp_min(1e-6)
    return best[:,None].expand(-1,5,-1).reshape(horizon,4),dict(predicted_terminal_goal_mse=best_score,
        first_bank_prefix_sha256=first_bank,proposal='fixed_DCT_covariance' if CORRELATED else 'original_independent')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['config','manifest','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--kind',choices=['white','correlated'],required=True)
    p.add_argument('--budget',type=int,default=500);a=p.parse_args()
    CORRELATED=a.kind=='correlated';original.build=build;original.plan=plan
    torch.cuda.set_per_process_memory_fraction(.30)
    try:original.run(a.config,12000,a.manifest,a.output,60,15,a.budget)
    except BaseException:
        save(a.output/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise
