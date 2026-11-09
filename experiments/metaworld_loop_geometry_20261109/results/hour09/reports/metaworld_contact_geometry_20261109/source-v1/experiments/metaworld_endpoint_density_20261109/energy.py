"""Finite reverse-Gaussian diffusion NELBO, up to action-constant prior.

Any denoiser defines this normalized reverse chain. It is a likelihood bound,
not exact likelihood. Training flow matching need not minimize this very bound.
The mean control is the identical network trained at pure-noise time only.
"""
import math
import torch

STEPS=64
MINIMUM=-10.
MAXIMUM=10.
QUERIES=5


def log_discretized(target,mean,scale):
    low=(target-mean-1/255)/scale;high=(target-mean+1/255)/scale
    mirror=(low+high)>0
    a,b=torch.where(mirror,-high,low),torch.where(mirror,-low,high)
    upper,lower=torch.special.log_ndtr(b),torch.special.log_ndtr(a)
    inside=upper+torch.log(-torch.expm1((lower-upper).clamp_max(-torch.finfo(torch.float32).eps)))
    return torch.where(target<=-1+1e-6,torch.special.log_ndtr(high),
        torch.where(target>=1-1e-6,torch.special.log_ndtr(-low),inside))


@torch.no_grad()
def cost(renderer,hidden,current,goal,duration,mode,seed):
    n=len(hidden);generator=torch.Generator(device=hidden.device).manual_seed(seed)
    # Common random numbers across candidates and modes. Four strata form an
    # unbiased estimate of the64 transition KLs; fifth query is the endpoint.
    indices=torch.arange(4,device=hidden.device)*16+torch.randint(16,(4,),device=hidden.device,generator=generator)
    noises=torch.randn(QUERIES,1,3,128,128,device=hidden.device,generator=generator)
    durations=torch.full((n,),duration,device=hidden.device,dtype=torch.long)
    result=torch.zeros(n,device=hidden.device);mean=torch.zeros(n,3,128,128,device=hidden.device)
    if mode=='mean':
        for noise in noises:
            z=noise.expand(n,-1,-1,-1)
            with torch.autocast('cuda',dtype=torch.bfloat16):v=renderer(z,torch.ones(n,device=z.device),hidden,current,durations)
            mean+=(z+v).clamp(-1,1)/QUERIES
        return (mean-goal).square().mean((1,2,3))
    assert mode in ['uniform','highnoise']
    for index,noise in zip(indices,noises[:4]):
        ell=MINIMUM+index.float()*((MAXIMUM-MINIMUM)/STEPS);cleaner=ell+(MAXIMUM-MINIMUM)/STEPS
        time=torch.sigmoid(-ell/2);z=(1-time)*goal+time*noise
        with torch.autocast('cuda',dtype=torch.bfloat16):v=renderer(z.expand(n,-1,-1,-1),time.expand(n),hidden,current,durations)
        estimate=z+time*v
        result+=(STEPS/4)*.5*(cleaner.exp()-ell.exp())*(goal-estimate).square().mean((1,2,3))
    time=torch.sigmoid(torch.tensor(-MAXIMUM/2,device=hidden.device));z=(1-time)*goal+time*noises[-1]
    with torch.autocast('cuda',dtype=torch.bfloat16):v=renderer(z.expand(n,-1,-1,-1),time.expand(n),hidden,current,durations)
    estimate=z+time*v
    result-=log_discretized(goal,estimate,math.exp(-MAXIMUM/2)).mean((1,2,3))
    assert torch.isfinite(result).all()
    return result
