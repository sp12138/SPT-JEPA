"""Compose factual60-step observation distributions; goals only score outcomes.

One and two60-step transitions use the SAME declared16-step reverse-Gaussian
image model. Midpoint observations are sampled from the model, never taken
from expert/evaluation trajectories. All variants use49rendererqueries/action.
"""
import hashlib
import math
import torch
from .model import advance
from .energy import log_discretized

STEPS=16
PIXELS=3*128*128


def coefficients(ell):return torch.sigmoid(ell).sqrt(),torch.sigmoid(-ell).sqrt()


@torch.no_grad()
def evidence(head,hidden,anchor,goal,seed,full=False):
    n=len(hidden);generator=torch.Generator(device=hidden.device).manual_seed(seed)
    indices=torch.arange(16,device=hidden.device).repeat_interleave(3) if full else torch.arange(4,device=hidden.device)*4+torch.randint(4,(4,),device=hidden.device,generator=generator)
    result=torch.zeros(n,device=hidden.device);duration=torch.full((n,),60,device=hidden.device,dtype=torch.long)
    for index in indices:
        ell=-10+index.float()*20/STEPS;cleaner=ell+20/STEPS;time=torch.sigmoid(-ell/2)
        noise=torch.randn(1,3,128,128,device=hidden.device,generator=generator);bridge=(1-time)*goal+time*noise
        with torch.autocast('cuda',dtype=torch.bfloat16):velocity=head(bridge.expand(n,-1,-1,-1),time.expand(n),hidden,anchor,duration)
        estimate=bridge+time*velocity
        result+=(STEPS/len(indices))*.5*(cleaner.exp()-ell.exp())*(goal-estimate).square().mean((1,2,3))
    time=torch.sigmoid(torch.tensor(-5.,device=hidden.device));noise=torch.randn(1,3,128,128,device=hidden.device,generator=generator)
    bridge=(1-time)*goal+time*noise
    with torch.autocast('cuda',dtype=torch.bfloat16):velocity=head(bridge.expand(n,-1,-1,-1),time.expand(n),hidden,anchor,duration)
    result-=log_discretized(goal,bridge+time*velocity,math.exp(-5)).mean((1,2,3))
    assert torch.isfinite(result).all()
    return result


@torch.no_grad()
def sample(head,hidden,anchor,seed,particles=2):
    n=len(hidden);device=hidden.device;generator=torch.Generator(device=device).manual_seed(seed)
    state=hidden.repeat(particles,1,1,1);current=anchor.repeat(particles,1,1,1)
    def noise():return torch.randn(particles,1,3,128,128,device=device,generator=generator).expand(-1,n,-1,-1,-1).reshape(particles*n,3,128,128)
    z=noise();levels=torch.linspace(-10,10,STEPS+1,device=device,dtype=torch.float64)
    duration=torch.full((particles*n,),60,device=device,dtype=torch.long)
    for noisy,cleaner in zip(levels[:-1],levels[1:]):
        time=torch.sigmoid(-noisy/2).float();norm=((1-time).square()+time.square()).sqrt()
        bridge=z*norm
        with torch.autocast('cuda',dtype=torch.bfloat16):velocity=head(bridge,time.expand(particles*n),state,current,duration)
        estimate=bridge+time*velocity
        at,st=coefficients(noisy);ass,ss=coefficients(cleaner)
        delta=-torch.expm1(torch.nn.functional.logsigmoid(noisy)-torch.nn.functional.logsigmoid(cleaner))
        mean=ass*delta/st.square()*estimate+(at/ass)*ss.square()/st.square()*z
        variance=ss.square()*delta/st.square()
        z=mean.float()+variance.sqrt().float()*noise()
    time=torch.sigmoid(-levels[-1]/2).float();bridge=z*((1-time).square()+time.square()).sqrt()
    with torch.autocast('cuda',dtype=torch.bfloat16):velocity=head(bridge,time.expand(particles*n),state,current,duration)
    observed=(bridge+time*velocity+math.exp(-5)*noise()).clamp(-1,1)
    return ((observed+1)*127.5).round()/127.5-1


def mixture(values):
    return -(torch.logsumexp(-values.double()*PIXELS,dim=0)-math.log(len(values)))/PIXELS


@torch.no_grad()
def scores(parent,head,hidden,image,goal,commands,seed,kind):
    assert kind in ['short','long','free_time']
    state=hidden.expand(64,-1,-1,-1).clone()
    anchor=(image*2-1).expand(64,-1,-1,-1);target=goal*2-1
    if kind=='short':
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for t in range(60):state=advance(parent,state,commands[:,t//5])
        return evidence(head,state,anchor,target,seed,full=True),None
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for t in range(60):state=advance(parent,state,commands[:,t//10])
    midpoint_score=evidence(head,state,anchor,target,seed)
    observations=sample(head,state,anchor,seed+700001,particles=2)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        future=parent.observe(state.repeat(2,1,1,1),(observations+1)/2)
        for t in range(60,120):future=advance(parent,future,commands[:,t//10].repeat(2,1))
    finals=[]
    for index in range(2):
        finals.append(evidence(head,future[index*64:(index+1)*64],observations[index*64:(index+1)*64],target,seed+index+1))
    long_score=mixture(torch.stack(finals))
    if kind=='long':return long_score,midpoint_score
    return mixture(torch.stack([midpoint_score,long_score])),midpoint_score


KIND='short'


@torch.no_grad()
def plan(parent,head,mode,hidden,image,goal,generator,warm):
    assert mode in ['uniform','highnoise']
    horizon=60 if KIND=='short' else 120;block=horizon//12
    mean=image.new_zeros(12,4) if warm is None else warm.reshape(12,block,4).mean(1)
    std=torch.ones_like(mean);best=None;best_score=float('inf');first=None;best_midpoint=None
    seed=int(hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()[:15],16)
    for iteration in range(4):
        raw=torch.randn(64,12,4,device=image.device,generator=generator)
        candidates=(mean[None]+std[None]*raw).clamp(-1,1);candidates[0]=mean
        if best is not None:candidates[1]=best
        if iteration==0:first=hashlib.sha256(candidates[:,:6].cpu().numpy().tobytes()).hexdigest()
        values,midpoint=scores(parent,head,hidden,image,goal,candidates,seed,KIND)
        assert torch.isfinite(values).all();elite_values,indices=values.topk(10,largest=False);elite=candidates[indices]
        if float(elite_values[0])<best_score:
            best_score=float(elite_values[0]);best=elite[0].clone();best_midpoint=float(midpoint[indices[0]]) if midpoint is not None else None
        mean=elite.mean(0);std=elite.std(0,unbiased=False).clamp_min(1e-6)
    return best[:,None].expand(-1,block,-1).reshape(horizon,4),dict(planner_score=best_score,
        score_kind='controlled_observation_composition_'+KIND,first_bank_prefix_sha256=first,
        renderer_queries_per_candidate=49,horizon_native=horizon,control_hold_native=block,
        selected_midpoint_NELBO=best_midpoint,goal_not_used_in_midpoint_sampling=True)
