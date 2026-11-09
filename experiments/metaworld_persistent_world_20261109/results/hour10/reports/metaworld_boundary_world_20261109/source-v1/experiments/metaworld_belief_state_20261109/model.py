"""Factual conditional latent world; goals exist only in inference scoring.

All terms are normalized by the SAME RGB dimension. The posterior sees the
actually recorded endpoint during training, never a desired goal or an action
label. Its KL trains the forward prior to anticipate that observed state.
"""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from experiments.metaworld_endpoint_density_20261109.model import advance
from experiments.metaworld_endpoint_density_20261109.energy import log_discretized

PIXELS=3*128*128
STEPS=64


class BeliefWorld(nn.Module):
    def __init__(self,parent,renderer,mode):
        super().__init__();assert mode in ['joint_anchor','belief_anchor','belief_state']
        self.parent=parent;self.renderer=renderer;self.mode=mode
        self.prior_scale=nn.Conv2d(128,128,1);self.posterior_scale=nn.Conv2d(128,128,1)
        for layer in [self.prior_scale,self.posterior_scale]:
            nn.init.zeros_(layer.weight);nn.init.constant_(layer.bias,math.log(.1))
        self.parent.requires_grad_(True)
        for name in ['decoder','rgb','inverse']:getattr(self.parent,name).requires_grad_(False)

    def call(self,function,*args):
        if self.training and torch.is_grad_enabled():
            return checkpoint(function,*args,use_reentrant=False)
        return function(*args)

    def prepare(self,batch):
        p=self.parent
        history=p.pixels(batch['history_pixels'].flatten(0,1)).unflatten(0,batch['history_pixels'].shape[:2])
        hidden=p.initial(len(history),history.device)
        for t in range(history.shape[1]):
            if t:
                prior=self.call(lambda h,a:advance(p,h,a),hidden,batch['history_actions'][:,t-1])
                hidden=torch.where(batch['history_valid'][:,t-1,None,None,None],prior,hidden)
            posterior=self.call(p.observe,hidden,history[:,t])
            hidden=torch.where(batch['history_valid'][:,t,None,None,None],posterior,hidden)
        return hidden,history[:,-1]*2-1

    def factual(self,batch,durations):
        assert not {'goal_pixels','desired_goal','boundary','reward','state','task_id','expert_actions'}&set(batch)
        assert torch.all((durations>=1)&(durations<=batch['future_valid'].sum(1)))
        hidden,current=self.prepare(batch)
        for t in range(int(durations.max())):
            next_hidden=self.call(lambda h,a:advance(self.parent,h,a),hidden,batch['actions'][:,t])
            hidden=torch.where((t<durations)[:,None,None,None],next_hidden,hidden)
        target=self.parent.pixels(batch['future_pixels'][torch.arange(len(durations),device=durations.device),durations-1])*2-1
        return hidden,current,target

    def distribution(self,prior,image=None):
        logp=self.prior_scale(prior).float().clamp(-5,1)
        if self.mode=='joint_anchor':return prior,logp,prior,logp
        assert image is not None
        posterior=self.call(self.parent.observe,prior,(image+1)/2)
        logq=self.posterior_scale(posterior).float().clamp(-5,1)
        return prior,logp,posterior,logq

    @staticmethod
    def kl(p,logp,q,logq):
        # Per-sample Gaussian KL in nats / number of RGB coordinates, not a
        # separately weighted average over latent coordinates.
        return (logp-logq+.5*((2*(logq-logp)).exp()+(q.float()-p.float()).square()*(-2*logp).exp()-1)).flatten(1).sum(1)/PIXELS

    def velocity(self,bridge,time,z,current,durations):
        anchor=torch.zeros_like(current) if self.mode=='belief_state' else current
        return self.renderer(bridge,time,z,anchor,durations)

    def image_bound(self,z,current,target,durations,index,noise0,noise1):
        ell=-10+index.float()*(20/STEPS);time=torch.sigmoid(-ell/2)
        bridge=(1-time[:,None,None,None])*target+time[:,None,None,None]*noise0
        velocity=self.velocity(bridge,time,z,current,durations)
        estimate=bridge+time[:,None,None,None]*velocity
        loss=STEPS*.5*((ell+20/STEPS).exp()-ell.exp())*(target-estimate).square().mean((1,2,3))
        time_end=torch.full_like(time,torch.sigmoid(torch.tensor(-5.)).item())
        bridge=(1-time_end[:,None,None,None])*target+time_end[:,None,None,None]*noise1
        velocity=self.velocity(bridge,time_end,z,current,durations)
        estimate=bridge+time_end[:,None,None,None]*velocity
        endpoint=-log_discretized(target,estimate,math.exp(-5)).mean((1,2,3))
        return loss+endpoint

    def objective(self,batch,random):
        durations,index,noise0,noise1,latent_noise=random
        prior,current,target=self.factual(batch,durations)
        p,logp,q,logq=self.distribution(prior,target)
        z=q+logq.exp()*latent_noise
        image=self.image_bound(z,current,target,durations,index,noise0,noise1)
        divergence=self.kl(p,logp,q,logq)
        loss=(image+divergence).mean()
        return loss,dict(loss=loss,image_bound=image.mean(),latent_kl=divergence.mean(),
            prior_std=logp.exp().mean(),posterior_std=logq.exp().mean(),duration=durations.float().mean())

    def forward(self,batch,random):return self.objective(batch,random)


def random_terms(batch):
    lengths=batch['future_valid'].sum(1);device=lengths.device;n=len(lengths)
    durations=(torch.rand(n,device=device)*lengths).floor().long()+1
    index=torch.randint(STEPS,(n,),device=device)
    return durations,index,torch.randn(n,3,128,128,device=device),torch.randn(n,3,128,128,device=device),torch.randn(n,128,16,16,device=device)
