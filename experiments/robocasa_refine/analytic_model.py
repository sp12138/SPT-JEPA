"""Predict physical means; propagate the known coding noise analytically.

For clean rollout m and encoder mean mu, use the conditional prior
  p(z_t | z_{t-1}, x_0, a) = N(m_t + rho*(z_{t-1}-m_{t-1}), sigma0^2 I).
The existing correlated Gaussian q then has exact KL mean residual
  (mu_t-m_t) - rho*(mu_{t-1}-m_{t-1}).
Initial-code regularization and image distortion remain unchanged. This is a
replacement prior, not an additional clean auxiliary loss or joint-ELBO claim.

An opt-in stationary coding channel matches the prior innovation variance to
q's sigma0^2*(1-rho^2) and renders rollout samples with the same stationary AR
noise as observation codes. Those innovations are random variables independent
of future RGB; only the clean rollout mean uses the observed initial code.
The Laplace option preserves variance and uses exact conditional Laplace KL;
its AR marginal shape can vary with time, equally for both coding branches.
"""
import hashlib
import math
import os

import torch

from as_jepa_lewm.robust import RobustLeWM
from vforge.models.as_jepa.codes import entry_nats,surprise_nats
from vforge.models.as_jepa.constants import FRAME_ELEMS
from vforge.models.as_jepa.flow import flow_loss
from vforge.models.as_jepa.pixels import pack


def innovation_error(mu,means,rho):
    error=mu.float()-means.float()
    return error[:,1:]-rho*error[:,:-1]


def root_marginal_error(mu,means,rho):
    """A root-conditioned prior with the same one-time forecast marginals.

    Replacing the AR future graph by conditionally independent future codes
    gives variance sigma0^2*S_h, S_h=sum_j rho^(2*j). The additional Gaussian
    KL constant is code_dim/2*sum_h log(S_h); no new scale is introduced.
    """
    powers=mu.new_tensor([rho**(2*t) for t in range(mu.shape[1]-1)],dtype=torch.float32)
    sums=powers.cumsum(0)
    residual=mu.float()-means.float()
    root_powers=mu.new_tensor([rho**t for t in range(1,mu.shape[1])],dtype=torch.float32)
    error=(residual[:,1:]-root_powers[None,:,None,None]*residual[:,:1])/sums.sqrt()[None,:,None,None]
    return error,.5*sums.log()


def laplace_shift_nats(error,scale):
    """Exact equal-scale Laplace KL, elementwise, with a bounded tail slope."""
    magnitude=error.float().abs()/scale
    return magnitude+torch.expm1(-magnitude)


class AnalyticResidualSPT(RobustLeWM):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        assert self.rate_prior=="forecast" and not self.sample_forecast
        assert self.posterior_noise_scale==1. and self.response_query_mode=="none"
        assert not self.flow_force_t_one

    def mean_trajectory(self,initial,actions):
        # No observed future code is an argument or rollout input.
        history=[initial]
        for t in range(actions.shape[1]):
            history.append(self.predict(torch.stack(history,1),actions[:,:t+1])[:,-1])
        return torch.stack(history,1)

    def distortion(self,batch,render,anchor,condition):
        """Default pixel distortion; refinements may change the observed target."""
        w=render.shape[1]
        y=pack(render[:,1:].flatten(0,1))
        anchors=pack(anchor).repeat_interleave(w-1,dim=0)
        return flow_loss(self.renderer,y,condition.flatten(0,1),anchors,self.training_shift)

    def initial_rate(self,mean):
        if getattr(self,"coding_family","gaussian")=="laplace":
            # q0 is Laplace with variance sigma0^2; the initial prior stays N(0,I).
            width=self.sigma0/math.sqrt(2)
            constant=.5*self.sigma0**2+.5*math.log(2*math.pi)-1-math.log(2*width)
            return .5*mean.float().square().sum((-2,-1))+mean[0].numel()*constant
        return entry_nats(mean,self.sigma0)

    def posterior_means(self,mean,batch,step):
        """Randomize future sensors within a synchronized physical sample.

        A transitive cyclic permutation gives every source an equally likely
        target view. The initial code, RGB rendering targets and actions stay
        fixed. A named generator preserves all existing model RNG streams.
        """
        if not getattr(self,"randomize_target_views",False):
            return mean
        views=int(batch["physical_views"])
        assert views>=1 and mean.shape[0]%views==0
        if views==1:
            return mean
        b=mean.shape[0]//views;w=mean.shape[1]
        sequence=mean.reshape(b,views,w,*mean.shape[2:])
        key=f"spt-target-view:{self.forecast_seed}:{int(step)}:{os.environ.get('RANK','0')}"
        seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"little")&((1<<63)-1)
        generator=torch.Generator(device=mean.device).manual_seed(seed)
        shift=torch.randint(views,(b,w-1,1),device=mean.device,generator=generator)
        order=(torch.arange(views,device=mean.device)[None,None,:]+shift)%views
        future=sequence[:,:,1:].transpose(1,2)
        index=order[...,None,None].expand_as(future)
        target=future.gather(2,index).transpose(1,2)
        return torch.cat((sequence[:,:,:1],target),2).reshape_as(mean)

    def loss(self,batch,step=0,ramp_steps=0):
        pixels,render,anchor=self.prepare(batch)
        info=self.encode(dict(pixels=pixels,action=batch["action"]))
        flat=info["emb"];b,w=flat.shape[:2]
        mu=self.posterior_means(flat.reshape(b,w,self.state_slots,self.state_width),batch,step)
        family=getattr(self,"coding_family","gaussian")
        if family=="laplace":
            assert getattr(self,"stationary_coding_channel",False)
            noise=torch.distributions.Laplace(mu.new_zeros(()),mu.new_tensor(1/math.sqrt(2))).sample(mu.shape)
        else:
            assert family=="gaussian"
            noise=torch.randn_like(mu)
        for t in range(1,w):
            noise[:,t]=self.rho*noise[:,t-1]+math.sqrt(1-self.rho**2)*noise[:,t]
        noisy=mu+self.sigma0*noise
        # Preserve the baseline's unused reference forward and outer RNG draws.
        reference=self.predict(noisy.flatten(2)[:,:-1],info["act_emb"][:,:-1]);del reference
        devices=[flat.device.index] if flat.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.forecast_seed+int(step))
            means=self.mean_trajectory(flat[:,0],info["act_emb"][:,:-1]).reshape_as(mu)
            if getattr(self,"stationary_coding_channel",False):
                forecast=means[:,1:]+self.sigma0*noise[:,1:]
            else:
                powers=mu.new_tensor([self.rho**t for t in range(1,w)])[None,:,None,None]
                forecast=means[:,1:]+powers*(noisy[:,:1]-mu[:,:1])
            if not getattr(self,"decoder_forecast_noise",True):
                # Keep the original posterior channel and analytic code rate.
                # The conditional-mean forecast need not decode coding noise.
                forecast=means[:,1:]
            choose=torch.rand(b,w-1,1,1,device=flat.device)<self.forecast_fraction
            condition=torch.where(choose,forecast,noisy[:,1:])
        geometry=getattr(self,"forecast_rate_geometry","markov")
        if geometry=="root-marginal":
            assert family=="gaussian" and not getattr(self,"stationary_coding_channel",False)
            error,marginal_constant=root_marginal_error(mu,means,self.rho)
        else:
            assert geometry=="markov"
            error=innovation_error(mu,means,self.rho)
        entry=self.initial_rate(mu[:,0])
        coding_sigma=self.sigma0
        if getattr(self,"stationary_coding_channel",False):
            coding_sigma*=math.sqrt(1-self.rho**2)
        if family=="laplace":
            innovation_nats=laplace_shift_nats(error,coding_sigma/math.sqrt(2))
            surprise=innovation_nats.sum((-2,-1))
        else:
            surprise=surprise_nats(error,torch.zeros_like(error),coding_sigma)
        if not getattr(self,"stationary_coding_channel",False):
            surprise=surprise+self.state_dim*.5*(-self.rho**2-math.log1p(-self.rho**2))
            if geometry=="root-marginal":
                surprise=surprise+self.state_dim*marginal_constant[None,:]
        fm=self.distortion(batch,render,anchor,condition)
        rate=(entry+surprise.sum(1)).mean()/((w-1)*FRAME_ELEMS)
        beta=self.beta*min(1.,step/max(1,ramp_steps)) if ramp_steps else self.beta
        loss=fm+beta*rate
        if family=="laplace":
            variable=(.5*mu[:,0].float().square()+innovation_nats.sum(1)).mean(0)
        else:
            variable=(.5*mu[:,0].float().square()+error.square().sum(1)/(2*coding_sigma**2)).mean(0)
        return loss,dict(loss=loss,fm=fm,rate=rate,entry=entry.mean(),surprise=surprise.mean(),
            rate_share=beta*rate/fm.detach().clamp_min(1e-8),feature_std=mu.float().std((0,1)).mean(),
            active_dims=(variable>.1).sum(),temporal_noise_rho=loss.new_tensor(self.rho),
            forecast_fraction=loss.new_tensor(self.forecast_fraction),
            clean_rollout_mse=(mu[:,1:].float()-means[:,1:].float()).square().mean(),
            analytic_coding_noise=loss.new_tensor(1.))
