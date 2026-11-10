"""Predict physical means; propagate the known coding noise analytically.

For clean rollout m and encoder mean mu, use the conditional prior
  p(z_t | z_{t-1}, x_0, a) = N(m_t + rho*(z_{t-1}-m_{t-1}), sigma0^2 I).
The existing correlated Gaussian q then has exact KL mean residual
  (mu_t-m_t) - rho*(mu_{t-1}-m_{t-1}).
Initial-code regularization and image distortion remain unchanged. This is a
replacement prior, not an additional clean auxiliary loss or joint-ELBO claim.
"""
import math

import torch

from as_jepa_lewm.robust import RobustLeWM
from vforge.models.as_jepa.codes import entry_nats,surprise_nats
from vforge.models.as_jepa.constants import FRAME_ELEMS
from vforge.models.as_jepa.flow import flow_loss
from vforge.models.as_jepa.pixels import pack


def innovation_error(mu,means,rho):
    error=mu.float()-means.float()
    return error[:,1:]-rho*error[:,:-1]


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

    def loss(self,batch,step=0,ramp_steps=0):
        pixels,render,anchor=self.prepare(batch)
        info=self.encode(dict(pixels=pixels,action=batch["action"]))
        flat=info["emb"];b,w=flat.shape[:2]
        mu=flat.reshape(b,w,self.state_slots,self.state_width)
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
            powers=mu.new_tensor([self.rho**t for t in range(1,w)])[None,:,None,None]
            forecast=means[:,1:]+powers*(noisy[:,:1]-mu[:,:1])
            choose=torch.rand(b,w-1,1,1,device=flat.device)<self.forecast_fraction
            condition=torch.where(choose,forecast,noisy[:,1:])
        error=innovation_error(mu,means,self.rho)
        entry=entry_nats(mu[:,0],self.sigma0)
        surprise=surprise_nats(error,torch.zeros_like(error),self.sigma0)
        surprise=surprise+self.state_dim*.5*(-self.rho**2-math.log1p(-self.rho**2))
        y=pack(render[:,1:].flatten(0,1));anchors=pack(anchor).repeat_interleave(w-1,dim=0)
        fm=flow_loss(self.renderer,y,condition.flatten(0,1),anchors,self.training_shift)
        rate=(entry+surprise.sum(1)).mean()/((w-1)*FRAME_ELEMS)
        beta=self.beta*min(1.,step/max(1,ramp_steps)) if ramp_steps else self.beta
        loss=fm+beta*rate
        variable=(.5*mu[:,0].float().square()+error.square().sum(1)/(2*self.sigma0**2)).mean(0)
        return loss,dict(loss=loss,fm=fm,rate=rate,entry=entry.mean(),surprise=surprise.mean(),
            rate_share=beta*rate/fm.detach().clamp_min(1e-8),feature_std=mu.float().std((0,1)).mean(),
            active_dims=(variable>.1).sum(),temporal_noise_rho=loss.new_tensor(self.rho),
            forecast_fraction=loss.new_tensor(self.forecast_fraction),
            clean_rollout_mse=(mu[:,1:].float()-means[:,1:].float()).square().mean(),
            analytic_coding_noise=loss.new_tensor(1.))
