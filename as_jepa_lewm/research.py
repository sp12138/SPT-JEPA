"""Explicit research variants; the original AS-JEPA class and frozen runs stay separate."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from as_jepa_lewm.model import ASJepaLeWM
from vforge.models.as_jepa.codes import entry_nats
from vforge.models.as_jepa.pixels import pack
from vforge.models.as_jepa.flow import draw_timesteps

class ResearchLeWM(ASJepaLeWM):
    def __init__(self,method="baseline",rho=.8,forecast_fraction=.5,noise_diagonal=None,**kwargs):
        super().__init__(**kwargs);self.method=method;self.rho=rho;self.forecast_fraction=forecast_fraction
        diagonal=torch.tensor(noise_diagonal if noise_diagonal is not None else [1.]*self.state_dim,dtype=torch.float32)
        if diagonal.shape!=(self.state_dim,) or bool((diagonal<=0).any()):raise ValueError("Positive state-width noise diagonal required")
        self.register_buffer("noise_diagonal",diagonal)
        self.variance_head=nn.Sequential(nn.Linear(self.state_dim+1,128),nn.GELU(),nn.Linear(128,256)) if method=="predictable_weight" else None
        self.inverse_head=nn.Sequential(nn.Linear(2*self.state_dim+80,192),nn.GELU(),nn.Linear(192,1)) if method in {"action_forks","action_forks_null"} else None
        self._loss_terms=None
    def encode(self,info):
        if self.method=="canonical_color":
            x=info["pixels"];flat=x.flatten(0,1)
            raw=(flat*self.image_std+self.image_mean)
            quantile=torch.quantile(raw.float().flatten(2),torch.tensor([.05,.95],device=raw.device),dim=-1)
            lower,upper=quantile[0],quantile[1]
            canonical=(raw-lower[:,:,None,None])/(upper-lower).clamp_min(.05)[:,:,None,None]
            info={**info,"pixels":((canonical-self.image_mean)/self.image_std).unflatten(0,x.shape[:2])}
        return super().encode(info)
    def context_samples(self,mu,samples,generator):
        noise=torch.randn((samples,*mu.shape[1:]),device=mu.device,generator=generator)
        if self.method=="correlated_noise":
            for t in range(1,noise.shape[1]):noise[:,t]=self.rho*noise[:,t-1]+math.sqrt(1-self.rho**2)*noise[:,t]
        noise=noise*self.noise_diagonal.sqrt()
        return mu.expand(samples,-1,-1)+self.sigma0*noise
    def transition_noise(self,value,generator):
        return self.sigma0*torch.randn(value.shape,device=value.device,generator=generator)*self.noise_diagonal.sqrt()
    def loss(self,batch,step=0,ramp_steps=0):
        if self.mode=="L0":return super().loss(batch,step,ramp_steps)
        pixels,render,anchor=self.prepare(batch)
        info=self.encode(dict(pixels=pixels,action=batch["action"]));flat=info["emb"];b,w,dim=flat.shape
        noise=torch.randn_like(flat)
        if self.method=="correlated_noise":
            for t in range(1,w):noise[:,t]=self.rho*noise[:,t-1]+math.sqrt(1-self.rho**2)*noise[:,t]
        noisy=flat+self.sigma0*noise*self.noise_diagonal.sqrt()
        predicted=self.predict(noisy[:,:-1],info["act_emb"][:,:-1])
        entry=.5*(flat[:,0].float().square().sum(-1)+self.sigma0**2*self.noise_diagonal.sum()-dim-torch.log(self.sigma0**2*self.noise_diagonal).sum())
        target_mean=flat[:,1:]
        constant=0.
        if self.method=="correlated_noise":
            target_mean=target_mean+self.rho*(noisy[:,:-1]-flat[:,:-1])
            constant=.5*dim*(-self.rho**2-math.log(1-self.rho**2))
        per_dim=(target_mean.float()-predicted.float()).square()/(2*self.sigma0**2*self.noise_diagonal)
        surprise=per_dim.sum(-1)+constant
        rate=(entry+surprise.sum(1)).mean()/((w-1)*12288)
        conditioning=noisy[:,1:]
        if self.method=="forecast_render":
            history=[noisy[:,0]]
            future=[]
            for t in range(w-1):
                past=torch.stack(history,1)
                nxt=self.predict(past,info["act_emb"][:,:t+1])[:,-1]
                history.append(nxt);future.append(nxt)
            future=torch.stack(future,1)
            choose=torch.rand(b,w-1,1,device=flat.device)<self.forecast_fraction
            conditioning=torch.where(choose,future,conditioning)
        y=pack(render[:,1:].flatten(0,1));aa=pack(anchor).repeat_interleave(w-1,0)
        condition=conditioning.flatten(0,1).unflatten(-1,(self.state_slots,self.state_width))
        n=len(y);eps=torch.randn_like(y)
        if self.method=="snr_weighted":
            ell=torch.rand(n,1,device=y.device)*20-10;t=torch.sigmoid(-.5*ell)
            weight=10*(1-t).square()
        else:
            t=draw_timesteps(n,y.device,self.shift);weight=torch.ones_like(t)
        tv=t.reshape(-1,1,1,1,1)
        with self.renderer.anchored(aa):velocity=self.renderer((1-tv)*y+tv*eps,t,condition=condition)
        residual=(y-eps-velocity).float().square()
        raw_fm=residual.mean()
        if self.method=="predictable_weight":
            # Causal forecast and noise time only; no future state/target pixel enters variance prediction.
            variance_input=torch.cat([predicted.flatten(0,1).float(),t.float()],-1)
            logv=self.variance_head(variance_input).clamp(-8,4).reshape(n,1,1,16,16)
            fm=(.5*torch.exp(-logv)*residual+.5*logv).mean()
        else:fm=(residual*weight.reshape(-1,1,1,1,1)).mean()
        inverse=fm*0
        if self.method in {"action_forks","action_forks_null"} and "fork_futures" in batch:
            context=self.normalize(batch["fork_context"])
            futures=self.normalize(batch["fork_futures"])
            bf,alternatives=futures.shape[:2]
            pair=self.encode(dict(pixels=torch.cat([context[:,None],futures],1)))["emb"]
            candidates=batch["fork_actions"].flatten(2)
            origin=pair[:,0,None,None].expand(-1,alternatives,alternatives,-1)
            endpoint=pair[:,1:,None].expand(-1,-1,alternatives,-1)
            action=candidates[:,None].expand(-1,alternatives,-1,-1)
            logits=self.inverse_head(torch.cat([origin,endpoint,action],-1)).squeeze(-1)
            inverse=F.cross_entropy(logits.flatten(0,1),torch.arange(alternatives,device=logits.device).repeat(bf))
        loss=fm+self.beta*rate+(inverse if self.method=="action_forks" else inverse*0)
        self._loss_terms=(fm,self.beta*rate)
        metrics=dict(loss=loss,fm=fm,raw_fm=raw_fm,rate=rate,entry=entry.mean(),surprise=surprise.mean(),
                     feature_std=flat.float().std((0,1)).mean(),rate_share=self.beta*rate/fm.detach().abs().clamp_min(1e-8),
                     active_dims=(per_dim.mean((0,1))>.1).sum(),inverse_action=inverse)
        if step%2000==0:self.rate_dimensions=(.5*flat[:,0].float().square()+per_dim.sum(1)).mean(0).detach().cpu().tolist()
        return loss,metrics
