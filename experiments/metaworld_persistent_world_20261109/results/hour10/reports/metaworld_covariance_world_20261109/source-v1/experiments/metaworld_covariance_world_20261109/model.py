"""Sample-level Gaussian observation geometry, with analytic nuisance marginalization.

Sigma = diagonal sensor/innovation variance + U U^T. U is learned from
ordinary observed forecast residuals, never actions, goals, or success labels.
All mean-world weights remain frozen. Low rank is a global architecture choice.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

RANK=16
SENSOR_VARIANCE=(1/255)**2/12


def gaussian_nll(residual,variance,basis=None):
    """Per-RGB-dimension nats. Stable ridge form avoids cancellation in Woodbury."""
    r=residual.float().flatten(1);d=variance.float().expand_as(residual).flatten(1)
    y=r*d.rsqrt();logdet=d.log().sum(1)
    if basis is None:q=y.square().sum(1)
    else:
        w=basis.float()*d.rsqrt()[:,:,None]
        gram=w.transpose(1,2)@w+torch.eye(w.shape[-1],device=w.device)[None]
        chol=torch.linalg.cholesky(gram)
        z=torch.cholesky_solve((w.transpose(1,2)@y[:,:,None]),chol)
        q=(y-(w@z).squeeze(-1)).square().sum(1)+z.square().sum((1,2))
        logdet=logdet+2*chol.diagonal(dim1=-2,dim2=-1).log().sum(1)
    result=.5*(q+logdet)/r.shape[1]+.5*math.log(2*math.pi)
    assert torch.isfinite(result).all()
    return result


class CovarianceWorld(nn.Module):
    def __init__(self,parent,mode):
        super().__init__();assert mode in ['scalar','diagonal','lowrank'];self.parent=parent.requires_grad_(False);self.mode=mode
        self.head=nn.Conv2d(32,3*(1+RANK),1);self.scalar=nn.Linear(128,1)
        nn.init.normal_(self.head.weight,std=.00001);nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.scalar.weight,std=.00001)
        bias=math.log(math.expm1(.05**2-SENSOR_VARIANCE))
        with torch.no_grad():self.head.bias[:3].fill_(bias);self.scalar.bias.fill_(bias)
        self.head.requires_grad_(mode!='scalar');self.scalar.requires_grad_(mode=='scalar')

    @torch.no_grad()
    def factual(self,batch,durations):
        assert not {'goal_pixels','boundary','desired_goal','reward','state','task_id','expert_actions'}&set(batch)
        direction=torch.zeros(len(durations),device=durations.device,dtype=torch.long)
        hidden,current=self.parent.prepare(batch,direction);last=hidden
        for t in range(int(durations.max())):
            hidden=self.parent.advance(hidden,batch['actions'][:,t],direction)
            last=torch.where((durations==t+1)[:,None,None,None],hidden,last)
        return last,current,self.parent.render(last,current)

    def distribution(self,hidden):
        if self.mode=='scalar':return F.softplus(self.scalar(hidden.float().mean((2,3))))[:,:,None,None]+SENSOR_VARIANCE,None
        with torch.no_grad():feature=self.parent.parent.decoder(hidden)
        raw=F.interpolate(self.head(feature).float(),(128,128),mode='bilinear',align_corners=False)
        variance=F.softplus(raw[:,:3])+SENSOR_VARIANCE
        basis=raw[:,3:].reshape(len(raw),3,RANK,128,128).permute(0,1,3,4,2).reshape(len(raw),3*128*128,RANK) if self.mode=='lowrank' else None
        return variance,basis

    def objective(self,batch,terms):
        hidden,current,mean=self.factual(batch,terms[0]);target=self.parent.parent.pixels(batch['endpoint_pixels'])
        variance,basis=self.distribution(hidden);per=gaussian_nll(target-mean,variance,basis)
        loss=per.mean()
        return loss,dict(loss=loss,nll_nats_per_RGB=loss,frozen_mean_RGB_MSE=(target-mean).square().mean(),
            diagonal_std=variance.mean().sqrt(),rank_energy=mean.new_zeros(()) if basis is None else basis.square().mean(),duration=terms[0].float().mean())

    def forward(self,batch,terms):return self.objective(batch,terms)

    def goal_cost(self,hidden,current,goal):
        mean=self.parent.render(hidden,current);variance,basis=self.distribution(hidden)
        return gaussian_nll(goal.expand_as(mean)-mean,variance,basis)
