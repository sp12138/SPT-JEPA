"""Future-observation kernels; no image decoder or action-prediction target.

The fixed random feature map prevents target collapse without reconstruction,
pretraining, or batch statistics. All trainable weights start randomly.
"""
import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F


class ObservationFeatures(nn.Module):
    def __init__(self, seed, channels=32, size=128):
        super().__init__(); self.size=size
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(610000+seed)
            layers=[];previous=3
            for _ in range(3):
                layers += [nn.Conv2d(previous,channels,3,2,1,bias=False),nn.GroupNorm(8,channels),nn.SiLU()]
                previous=channels
            self.net=nn.Sequential(*layers)
        self.requires_grad_(False)

    def forward(self,pixels):
        with torch.no_grad(),torch.autocast(device_type=pixels.device.type,enabled=False):
            value=F.interpolate(pixels.float()/127.5-1.,(self.size,self.size),mode='bilinear',align_corners=False,antialias=True)
            return self.net(value).float()


class Block(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.net=nn.Sequential(nn.GroupNorm(8,width),nn.SiLU(),nn.Conv2d(width,width,3,padding=1),
                               nn.GroupNorm(8,width),nn.SiLU(),nn.Conv2d(width,width,3,padding=1))

    def forward(self,x):return x+self.net(x)


class PredictiveKernel(nn.Module):
    def __init__(self,feature_seed=1,channels=32,width=128,blocks=6,boundary=True,gaussian=True,affine=True,window=2):
        super().__init__();self.window=window;self.boundary=boundary;self.gaussian=gaussian;self.affine=affine
        self.encoder=ObservationFeatures(feature_seed,channels);self.renderer=None
        self.predictor=nn.Sequential(nn.Conv2d(channels*3,width,1),*[Block(width)for _ in range(blocks)],nn.GroupNorm(8,width),nn.SiLU())
        self.context=nn.Sequential(nn.Linear(6,width*2),nn.SiLU(),nn.Linear(width*2,width*2))
        self.mean_head=nn.Conv2d(width,channels,1)
        self.effect_head=nn.Conv2d(width,channels*4,1) if affine else None
        self.variance_head=nn.Conv2d(width,channels,1)
        self.variance_head.requires_grad_(gaussian)
        self.register_buffer('faces',torch.tensor(list(itertools.product([-1.,0.,1.],repeat=4))),persistent=False)

    def fields(self,previous,current,goal,control,boundary,history_valid):
        flag=boundary.float().reshape(-1,1,1,1)
        difference=(goal-current)*flag
        x=self.predictor(torch.cat([previous,current,difference],1))
        command=torch.zeros_like(control)if self.affine else control*(1-boundary[:,None])
        context=torch.cat([command,boundary[:,None],history_valid[:,None]],-1)
        gamma,beta=self.context(context.float()).chunk(2,-1)
        x=x*(1+.1*gamma[:,:,None,None])+beta[:,:,None,None]
        drift=self.mean_head(x).float()
        # Absorbing goal boundary,as an architectural condition,not a
        # current-observation reconstruction loss.
        distance=(goal-current).float().square().mean((1,2,3),keepdim=True).sqrt()
        drift=drift*((1-flag)+flag*distance)
        effect=self.effect_head(x).float().unflatten(1,(4,current.shape[1])) if self.affine else None
        variance=F.softplus(self.variance_head(x).float())+torch.finfo(torch.float32).eps if self.gaussian else torch.ones_like(drift)
        return drift,effect,variance

    def distribution(self,batch):
        boundary=batch['boundary'].float();control=batch['executed_control'].float()
        assert not torch.any(control[boundary.bool()]!=0),'Boundary prediction cannot read actions'
        drift,effect,var=self.fields(batch['previous_z'],batch['current_z'],batch['goal_z'],control,boundary,batch['history_valid'])
        delta=drift
        if effect is not None:
            delta=delta+torch.einsum('na,nachw->nchw',control*(1-boundary[:,None]),effect)
        return batch['current_z'].float()+delta,var

    def loss(self,batch,step=0,ramp_steps=0):
        mean,var=self.distribution(batch);error=(mean-batch['next_z'].float()).square()
        value=.5*(error/var+var.log())if self.gaussian else .5*error
        loss=value.flatten(1).mean(1).mean()
        return loss,dict(loss=loss,future_prediction=loss,future_mse=error.mean(),predictive_variance=var.mean())

    @torch.no_grad()
    def box_projection(self,drift,effect,target,variance):
        """Exact four-command box QP,enumerating its81 active faces."""
        basis=effect.flatten(2).transpose(1,2).double()
        rhs=(target-drift).flatten(1).double();weight=variance.flatten(1).double().reciprocal()
        matrix=torch.einsum('npi,np,npj->nij',basis,weight,basis)
        linear=torch.einsum('npi,np,np->ni',basis,weight,rhs)
        eps=torch.finfo(torch.float64).eps
        ridge=matrix.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(eps)*eps*32
        matrix=matrix+torch.eye(4,device=matrix.device)[None]*ridge[:,None,None]
        fixed=self.faces.double()[None];free=(fixed==0).double()
        m=matrix[:,None]*free[:,:,:,None]*free[:,:,None,:]+torch.diag_embed(1-free)
        b=(linear[:,None]-torch.einsum('nij,nkj->nki',matrix,fixed.expand(len(matrix),-1,-1)))*free
        solution=torch.linalg.solve(m,b.unsqueeze(-1)).squeeze(-1)*free+fixed
        valid=(solution.abs()<=1+1e-7).all(-1);solution=solution.clamp(-1,1)
        cost=torch.einsum('nki,nij,nkj->nk',solution,matrix,solution)-2*torch.einsum('nki,ni->nk',solution,linear)
        cost=cost.masked_fill(~valid,float('inf'));assert torch.isfinite(cost.min(-1).values).all()
        chosen=solution[torch.arange(len(matrix),device=matrix.device),cost.argmin(-1)]
        return chosen.float()

    def training_optimizer_config(self,max_steps):
        optimizer=torch.optim.AdamW([p for p in self.parameters()if p.requires_grad],lr=3e-4,weight_decay=1e-3)
        warmup=max(1,int(.01*max_steps))
        def schedule(step):
            if step<warmup:return max(1,step)/warmup
            return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,max_steps-warmup))))
        return dict(optimizer=optimizer,lr_scheduler=dict(scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule),interval='step'))

    def audit_expected_gradients(self):
        out=dict(encoder='frozen',predictor='active',context='active',mean_head='active',variance_head='active'if self.gaussian else'frozen')
        if self.affine:out['effect_head']='active'
        return out
