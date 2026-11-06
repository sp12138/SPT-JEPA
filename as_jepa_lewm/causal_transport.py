"""Learn forward image transformations,never actions or inverse-action targets.

Known current controls generate a native-step image warp.Goal-conditioned
queries predict a native future image with the same observation likelihood.
Planning optimizes self-generated controls;no inverse learner is fitted.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from as_jepa_lewm.predictive_kernel import Block
from as_jepa_lewm.signed_density import signed_projection


class ImageState(nn.Module):
    def forward(self,pixels):
        value=pixels.float()
        if value.shape[-2:]!=(128,128):
            value=F.interpolate(value,(128,128),mode='bilinear',align_corners=False,antialias=True).round().clamp(0,255)
        return value/255.


class CausalTransport(nn.Module):
    def __init__(self,mode='stencil',width=128,blocks=4,window=2):
        super().__init__();assert mode in['free','linear','signed','stencil']
        self.mode=mode;self.window=window;self.native_generator=True;self.renderer=None;self.encoder=ImageState()
        self.vision=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.GroupNorm(8,32),nn.SiLU(),
            nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,96,3,2,1),nn.GroupNorm(8,96),nn.SiLU())
        self.predictor=nn.Sequential(nn.Conv2d(192,width,1),*[Block(width)for _ in range(blocks)],nn.GroupNorm(8,width),nn.SiLU())
        self.context=nn.Sequential(nn.Linear(6,width*2),nn.SiLU(),nn.Linear(width*2,width*2))
        self.decoder=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(width,64,3,padding=1),nn.GroupNorm(8,64),nn.SiLU(),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(64,32,3,padding=1),nn.GroupNorm(8,32),nn.SiLU())
        self.effect_head=nn.Conv2d(32,8,1);self.absolute_head=nn.Conv2d(32,8,1)
        self.transient_head=nn.Conv2d(32,8,1);self.absolute_transient_head=nn.Conv2d(32,8,1)
        self.drift_head=nn.Conv2d(32,2,1);self.goal_head=nn.Conv2d(32,2,1);self.uncertainty=nn.Linear(width,1)
        for head in[self.absolute_head,self.transient_head,self.absolute_transient_head,self.drift_head]:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.absolute_head.requires_grad_(mode in['signed','stencil'])
        self.transient_head.requires_grad_(mode=='stencil');self.absolute_transient_head.requires_grad_(mode=='stencil')
        self.drift_head.requires_grad_(mode=='free')
        coordinate=torch.arange(128,dtype=torch.float32).add(.5).mul(2/128).sub(1)
        yy,xx=torch.meshgrid(coordinate,coordinate,indexing='ij')
        self.register_buffer('grid',torch.stack([xx,yy],-1)[None],persistent=False)

    def warp(self,image,flow):
        with torch.autocast(device_type=image.device.type,enabled=False):
            grid=self.grid.to(image.device)+flow.float().permute(0,2,3,1)*(2/128)
            return F.grid_sample(image.float(),grid,mode='bilinear',padding_mode='border',align_corners=False)

    def fields(self,current,goal,boundary,past,valid):
        current_feature=self.vision(current)
        goal_feature=self.vision(goal)
        flag=boundary[:,None,None,None]
        x=self.predictor(torch.cat([current_feature,(goal_feature-current_feature)*flag],1))
        context=torch.cat([past,boundary[:,None],valid[:,None]],-1)
        gamma,beta=self.context(context.float()).chunk(2,-1);x=x*(1+.1*gamma[:,:,None,None])+beta[:,:,None,None]
        variance=F.softplus(self.uncertainty(x.float().mean((2,3))).float()).reshape(-1,1,1,1)+torch.finfo(torch.float32).eps
        visual=self.decoder(x)
        def field(head):return F.interpolate(head(visual).float(),(128,128),mode='bilinear',align_corners=False)
        J=field(self.effect_head).unflatten(1,(4,2))
        R=field(self.absolute_head).unflatten(1,(4,2))if self.mode in['signed','stencil']else torch.zeros_like(J)
        K=field(self.transient_head).unflatten(1,(4,2))if self.mode=='stencil'else torch.zeros_like(J)
        T=field(self.absolute_transient_head).unflatten(1,(4,2))if self.mode=='stencil'else torch.zeros_like(J)
        drift=field(self.drift_head)if self.mode=='free'else-torch.einsum('na,nachw->nchw',past,K)-torch.einsum('na,nachw->nchw',past.abs(),T)
        effect=torch.cat([J+K,R+T],1)
        return drift,effect,field(self.goal_head),variance

    def distribution(self,batch):
        current=self.encoder(batch['current_pixels']);goal=self.encoder(batch['goal_pixels'])
        role=batch['boundary'].float();control=batch['executed_control'].float()
        assert not torch.any(control[role.bool()]!=0),'Goal forecast cannot read a current/future action reference'
        drift,effect,desired,variance=self.fields(current,goal,role,batch['previous_control'].float(),batch['history_valid'].float())
        known=torch.cat([control,control.abs()],-1)
        forward=drift+torch.einsum('na,nachw->nchw',known,effect)
        flow=torch.where(role[:,None,None,None].bool(),desired,forward)
        return self.warp(current,flow),variance

    def loss(self,batch,step=0,ramp_steps=0):
        predicted,variance=self.distribution(batch);target=self.encoder(batch['next_pixels'])
        error=(predicted-target).square().mean((1,2,3),keepdim=True)
        loss=(.5*(error/variance+variance.log())).mean()
        return loss,dict(loss=loss,future_mse=error.mean(),predictive_variance=variance.mean())

    @torch.no_grad()
    def choose_control(self,current,goal,valid,previous_control=None):
        past=current.new_zeros(len(current),4)if previous_control is None else previous_control
        role=current.new_zeros(len(current))
        drift,effect,_,variance=self.fields(current,torch.zeros_like(goal),role,past,valid)
        _,_,desired,goalvar=self.fields(current,goal,role+1,past,valid)
        # Pull the photometric metric back to displacement space.Unobservable
        # flow directions carry no evidence;an own-command Gaussian prior
        # retains maintained actuation there.Its unit scale is common to all tasks.
        gx=F.pad((current[:,:,:,2:]-current[:,:,:,:-2])*.5,(1,1,0,0),mode='replicate')
        gy=F.pad((current[:,:,2:,:]-current[:,:,:-2,:])*.5,(0,0,1,1),mode='replicate')
        gx,gy=self.warp(gx,drift),self.warp(gy,drift)
        photometric=gx[:,None]*effect[:,:,0,None]+gy[:,None]*effect[:,:,1,None]
        delta=desired-drift;target=gx*delta[:,0,None]+gy*delta[:,1,None]
        flat=photometric.flatten(2);identity=torch.cat([torch.eye(4,device=current.device),torch.zeros(4,4,device=current.device)],0)
        basis=torch.cat([flat,identity[None].expand(len(current),-1,-1)],-1)[:,:,:,None,None]
        rhs=torch.cat([target.flatten(1),past],-1)[:,:,None,None]
        noise=torch.cat([(variance+goalvar).flatten(1).expand(-1,target[0].numel()),torch.ones_like(past)],-1)[:,:,None,None]
        action=signed_projection(torch.zeros_like(rhs),basis,rhs,noise,past)
        flow=drift+torch.einsum('na,nachw->nchw',torch.cat([action,action.abs()],-1),effect)
        return action,flow

    @torch.no_grad()
    def command(self,previous,current,goal,valid,objective,previous_control=None):
        assert objective=='projection'
        action,flow=self.choose_control(current,goal,valid,previous_control)
        return action,self.warp(current,flow)

    def training_optimizer_config(self,max_steps):
        optimizer=torch.optim.AdamW([p for p in self.parameters()if p.requires_grad],lr=3e-4,weight_decay=1e-3)
        warmup=max(1,int(.01*max_steps))
        def schedule(step):
            if step<warmup:return max(1,step)/warmup
            return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,max_steps-warmup))))
        return dict(optimizer=optimizer,lr_scheduler=dict(scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule),interval='step'))

    def audit_expected_gradients(self):
        return dict(encoder='frozen',vision='active',predictor='active',context='active',decoder='active',effect_head='active',goal_head='active',uncertainty='active',
            absolute_head='active'if self.mode in['signed','stencil']else'frozen',transient_head='active'if self.mode=='stencil'else'frozen',
            absolute_transient_head='active'if self.mode=='stencil'else'frozen',drift_head='active'if self.mode=='free'else'frozen')
