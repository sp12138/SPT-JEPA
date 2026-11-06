"""A goal-directed observation generator,not an inverse action model.

Composed training asks repeated local image changes to explain an observed
path endpoint.The frozen controlled world has only our own scratch forward
ancestry;its full cost counts.No training path predicts an action vector.
"""
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from as_jepa_lewm.causal_transport import CausalTransport,ImageState
from as_jepa_lewm.predictive_kernel import Block


class GoalTransport(nn.Module):
    def __init__(self,world_checkpoint,world_config,world_audit,mode='compose5',horizon=5,width=128,blocks=4,window=2):
        super().__init__();assert(mode,horizon)in[('point1',1),('jump5',5),('compose5',5),('compose15',15)]
        self.mode=mode;self.horizon=horizon;self.window=window;self.renderer=None;self.encoder=ImageState();self.native_generator=True
        assert json.loads(Path(world_audit).read_text())['passed']
        self.world=CausalTransport(**world_config)
        self.world.load_state_dict(torch.load(world_checkpoint,map_location='cpu',weights_only=True),strict=True)
        self.world.requires_grad_(False);self.world.eval()
        self.vision=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.GroupNorm(8,32),nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,96,3,2,1),nn.GroupNorm(8,96),nn.SiLU())
        self.predictor=nn.Sequential(nn.Conv2d(192,width,1),*[Block(width)for _ in range(blocks)],nn.GroupNorm(8,width),nn.SiLU())
        self.decoder=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(width,64,3,padding=1),nn.GroupNorm(8,64),nn.SiLU(),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(64,32,3,padding=1),nn.GroupNorm(8,32),nn.SiLU())
        self.goal_head=nn.Conv2d(32,2,1)
        self.world_checkpoint_sha256=hashlib.sha256(Path(world_checkpoint).read_bytes()).hexdigest()

    @property
    def grid(self):return self.world.grid

    def train(self,mode=True):
        super().train(mode);self.world.eval();return self

    def warp(self,image,flow):return self.world.warp(image,flow)

    def velocity(self,current,goal):
        c=self.vision(current);g=self.vision(goal)
        output=self.decoder(self.predictor(torch.cat([c,g-c],1)))
        return F.interpolate(self.goal_head(output).float(),(128,128),mode='bilinear',align_corners=False)

    def distribution(self,batch):
        assert not torch.any(batch['executed_control']!=0),'Observation generator never reads action inputs'
        current=self.encoder(batch['current_pixels']);goal=self.encoder(batch['goal_pixels']);steps=batch['forecast_steps']
        if self.mode in['point1','jump5']:
            return self.warp(current,self.velocity(current,goal)*steps[:,None,None,None])
        base=current;mapping=self.grid.permute(0,3,1,2).expand(len(current),-1,-1,-1).clone()
        for index in range(self.horizon):
            updated=self.warp(mapping,self.velocity(current,goal))
            mapping=torch.where((steps>index)[:,None,None,None],updated,mapping)
            current=F.grid_sample(base,mapping.permute(0,2,3,1),mode='bilinear',padding_mode='border',align_corners=False)
        return current

    def loss(self,batch,step=0,ramp_steps=0):
        predicted=self.distribution(batch);target=self.encoder(batch['next_pixels'])
        # One Gaussian future-image likelihood in native 8-bit RGB units.
        error=((predicted-target)*255).square().mean((1,2,3));loss=.5*error.mean()
        return loss,dict(loss=loss,future_rgb8_mse=error.mean(),future_mse=error.mean()/(255**2))

    def fields(self,current,goal,boundary,past,valid):
        assert torch.all(boundary==boundary[0])
        if not bool(boundary[0]):return self.world.fields(current,goal,boundary,past,valid)
        velocity=self.velocity(current,goal)
        return torch.zeros_like(velocity),current.new_zeros(len(current),8,2,128,128),velocity,current.new_zeros(len(current),1,1,1)

    @torch.no_grad()
    def choose_control(self,current,goal,valid,previous_control=None):
        return CausalTransport.choose_control(self,current,goal,valid,previous_control)

    @torch.no_grad()
    def command(self,previous,current,goal,valid,objective,previous_control=None):
        assert objective=='projection';action,flow=self.choose_control(current,goal,valid,previous_control)
        return action,self.warp(current,flow)

    def training_optimizer_config(self,max_steps):
        optimizer=torch.optim.AdamW([p for p in self.parameters()if p.requires_grad],lr=3e-4,weight_decay=1e-3)
        warmup=max(1,int(.01*max_steps))
        def schedule(step):
            if step<warmup:return max(1,step)/warmup
            return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,max_steps-warmup))))
        return dict(optimizer=optimizer,lr_scheduler=dict(scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule),interval='step'))

    def audit_expected_gradients(self):
        return dict(world='frozen',encoder='frozen',vision='active',predictor='active',decoder='active',goal_head='active')
