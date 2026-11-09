"""Factual endpoint rendering. No goal/path/action-policy training branch."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def embedding(value,width=128):
    frequencies=torch.exp(-math.log(10000)*torch.arange(width//2,device=value.device)/(width//2-1))
    angle=value.float()[:,None]*frequencies[None]
    return torch.cat([angle.sin(),angle.cos()],1)


class Block(nn.Module):
    def __init__(self,incoming,outgoing):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(incoming,outgoing,3,padding=1),nn.GroupNorm(8,outgoing),nn.SiLU(),
            nn.Conv2d(outgoing,outgoing,3,padding=1),nn.GroupNorm(8,outgoing),nn.SiLU())

    def forward(self,x):return self.body(x)


class EndpointRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.time=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128))
        self.duration=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128))
        self.down0=Block(6,32);self.down1=Block(32,64);self.down2=Block(64,96);self.down3=Block(96,128)
        self.context=nn.Conv2d(128,128,1);self.middle=Block(128,128)
        self.up2=Block(128+96,96);self.up1=Block(96+64,64);self.up0=Block(64+32,32)
        self.output=nn.Conv2d(32,3,1)
        nn.init.normal_(self.output.weight,std=.001);nn.init.zeros_(self.output.bias)

    def forward(self,noisy,time,hidden,current,duration):
        a=self.down0(torch.cat([noisy,current],1));b=self.down1(F.avg_pool2d(a,2))
        c=self.down2(F.avg_pool2d(b,2));d=self.down3(F.avg_pool2d(c,2))
        condition=self.time(embedding(time*1000))+self.duration(embedding(duration.float()))
        d=self.middle(d+self.context(hidden)+condition[:,:,None,None])
        c=self.up2(torch.cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        b=self.up1(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        a=self.up0(torch.cat([F.interpolate(b,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return self.output(a).float()


def advance(parent,hidden,action):
    # Algebraically the exact hidden part of parent.predict; its RGB decoder
    # does not feed the hidden transition and can be skipped for endpoint use.
    context=parent.control(action.float())[:,:,None,None].expand(-1,-1,16,16)
    return parent.transition(hidden,context)


@torch.no_grad()
def factual_condition(parent,batch,durations):
    assert not {'goal_pixels','desired_goal','boundary','reward','state','task_id','expert_actions'}&set(batch)
    assert torch.all((durations>=1)&(durations<=batch['future_valid'].sum(1)))
    hidden,current=parent.prepare(batch)
    for t in range(int(durations.max())):
        next_hidden=advance(parent,hidden,batch['actions'][:,t])
        hidden=torch.where((t<durations)[:,None,None,None],next_hidden,hidden)
    target=parent.pixels(batch['future_pixels'][torch.arange(len(durations),device=durations.device),durations-1])*2-1
    return hidden,current*2-1,target


def objective(renderer,hidden,current,target,durations,uniform,noise,mode):
    assert mode in ['mean','uniform','highnoise']
    time=torch.ones_like(uniform) if mode=='mean' else uniform
    if mode=='highnoise':time=4*time/(1+3*time)
    shape=time[:,None,None,None]
    bridge=(1-shape)*target+shape*noise
    velocity=renderer(bridge,time,hidden,current,durations)
    per_sample=(velocity-(target-noise)).square().flatten(1).mean(1)
    return per_sample.mean(),dict(loss=per_sample.mean(),noise_time=time.mean(),
        target_duration=durations.float().mean())
