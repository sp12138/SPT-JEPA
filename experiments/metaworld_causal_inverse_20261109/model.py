"""A factual forward world and the posterior of its SAME executed input a0.

Every example holds a0 for five native steps. Forward targets are observed RGB;
inverse targets are that very input a0, never future/optimal commands. No
desired goal, expert route, policy label, reward or state is used in learning.
"""
import copy
import math
import torch
from torch import nn


class CausalInverseWorld(nn.Module):
    def __init__(self,body,mode):
        super().__init__();assert mode in ['native1','held5','flow_uniform','flow_highnoise'];self.body=body;self.mode=mode
        body.horizon=5;body.inverse.requires_grad_(False)
        self.context=nn.Sequential(nn.AdaptiveAvgPool2d(4),nn.Flatten(),nn.Linear((3*96+128)*16,256),nn.SiLU())
        with torch.no_grad():
            self.context[2].weight.zero_();self.context[2].weight[:,:3*96*16].copy_(body.inverse[2].weight)
            self.context[2].bias.copy_(body.inverse[2].bias)
        self.base_action=copy.deepcopy(body.inverse[-1]).requires_grad_(True)
        self.residual=nn.Sequential(nn.Linear(256+4+32,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,4))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
        self.register_buffer('frequencies',torch.logspace(0,3,16)*2*math.pi,persistent=False)

    def condition(self,hidden,current,outcome):
        a=self.body.vision(current);b=self.body.vision(outcome)
        return self.context(torch.cat([a,b,b-a,hidden],1))

    def correction(self,context,noisy,time):
        angle=time[:,None]*self.frequencies[None]
        return self.residual(torch.cat([context,noisy,angle.sin(),angle.cos()],1))

    def velocity(self,context,noisy,time):
        return noisy-self.base_action(context)+self.correction(context,noisy,time)

    def objective(self,batch,terms):
        forbidden={'goal_pixels','desired_goal','boundary','reward','state','task_id','expert_actions','future_action_targets'}
        assert not forbidden&set(batch)
        a0=batch['known_input_a0'].float();assert a0.shape[1:]==(4,)
        hidden,current=self.body.prepare(batch);origin=hidden
        futures=self.body.pixels(batch['future_pixels'].flatten(0,1)).unflatten(0,(len(a0),5));errors=[];image=current
        for t in range(5):
            hidden,image=self.body.predict(hidden,image,a0)
            errors.append((image-futures[:,t]).square().mean((1,2,3)))
        forward=torch.stack(errors,1).mean(1)
        context=self.condition(origin,current,futures[:,0 if self.mode=='native1' else 4])
        uniform,noise=terms
        if self.mode.startswith('flow_'):
            time=4*uniform/(1+3*uniform) if self.mode=='flow_highnoise' else uniform
            noisy=(1-time[:,None])*a0+time[:,None]*noise
            prediction=self.velocity(context,noisy,time);target=noise-a0
        else:
            time=torch.zeros_like(uniform)
            prediction=self.base_action(context)+self.correction(context,torch.zeros_like(a0),time);target=a0
        inverse=(prediction-target).square().mean(1);loss=(forward+inverse).mean()
        return loss,dict(loss=loss,forward_RGB_MSE=forward.mean(),current_input_inverse_loss=inverse.mean(),mean_noise_time=time.mean())

    def forward(self,batch,terms):return self.objective(batch,terms)

    @torch.no_grad()
    def proposals(self,hidden,current,outcome,generator,samples=16):
        context=self.condition(hidden,current,outcome)
        if self.mode.startswith('flow_'):
            context=context.expand(samples,-1);value=torch.randn(samples,4,device=current.device,generator=generator)
            for i in range(16):
                time=value.new_full((samples,),1-i/16);value=value-self.velocity(context,value,time)/16
            return value.clamp(-1,1)
        mean=self.base_action(context)+self.correction(context,current.new_zeros(1,4),current.new_zeros(1))
        value=(mean+torch.randn(samples,4,device=current.device,generator=generator)).clamp(-1,1);value[0]=mean[0].clamp(-1,1)
        return value


def random_terms(batch):
    a=batch['known_input_a0'];return torch.rand(len(a),device=a.device),torch.randn_like(a)
