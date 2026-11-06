"""One recurrent future-observation predictor with causal and goal queries.

No action decoder exists.Causal queries read known executed controls;goal
queries read a desired observation only.The shared latent transition can
evolve with zero control and remember effects before they become visible.
"""
import hashlib
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from as_jepa_lewm.causal_transport import ImageState
from as_jepa_lewm.predictive_kernel import Block


class ConvGRU(nn.Module):
    def __init__(self,width,context):
        super().__init__();self.gates=nn.Conv2d(width+context,2*width,3,padding=1);self.candidate=nn.Conv2d(width+context,width,3,padding=1)
    def forward(self,hidden,context):
        update,reset=self.gates(torch.cat([hidden,context],1)).sigmoid().chunk(2,1)
        candidate=self.candidate(torch.cat([hidden*reset,context],1)).tanh()
        return hidden*(1-update)+candidate*update


class QueryRecurrent(nn.Module):
    def __init__(self,init_checkpoint,training_mode='rollout',width=128,blocks=4,window=2,forecast_horizon=5,warm_start_checkpoint=None,warm_start_audit=None):
        super().__init__();assert training_mode in['teacher','rollout'];assert forecast_horizon in[5,15]
        self.forecast_horizon=forecast_horizon;self.training_mode=training_mode;self.window=window;self.renderer=None;self.encoder=ImageState()
        self.vision=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.GroupNorm(8,32),nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,96,3,2,1),nn.GroupNorm(8,96),nn.SiLU())
        self.initial=nn.Sequential(nn.Conv2d(192,width,1),*[Block(width)for _ in range(blocks)],nn.GroupNorm(8,width),nn.SiLU())
        self.past_control=nn.Linear(4,width);self.control=nn.Linear(4,96);self.goal=nn.Conv2d(96,96,1);self.role=nn.Embedding(2,96)
        self.predictor=ConvGRU(width,96)
        self.decoder=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(width,64,3,padding=1),nn.GroupNorm(8,64),nn.SiLU(),nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(64,32,3,padding=1),nn.GroupNorm(8,32),nn.SiLU())
        self.rgb_head=nn.Conv2d(32,3,1);nn.init.zeros_(self.rgb_head.weight);nn.init.zeros_(self.rgb_head.bias)
        parent=torch.load(init_checkpoint,map_location='cpu',weights_only=True)
        for name,dest in [('vision',self.vision),('predictor',self.initial),('decoder',self.decoder)]:
            dest.load_state_dict({k[len(name)+1:]:v for k,v in parent.items()if k.startswith(name+'.')},strict=True)
        self.parent_sha256=hashlib.sha256(Path(init_checkpoint).read_bytes()).hexdigest()
        if warm_start_checkpoint is not None:
            import json
            assert warm_start_audit is not None and json.loads(Path(warm_start_audit).read_text())['passed']
            self.load_state_dict(torch.load(warm_start_checkpoint,map_location='cpu',weights_only=True),strict=True)
            self.warm_start_sha256=hashlib.sha256(Path(warm_start_checkpoint).read_bytes()).hexdigest()

    def prepare(self,current,previous,goal,past,valid,role):
        c=self.vision(current);p=self.vision(previous);g=self.vision(goal);flag=role[:,None,None,None]
        difference=(p-c)*valid[:,None,None,None]*(1-flag)+(g-c)*flag
        hidden=self.initial(torch.cat([c,difference],1))+self.past_control(past)[:,:,None,None]*(1-flag)
        goal_context=self.goal(g-c)
        return hidden,goal_context

    def step(self,hidden,image,goal_context,control,role):
        context=self.control(control)[:,:,None,None].expand(-1,-1,16,16)
        context=torch.where(role[:,None,None,None].bool(),goal_context,context)+self.role(role.long())[:,:,None,None]
        hidden=self.predictor(hidden,context)
        delta=F.interpolate(self.rgb_head(self.decoder(hidden)).float(),(128,128),mode='bilinear',align_corners=False)
        return hidden,(image+delta).clamp(0,1)

    def distribution(self,batch):
        role=batch['boundary'].float();horizon=self.forecast_horizon;assert batch['executed_sequence'].shape[1]==horizon
        assert not torch.any(batch['executed_sequence'][role.bool()]!=0)
        assert not torch.any(batch['previous_control'][role.bool()]!=0)
        goal=self.encoder(batch['goal_pixels']);controls=batch['executed_sequence'].float()
        if self.training_mode=='teacher':
            current=self.encoder(batch['current_sequence_pixels'].flatten(0,1));previous=self.encoder(batch['previous_sequence_pixels'].flatten(0,1))
            roles=role[:,None].expand(-1,horizon).flatten();goals=goal[:,None].expand(-1,horizon,-1,-1,-1).flatten(0,1)
            hidden,context=self.prepare(current,previous,goals,batch['previous_control_sequence'].flatten(0,1).float(),batch['history_valid_sequence'].flatten(),roles)
            _,future=self.step(hidden,current,context,controls.flatten(0,1),roles)
            return future.unflatten(0,(len(role),horizon))
        image=self.encoder(batch['current_pixels']);previous=self.encoder(batch['previous_pixels'])
        hidden,context=self.prepare(image,previous,goal,batch['previous_control'].float(),batch['history_valid'].float(),role);futures=[]
        for index in range(horizon):
            hidden,image=self.step(hidden,image,context,controls[:,index],role);futures.append(image)
        return torch.stack(futures,1)

    def loss(self,batch,step=0,ramp_steps=0):
        prediction=self.distribution(batch);target=self.encoder(batch['future_pixels'].flatten(0,1)).unflatten(0,(len(prediction),self.forecast_horizon))
        error=((prediction-target)*255).square()
        if 'future_valid'in batch:
            mask=batch['future_valid'].float();assert(mask.sum(1)>=5).all()
            per_sample=(error.mean((2,3,4))*mask).sum(1)/mask.sum(1)
        else:per_sample=error.mean((1,2,3,4))
        loss=.5*per_sample.mean()
        return loss,dict(loss=loss,future_mse=per_sample.mean()/(255**2),future_rgb8_mse=per_sample.mean())

    @torch.no_grad()
    def plan_controls(self,current,previous,goal,valid,past,generator,warm=None,particles=32,iterations=4,blocks=3,block_steps=5):
        role=current.new_zeros(1);physical,_=self.prepare(current,previous,torch.zeros_like(goal),past,valid,role)
        desired,goal_context=self.prepare(current,previous,goal,torch.zeros_like(past),valid,role+1);target=current;targets=[]
        for _ in range(blocks*block_steps):
            desired,target=self.step(desired,target,goal_context,torch.zeros_like(past),role+1);targets.append(target)
        mean=past[:,None].expand(1,blocks,4).clone()if warm is None else warm.reshape(1,blocks,4).clone();std=torch.ones_like(mean)
        best=None;best_cost=float('inf');history=[]
        for _ in range(iterations):
            controls=(mean+std*torch.randn(particles,blocks,4,device=current.device,generator=generator)).clamp(-1,1);controls[0]=mean[0]
            if best is not None:controls[1]=best
            hidden=physical.expand(particles,-1,-1,-1).clone();image=current.expand(particles,-1,-1,-1);prior=past.expand(particles,-1);score=current.new_zeros(particles)
            flag=role.expand(particles);context=goal_context.new_zeros(particles,96,16,16)
            for index,target in enumerate(targets):
                command=controls[:,index//block_steps];hidden,image=self.step(hidden,image,context,command,flag)
                score+=.5*((image-target)*255).square().flatten(1).sum(-1)+.5*(command-prior).square().sum(-1);prior=command
            assert torch.isfinite(score).all();values,elite=score.topk(min(10,particles),largest=False);selected=controls[elite]
            if float(values[0])<best_cost:best_cost=float(values[0]);best=selected[0].clone()
            mean=selected.mean(0,keepdim=True);std=selected.std(0,unbiased=False,keepdim=True);history.append(best_cost)
        commands=best[:,None].expand(-1,block_steps,-1).reshape(blocks*block_steps,4).cpu().numpy()
        return list(commands),best,dict(best_cost=best_cost,cost_history=history,particles=particles,iterations=iterations,native_horizon=blocks*block_steps,no_action_references=True)

    @torch.no_grad()
    def command(self,previous,current,goal,valid,objective,previous_control=None):
        assert objective=='projection';past=current.new_zeros(1,4)if previous_control is None else previous_control
        commands,_,_=self.plan_controls(current,previous,goal,valid,past,torch.Generator(device=current.device).manual_seed(831),particles=4,iterations=2,blocks=1,block_steps=1)
        role=current.new_zeros(1);hidden,context=self.prepare(current,previous,torch.zeros_like(goal),past,valid,role)
        action=torch.as_tensor(commands[0],device=current.device).reshape(1,4);_,image=self.step(hidden,current,context,action,role)
        return action,image

    def training_optimizer_config(self,max_steps):
        optimizer=torch.optim.AdamW([p for p in self.parameters()if p.requires_grad],lr=3e-4,weight_decay=1e-3);warmup=max(1,int(.01*max_steps))
        def schedule(step):
            if step<warmup:return max(1,step)/warmup
            return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,max_steps-warmup))))
        return dict(optimizer=optimizer,lr_scheduler=dict(scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule),interval='step'))

    def audit_expected_gradients(self):
        return dict(encoder='frozen',vision='active',initial='active',past_control='active',control='active',goal='active',role='active',predictor='active',decoder='active',rgb_head='active')
