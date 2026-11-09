"""Random-init recurrent observation dynamics; goals never enter training.

The posterior assimilates only already observed RGB. The prior receives the
known/candidate native input. Inverse uses two OBSERVED adjacent frames and
reconstructs exactly the input used by the forward loss. No inverse is deployed.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ConvGRU(nn.Module):
    def __init__(self,width,context):
        super().__init__()
        self.gates=nn.Conv2d(width+context,2*width,3,padding=1)
        self.candidate=nn.Conv2d(width+context,width,3,padding=1)

    def forward(self,hidden,context):
        update,reset=self.gates(torch.cat([hidden,context],1)).sigmoid().chunk(2,1)
        candidate=self.candidate(torch.cat([hidden*reset,context],1)).tanh()
        return hidden*(1-update)+candidate*update


class FactualRecurrent(nn.Module):
    def __init__(self,horizon=15,width=128,inverse_weight=0.):
        super().__init__(); assert horizon in [1,5,15]
        self.horizon=horizon; self.width=width
        self.vision=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.GroupNorm(8,32),nn.SiLU(),
            nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),
            nn.Conv2d(64,96,3,2,1),nn.GroupNorm(8,96),nn.SiLU())
        self.observer=ConvGRU(width,96)
        self.control=nn.Linear(4,96)
        self.transition=ConvGRU(width,96)
        self.decoder=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(width,64,3,padding=1),nn.GroupNorm(8,64),nn.SiLU(),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(64,32,3,padding=1),nn.GroupNorm(8,32),nn.SiLU())
        self.rgb=nn.Conv2d(32,3,1)
        self.inverse=nn.Sequential(nn.AdaptiveAvgPool2d(4),nn.Flatten(),
            nn.Linear(3*96*4*4,256),nn.SiLU(),nn.Linear(256,4))
        # Small nonzero readouts allow both factual losses to reach vision at
        # initialization, making the one-time unit calibration well-defined.
        nn.init.normal_(self.rgb.weight,std=.001); nn.init.zeros_(self.rgb.bias)
        nn.init.normal_(self.inverse[-1].weight,std=.001); nn.init.zeros_(self.inverse[-1].bias)
        self.register_buffer('inverse_weight',torch.tensor(float(inverse_weight)))

    @staticmethod
    def pixels(value):
        value=value.float()
        if value.shape[-2:] != (128,128):
            value=F.interpolate(value,(128,128),mode='bilinear',align_corners=False,antialias=True).round().clamp(0,255)
        return value/255.

    def initial(self,batch,device=None):
        return torch.zeros(batch,self.width,16,16,device=device or next(self.parameters()).device)

    def observe(self,hidden,image):
        return self.observer(hidden,self.vision(image))

    def predict(self,hidden,image,action):
        context=self.control(action.float())[:,:,None,None].expand(-1,-1,16,16)
        hidden=self.transition(hidden,context)
        delta=F.interpolate(self.rgb(self.decoder(hidden)).float(),(128,128),mode='bilinear',align_corners=False)
        return hidden,(image+delta).clamp(0,1)

    def prepare(self,batch):
        history=self.pixels(batch['history_pixels'].flatten(0,1)).unflatten(0,batch['history_pixels'].shape[:2])
        hidden=self.initial(len(history),history.device)
        for t in range(history.shape[1]):
            if t:
                prior,_=self.predict(hidden,history[:,t-1],batch['history_actions'][:,t-1])
                hidden=torch.where(batch['history_valid'][:,t-1,None,None,None],prior,hidden)
            posterior=self.observe(hidden,history[:,t])
            hidden=torch.where(batch['history_valid'][:,t,None,None,None],posterior,hidden)
        return hidden,history[:,-1]

    def terms(self,batch):
        forbidden={'goal_pixels','boundary','desired_goal','reward','state','task_id','expert_actions'}
        assert not (forbidden & set(batch)),forbidden & set(batch)
        assert batch['actions'].shape[-1] == 4 and batch['future_valid'][:,0].all()
        hidden,image=self.prepare(batch);current=image
        futures=self.pixels(batch['future_pixels'][:,:self.horizon].flatten(0,1)).unflatten(0,(len(image),self.horizon))
        errors=[]
        for t in range(self.horizon):
            hidden,image=self.predict(hidden,image,batch['actions'][:,t])
            errors.append((image-futures[:,t]).square().mean((1,2,3)))
        mask=batch['future_valid'][:,:self.horizon].float()
        forward_per=(torch.stack(errors,1)*mask).sum(1)/mask.sum(1)
        # Inverse reads actual next RGB, not an imagined endpoint and not a goal.
        start_feature=self.vision(current)
        outcome_feature=self.vision(futures[:,0])
        reconstructed=self.inverse(torch.cat([start_feature,outcome_feature,outcome_feature-start_feature],1))
        inverse_per=(reconstructed.float()-batch['actions'][:,0].float()).square().mean(1)
        return forward_per,inverse_per

    def loss(self,batch):
        forward,inverse=self.terms(batch)
        loss=(forward+self.inverse_weight*inverse).mean()
        return loss,dict(forward=forward.mean(),inverse=inverse.mean(),loss=loss,
                         inverse_weight=self.inverse_weight)
