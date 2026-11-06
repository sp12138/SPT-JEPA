"""Continue our own scratch forward observation learner as a goal generator.

The entire20000-update parent cost counts.There are no external or inverse
weights.The constant goal-query modulation preserves the parent's observation
prediction when no past command is given;the new generator reads pixels only.
"""
import torch
from torch.nn import functional as F
from as_jepa_lewm.goal_transport import GoalTransport


class WarmGoalTransport(GoalTransport):
    def __init__(self,init_from_world=True,**kwargs):
        super().__init__(**kwargs)
        channels=self.predictor[0].out_channels
        scale=torch.ones(1,channels,1,1);bias=torch.zeros_like(scale)
        if init_from_world:
            for name in ['vision','predictor','decoder','goal_head']:
                getattr(self,name).load_state_dict(getattr(self.world,name).state_dict(),strict=True)
            with torch.no_grad():
                query=torch.zeros(1,6);query[:,4]=1
                gamma,beta=self.world.context(query).chunk(2,-1)
                scale=(1+.1*gamma)[:,:,None,None];bias=beta[:,:,None,None]
        self.register_buffer('goal_query_scale',scale);self.register_buffer('goal_query_bias',bias)
        self.init_from_world=init_from_world

    def velocity(self,current,goal):
        c=self.vision(current);g=self.vision(goal)
        output=self.predictor(torch.cat([c,g-c],1))
        output=output*self.goal_query_scale.to(output.dtype)+self.goal_query_bias.to(output.dtype)
        output=self.decoder(output)
        return F.interpolate(self.goal_head(output).float(),(128,128),mode='bilinear',align_corners=False)
