"""Native semantic-memory feedback experiment using the existing pixel renderer."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from vforge.models.as_jepa.spt_baseline import ActionRepWorld
from vforge.models.as_jepa.pixels import pack,unpack
from vforge.models.rep_world.model import shift_timestep

class SemanticCarry(ActionRepWorld):
    def __init__(self,width=192,renderer_depth=6,window=10,latent_weight=.1,use_semantic_feedback=True,freeze_frame_encoder=True):
        super().__init__(anchored=True,width=width,renderer_depth=renderer_depth,window=window)
        self.mode="S1";self.latent_weight=latent_weight;self.use_semantic_feedback=use_semantic_feedback
        self.transition=nn.Sequential(nn.Linear(width+10,2*width),nn.GELU(),nn.Linear(2*width,width))
        self.spatial=nn.Conv2d(width,width,3,padding=1,groups=width)
        self._initialize_transition()
        if freeze_frame_encoder:self.causal_semantic_encoder.vit_e.requires_grad_(False)
    def _initialize_transition(self):
        nn.init.zeros_(self.transition[-1].weight);nn.init.zeros_(self.transition[-1].bias)
        nn.init.zeros_(self.spatial.weight);nn.init.zeros_(self.spatial.bias)
    def initialize_weights(self):
        super().initialize_weights()
        if hasattr(self,"transition"):self._initialize_transition()
    def advance(self,encoded,action):
        b,s,v,d=encoded.shape
        a=action[:,:,None].expand(-1,-1,v,-1)
        local=self.transition(torch.cat([encoded,a.to(encoded.dtype)],-1))
        spatial=self.spatial(encoded.reshape(b*s,16,16,d).permute(0,3,1,2)).permute(0,2,3,1).reshape_as(encoded)
        return encoded+local+spatial
    def forward(self,frames,actions,anchor,return_metrics=False):
        result=super().forward(frames,actions,anchor,return_metrics=True)
        b,w=frames.shape[:2];packed=pack(frames.flatten(0,1)).reshape(b,w,48,16,16).transpose(1,2)
        encoded=self.causal_semantic_encoder.vit_e(packed)
        predicted=self.advance(encoded[:,:-1],actions[:,:-1]);latent=F.mse_loss(predicted.float(),encoded[:,1:].detach().float())
        result["loss"]=result["loss"]+self.latent_weight*latent
        result["metrics"]={**result.get("metrics",{}),"carry_prediction_mse":latent.detach()}
        return result if return_metrics else result["loss"]
    @torch.no_grad()
    def rollout_frames(self,context,actions,anchor,horizon=13,steps=50,seed=0):
        if not self.use_semantic_feedback:return super().rollout_frames(context,actions,anchor,horizon,steps,seed)
        b,w=context.shape[:2];packed=pack(context.flatten(0,1)).reshape(b,w,48,16,16).transpose(1,2)
        memory=list(self.causal_semantic_encoder.vit_e(packed).unbind(1));generated=[]
        generator=torch.Generator(device=context.device).manual_seed(seed)
        schedule=shift_timestep(torch.linspace(1,0,steps+1,device=context.device),self.timestep_shift)
        for i in range(horizon):
            lo=max(0,len(memory)-(self.window-1));encoded=torch.stack(memory[lo:],1)
            action=actions[:,lo:len(memory)]
            condition=self.causal_semantic_encoder.prefill(encoded+self.causal_semantic_encoder.action_proj(action.to(encoded.dtype))[:,:,None])[:,-1]
            z=torch.randn(b,48,1,16,16,device=context.device,generator=generator)
            with self.net.anchored(pack(anchor)):
                for now,nxt in zip(schedule[:-1],schedule[1:]):
                    velocity=self.net(z,now.expand(b,1),condition=condition);z=z-(nxt-now).reshape(1,1,1,1,1)*velocity
            generated.append(unpack(z));memory.append(self.advance(torch.stack(memory[-1:],1),actions[:,len(memory)-1:len(memory)])[:,0])
        return torch.stack(generated,1)
