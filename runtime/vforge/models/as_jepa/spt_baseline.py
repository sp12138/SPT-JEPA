"""Small SPT action baseline and anchored comparison, subclassing the Native model."""
import contextlib
from dataclasses import replace
import torch
from torch import nn
from vforge.models.rep_world.config import CodecGeometry,ConditionerConfig
from vforge.models.rep_world.model import RepWorld,FlowConfig,shift_timestep
from vforge.models.rep_world.split_conditioner import SplitViTConditioner
from vforge.models.as_jepa.renderer import AnchoredWorldDiT,renderer_config
from vforge.models.as_jepa.pixels import pack,unpack

class ActionSplitViT(SplitViTConditioner):
    def __init__(self,config,geometry):
        super().__init__(config,geometry);self.action_proj=nn.Linear(10,config.width);self._actions=None;self._zero()
    def _zero(self):nn.init.zeros_(self.action_proj.weight);nn.init.zeros_(self.action_proj.bias)
    def initialize_weights(self):
        super().initialize_weights()
        if hasattr(self,"action_proj"):self._zero()
    def encode_states(self,source,*,return_indices=False):
        out=super().encode_states(source,return_indices=return_indices);encoded,idx=out if return_indices else (out,None)
        if self._actions is None:raise RuntimeError("actions must be explicitly supplied")
        encoded=encoded+self.action_proj(self._actions[:,:encoded.shape[1]].to(encoded.dtype))[:,:,None]
        return (encoded,idx) if return_indices else encoded

class ActionRepWorld(RepWorld):
    def __init__(self,anchored=False,width=192,renderer_depth=6,window=10):
        geometry=CodecGeometry("pixel64","rgb1p4",1,48,1,16,True);heads=width//64
        conditioner=ConditionerConfig(input="latent",width=width,depth=9,encoder_depth=3,heads=heads,mlp_ratio=4.,feature_layers=(8,),pretrained_grid=16,context_mask_ratio=.9,architecture="split_ec",mask_stage="before_e")
        world=replace(renderer_config(16,condition_width=width,cross_attention_3d_rope=True),width=width,depth=renderer_depth,heads=heads,adaln_lora_dim=min(256,width))
        flow=FlowConfig(mode="no_zprev",distant_target_prob=0.,target_layout="single_state",target_origin_reserve=1,future_target_choices=2)
        super().__init__(geometry,world,conditioner,flow)
        self.causal_semantic_encoder=ActionSplitViT(conditioner,geometry)
        self.anchored=anchored;self.mode="S1" if anchored else "S0";self.window=window;self.anchor_gap=0;self.domain="standard"
        if anchored:self.net=AnchoredWorldDiT(world,state_dim=None)
        self.sequence_parallel=False;self.net.sequence_parallel=False
    def forward(self,frames,actions,anchor,return_metrics=False):
        b,w=frames.shape[:2];latents=pack(frames.flatten(0,1)).reshape(b,w,48,16,16).transpose(1,2)
        self.causal_semantic_encoder._actions=actions
        try:
            context=self.net.anchored(pack(anchor).repeat_interleave(self.flow.target_count(w),0)) if self.anchored else contextlib.nullcontext()
            with context:return super().forward(latents=latents,return_metrics=return_metrics)
        finally:self.causal_semantic_encoder._actions=None
    def prepare(self,batch):
        frames=batch["render"].float()/127.5-1.;anchor=batch["anchor"].float()/127.5-1.;return frames,frames,anchor
    def encode(self,info):
        frames=info["pixels"];b,w=frames.shape[:2];latents=pack(frames.flatten(0,1)).reshape(b,w,48,16,16).transpose(1,2)
        info["emb"]=self.causal_semantic_encoder.vit_e(latents).mean(2);return info
    @torch.no_grad()
    def rollout_frames(self,context,actions,anchor,horizon=13,steps=50,seed=0):
        history=list(context.unbind(1));generated=[];b=len(context);generator=torch.Generator(device=context.device).manual_seed(seed)
        schedule=shift_timestep(torch.linspace(1,0,steps+1,device=context.device),self.timestep_shift)
        for _ in range(horizon):
            lo=max(0,len(history)-(self.window-1));frames=torch.stack(history[lo:],1);w=frames.shape[1]
            latents=pack(frames.flatten(0,1)).reshape(b,w,48,16,16).transpose(1,2)
            self.causal_semantic_encoder._actions=actions[:,lo:len(history)]
            try:condition=self.causal_semantic_encoder.next_context(latents)
            finally:self.causal_semantic_encoder._actions=None
            z=torch.randn(b,48,1,16,16,device=context.device,generator=generator)
            manager=self.net.anchored(pack(anchor)) if self.anchored else contextlib.nullcontext()
            with manager:
                for now,nxt in zip(schedule[:-1],schedule[1:]):
                    velocity=self.net(z,now.expand(b,1),condition=condition)
                    z=z-(nxt-now).reshape(1,1,1,1,1)*velocity
            frame=unpack(z);history.append(frame);generated.append(frame)
        return torch.stack(generated,1)
