"""Native SPT AS-JEPA: existing frame encoder, state-causal blocks and renderer."""
from dataclasses import replace
import torch
from torch import nn
from vforge.models.rep_world.config import CodecGeometry,ConditionerConfig
from vforge.models.rep_world.split_conditioner import StateViTEncoder
from vforge.models.as_jepa.pooling import QueryPool
from vforge.models.as_jepa.prior import StatePrior
from vforge.models.as_jepa.renderer import AnchoredWorldDiT,renderer_config
from vforge.models.as_jepa.pixels import pack,unpack
from vforge.models.as_jepa.flow import flow_loss,sample,timestep_shift
from vforge.models.as_jepa.codes import entry_nats,surprise_nats
from vforge.models.as_jepa.constants import K,d,STATE_DIM,FRAME_ELEMS

class ASJepaSPT(nn.Module):
    def __init__(self,beta=4.,sigma0=.1,window=8,anchor_gap=8,width=192,encoder_depth=6,prior_depth=6,renderer_depth=6,state_slots=K,state_width=d):
        super().__init__();self.state_slots=state_slots;self.state_width=state_width;self.state_dim=state_slots*state_width;self.beta=beta;self.sigma0=sigma0;self.window=window;self.anchor_gap=anchor_gap;self.mode="S2";self.domain="standard"
        self.geometry=CodecGeometry("pixel64","rgb1p4",1,48,1,16,True)
        heads=width//64
        cfg=ConditionerConfig(input="latent",width=width,depth=encoder_depth+1,encoder_depth=encoder_depth,heads=heads,mlp_ratio=4.,feature_layers=(encoder_depth,),architecture="split_ec",context_mask_ratio=0.)
        self.encoder=StateViTEncoder(cfg,self.geometry);self.pool=QueryPool(width,heads,self.state_slots,self.state_width)
        self.prior=StatePrior(self.state_slots,self.state_width,10,width=width,depth=prior_depth,heads=heads)
        rcfg=replace(renderer_config(self.state_width),width=width,depth=renderer_depth,heads=heads,adaln_lora_dim=min(256,width))
        self.renderer=AnchoredWorldDiT(rcfg,state_dim=self.state_width,num_slots=self.state_slots);self.renderer.sequence_parallel=False;self.shift=timestep_shift(rcfg)
        self.initialize_weights()
    def initialize_weights(self):
        self.encoder.initialize_weights();self.pool.initialize();self.prior.initialize_weights();self.renderer.initialize_weights()
    def encoded(self,frames):
        # frames [B,W,3,64,64], one independent state per frame.
        b,w=frames.shape[:2];latents=pack(frames.flatten(0,1)).reshape(b,w,48,16,16).transpose(1,2)
        tokens=self.encoder(latents)
        return self.pool(tokens.flatten(0,1)).reshape(b,w,self.state_slots,self.state_width)
    def forward(self,frames,actions,anchor,return_metrics=False):
        mu=self.encoded(frames);s=mu+self.sigma0*torch.randn_like(mu);m=self.prior(s[:,:-1],actions[:,:-1])
        entry=entry_nats(mu[:,0],self.sigma0);surprise=surprise_nats(mu[:,1:],m,self.sigma0)
        y=pack(frames[:,1:].flatten(0,1));a=pack(anchor).repeat_interleave(frames.shape[1]-1,0)
        fm=flow_loss(self.renderer,y,s[:,1:].flatten(0,1),a,self.shift);rate=(entry+surprise.sum(1)).mean()/((frames.shape[1]-1)*FRAME_ELEMS)
        loss=fm+self.beta*rate
        variable_rate=(.5*mu[:,0].float().square()+(mu[:,1:].float()-m.float()).square().sum(1)/(2*self.sigma0**2)).mean(0).flatten()
        self.training_variable_rate=variable_rate.detach()
        metrics=dict(loss_fm=fm.detach(),entry_nats=entry.mean().detach(),surprise_nats=surprise.mean().detach(),rate_per_value=rate.detach(),rate_share=(self.beta*rate/fm.detach().clamp_min(1e-8)).detach(),active_dims_train_microbatch=(variable_rate>.1).sum().detach(),feature_std=mu.float().std((0,1)).mean().detach())
        return dict(loss=loss,metrics=metrics) if return_metrics else loss
    def prepare(self,batch):
        frames=batch["render"].float()/127.5-1.;anchor=batch["anchor"].float()/127.5-1.
        return frames,frames,anchor
    def encode(self,info):
        info["emb"]=self.encoded(info["pixels"]).flatten(2);return info
    def predict(self,emb,actions):return self.prior(emb.unflatten(-1,(self.state_slots,self.state_width)),actions).flatten(2)
    def action_encoder(self,actions):return actions
    @torch.no_grad()
    def render_states(self,states,anchor,steps=50,seed=0):
        states=states.unflatten(-1,(self.state_slots,self.state_width)) if states.shape[-1]==self.state_dim else states
        generator=torch.Generator(device=states.device).manual_seed(seed)
        return unpack(sample(self.renderer,states,pack(anchor),self.shift,steps=steps,generator=generator))
