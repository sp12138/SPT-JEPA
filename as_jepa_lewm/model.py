"""LeWM components plus the spec's rate chain and SPT renderer."""
from dataclasses import replace
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import ViTConfig, ViTModel
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Predictor,Embedder,MLP
from stable_worldmodel.wm.loss import SIGReg
from vforge.models.as_jepa.constants import K,d,STATE_DIM,FRAME_ELEMS
from vforge.models.as_jepa.pooling import QueryPool
from vforge.models.as_jepa.renderer import AnchoredWorldDiT,renderer_config
from vforge.models.as_jepa.pixels import pack,unpack
from vforge.models.as_jepa.flow import flow_loss,sample,timestep_shift
from vforge.models.as_jepa.codes import entry_nats,surprise_nats


class PartitionedQueryPool(QueryPool):
    """Same-parameter diagnostic of pooling addresses, not a new architecture.

    Four queries retain one balanced subset of patches each and share the CLS
    token. The encoder itself still has global attention. No mask is checkpointed.
    """
    def __init__(self, width, heads, K, d, partition):
        if K != 4 or partition not in {"quadrants", "shuffled"}:
            raise ValueError("Partitioned pooling requires four slots and a known partition")
        super().__init__(width, heads, K, d)
        self.partition = partition

    def attention_mask(self, token_count, device):
        patches = token_count - 1
        side = math.isqrt(patches)
        if side < 2 or side * side != patches or side % 2:
            raise ValueError("Partitioned pooling expects CLS plus an even square patch grid")
        if self.partition == "quadrants":
            index = torch.arange(patches, device=device)
            labels = 2 * (index // side >= side // 2).long() + (index % side >= side // 2).long()
        else:
            # Isolated generator: same balanced random partition for all models,
            # batches and devices, without changing training RNG consumption.
            generator = torch.Generator(device="cpu").manual_seed(20261030)
            order = torch.randperm(patches, generator=generator, device="cpu")
            labels = torch.empty(patches, dtype=torch.long, device="cpu")
            labels[order] = torch.arange(4, device="cpu").repeat_interleave(patches // 4)
            labels = labels.to(device)
        mask = torch.zeros((4, token_count), dtype=torch.bool, device=device)
        mask[:, 1:] = labels.unsqueeze(0) != torch.arange(4, device=device).unsqueeze(1)
        return mask

    def forward(self, tokens):
        q = self.queries.expand(tokens.shape[0], -1, -1)
        mask = self.attention_mask(tokens.shape[1], tokens.device)
        pooled, _ = self.attn(q, tokens, tokens, attn_mask=mask, need_weights=False)
        return self.head(self.norm(pooled))


class ASJepaLeWM(LeWM):
    renderer_type = AnchoredWorldDiT

    def __init__(self, mode="L2", beta=4.0, sigma0=.1, window=8, use_anchor=True,
                 renderer_width=192, renderer_depth=6, renderer_heads=3,
                 carry=True, domain="standard", encoder_size=224, encoder_depth=12,
                 lambda_sigreg=.09, objective="flow", anchor_gap=8, batchnorm_state=False, fp32_pool=False, batchnorm_projector=False, state_slots=4, state_width=16, action_dim=10, flow_logit_shift=0., renderer_anchor_mode="observed", pool_partition="full"):
        if action_dim < 1 or not math.isfinite(flow_logit_shift):raise ValueError("Invalid action dimension or flow logit shift")
        self.flow_logit_shift=float(flow_logit_shift)
        if renderer_anchor_mode not in {"observed","zero"}:raise ValueError("Unknown renderer anchor mode")
        self.renderer_anchor_mode=renderer_anchor_mode
        if pool_partition not in {"full", "quadrants", "shuffled"}:
            raise ValueError("Unknown pool_partition")
        if pool_partition != "full" and (mode == "L0" or state_slots != 4):
            raise ValueError("Partitioned pooling requires a four-slot AS-JEPA model")
        self.pool_partition=pool_partition
        self.state_slots=state_slots;self.state_width=state_width;self.state_dim=state_slots*state_width
        self.mode=mode; self.beta=beta; self.sigma0=sigma0; self.window=window
        self.carry=carry; self.domain=domain; self.encoder_size=encoder_size
        self.lambda_sigreg=lambda_sigreg; self.objective=objective; self.anchor_gap=anchor_gap; self.batchnorm_state=batchnorm_state; self.fp32_pool=fp32_pool; self.batchnorm_projector=batchnorm_projector
        encoder=ViTModel(ViTConfig(hidden_size=192,num_hidden_layers=12,num_attention_heads=3,intermediate_size=768,image_size=224,patch_size=14),add_pooling_layer=False,use_mask_token=False)
        encoder.config.interpolate_pos_encoding=True
        if encoder_depth!=12:encoder.encoder.layer=encoder.encoder.layer[:encoder_depth]
        if hasattr(encoder,"set_attn_implementation"):encoder.set_attn_implementation("sdpa")
        baseline=mode=="L0"; state_dim=192 if baseline else self.state_dim
        predictor=Predictor(num_frames=window-1,depth=6,heads=16,mlp_dim=2048,input_dim=state_dim,hidden_dim=192,output_dim=state_dim,dim_head=64,dropout=.1,emb_dropout=0.)
        action_encoder=Embedder(input_dim=action_dim,smoothed_dim=action_dim,emb_dim=state_dim,mlp_scale=4)
        projector=MLP(192,2048,192,norm_fn=nn.BatchNorm1d) if baseline else None
        pred_proj=MLP(192,2048,192,norm_fn=nn.BatchNorm1d) if baseline else None
        super().__init__(encoder,predictor,action_encoder,projector,pred_proj)
        if baseline:self.pool=None
        elif pool_partition == "full":self.pool=QueryPool(192,3,self.state_slots,self.state_width)
        else:
            self.pool=PartitionedQueryPool(192,3,self.state_slots,self.state_width,pool_partition)
        if self.pool is not None:self.pool.initialize()
        if batchnorm_projector and (batchnorm_state or mode!="L1"):
            raise ValueError("MLP BatchNorm projector is an explicit L1-only control")
        self.state_bn=MLP(self.state_dim,2048,self.state_dim,norm_fn=nn.BatchNorm1d) if batchnorm_projector else (nn.BatchNorm1d(self.state_dim) if batchnorm_state else nn.Identity())
        if batchnorm_state and mode!="L1":raise ValueError("BatchNorm state is only the explicit L1 fallback, never AS-JEPA")
        if not baseline and carry:
            nn.init.zeros_(self.predictor.transformer.output_proj.weight);nn.init.zeros_(self.predictor.transformer.output_proj.bias)
        self.sigreg=SIGReg()
        self.renderer=None
        if not baseline:
            cfg=replace(renderer_config(self.state_width),width=renderer_width,depth=renderer_depth,heads=renderer_heads,adaln_lora_dim=min(256,renderer_width))
            self.renderer=self.renderer_type(cfg,state_dim=self.state_width,num_slots=self.state_slots,use_anchor=use_anchor)
            self.renderer.sequence_parallel=False
            self.shift=timestep_shift(cfg)
            # Multiplication of odds adds exactly this amount in logit space.
            # Keep the inference schedule unchanged for matched evaluation.
            self.training_shift=self.shift*math.exp(self.flow_logit_shift)
        self.register_buffer("image_mean",torch.tensor([.485,.456,.406]).view(1,3,1,1),persistent=False)
        self.register_buffer("image_std",torch.tensor([.229,.224,.225]).view(1,3,1,1),persistent=False)
    def encode(self,info):
        if self.mode=="L0":return super().encode(info)
        pixels=info["pixels"];b,w=pixels.shape[:2]
        tokens=self.encoder(pixels.flatten(0,1),interpolate_pos_encoding=True).last_hidden_state
        if self.fp32_pool:
            with torch.autocast(device_type=tokens.device.type,enabled=False):
                pooled=self.pool(tokens.float())
        else:pooled=self.pool(tokens)
        info["emb"]=self.state_bn(pooled.reshape(b*w,self.state_dim)).reshape(b,w,self.state_dim)
        if "action" in info:info["act_emb"]=self.action_encoder(info["action"])
        return info
    def predict(self,emb,act_emb):
        predicted=super().predict(emb,act_emb)
        return predicted+emb if self.mode!="L0" and self.carry else predicted
    def normalize(self,pixels):
        original=pixels.shape; flat=pixels.flatten(0,1) if pixels.ndim==5 else pixels
        flat=flat.float()/255.
        if flat.shape[-1]!=self.encoder_size:flat=F.interpolate(flat,size=(self.encoder_size,self.encoder_size),mode="bilinear",align_corners=False,antialias=True)
        flat=(flat-self.image_mean)/self.image_std
        return flat.unflatten(0,original[:2]) if pixels.ndim==5 else flat
    def prepare(self,batch):
        pixels=batch["pixels"].float(); anchor=batch["anchor224"].float()
        if self.domain=="color":
            # A controlled appearance dataset: deterministic RGB affine palette per episode,
            # constant through time and shared with the anchor. Physics and labels stay fixed.
            ep=batch["episode"].to(pixels.device).float()
            factors=torch.stack([.35+.65*torch.remainder(torch.sin(ep*12.9898+k*78.233)*43758.5453,1) for k in range(3)],1)
            bias=torch.stack([35*torch.remainder(torch.sin(ep*9.125+k*23.77)*12781.13,1) for k in range(3)],1)
            pixels=(pixels*factors[:,None,:,None,None]+bias[:,None,:,None,None]).clamp(0,255)
            anchor=(anchor*factors[:,:,None,None]+bias[:,:,None,None]).clamp(0,255)
        if self.domain=="dot":
            pixels=self.add_dot(pixels,batch["rows"])
            anchor=self.add_dot(anchor[:,None],batch["anchor_row"][:,None])[:,0]
        if self.domain=="standard":render=batch["render"].float(); small_anchor=batch["anchor"].float()
        else:
            render=F.interpolate(pixels.flatten(0,1),size=(64,64),mode="area").round().unflatten(0,pixels.shape[:2])
            small_anchor=F.interpolate(anchor,size=(64,64),mode="area").round()
        return self.normalize(pixels),render/127.5-1.,self.render_anchor(small_anchor/127.5-1.)
    def render_anchor(self,anchor):
        # Constant visual input preserves the original parameter/state structure.
        # Apply the same information restriction during training and generation.
        return torch.zeros_like(anchor) if self.renderer_anchor_mode=="zero" else anchor
    def add_dot(self,pixels,rows):
        b,w,c,h,side=pixels.shape; key=rows.to(pixels.device).float()
        def frac(k):return torch.frac(torch.sin(key*12.9898+k*78.233)*43758.5453).abs()
        cx=frac(1)*side;cy=frac(2)*h;colors=torch.stack([frac(i)*255 for i in [3,4,5]],-1)
        yy=torch.arange(h,device=pixels.device)[None,None,:,None];xx=torch.arange(side,device=pixels.device)[None,None,None,:]
        mask=((xx-cx[:,:,None,None])**2+(yy-cy[:,:,None,None])**2)<(.04*side)**2
        return torch.where(mask[:,:,None],colors[:,:,:,None,None],pixels)
    def loss(self,batch,step=0,ramp_steps=0):
        pixels,render,anchor=self.prepare(batch)
        info=self.encode(dict(pixels=pixels,action=batch["action"]))
        flat=info["emb"];b,w=flat.shape[:2]
        if self.mode=="L0":
            pred=self.predict(flat[:,:-1],info["act_emb"][:,:-1]);prediction=F.mse_loss(pred.float(),flat[:,1:].float())
            sigreg=self.sigreg(flat.float().transpose(0,1));loss=prediction+self.lambda_sigreg*sigreg
            return loss,dict(loss=loss,pred=prediction,sigreg=sigreg,feature_std=flat.float().std((0,1)).mean())
        mu=flat.reshape(b,w,self.state_slots,self.state_width)
        s=mu+self.sigma0*torch.randn_like(mu) if self.mode=="L2" else mu
        predicted=self.predict(s.flatten(2)[:,:-1],info["act_emb"][:,:-1]).reshape_as(mu[:,1:])
        entry=entry_nats(mu[:,0],self.sigma0);surprise=surprise_nats(mu[:,1:],predicted,self.sigma0)
        y=pack(render[:,1:].flatten(0,1));a=pack(anchor).repeat_interleave(w-1,dim=0);condition=s[:,1:].flatten(0,1)
        if self.objective=="mse":
            with self.renderer.anchored(a):velocity=self.renderer(torch.zeros_like(y),torch.ones(len(y),1,device=y.device),condition=condition)
            fm=F.mse_loss(velocity.float(),y.float())
        else:fm=flow_loss(self.renderer,y,condition,a,self.training_shift)
        rate=(entry+surprise.sum(1)).mean()/((w-1)*FRAME_ELEMS)
        if self.mode=="L1":
            predloss=F.mse_loss(predicted.float().flatten(2),flat[:,1:].float())
            sigreg=self.sigreg(flat.float().transpose(0,1));loss=fm+predloss+self.lambda_sigreg*sigreg
        else:
            beta=self.beta*min(1.,step/max(1,ramp_steps)) if ramp_steps else self.beta
            loss=fm+beta*rate
        return loss,dict(loss=loss,fm=fm,rate=rate,entry=entry.mean(),surprise=surprise.mean(),rate_share=self.beta*rate/fm.detach().clamp_min(1e-8),feature_std=mu.float().std((0,1)).mean(),active_dims=((.5*mu[:,0].float().square()+(mu[:,1:].float()-predicted.float()).square().sum(1)/(2*self.sigma0**2)).mean(0)>.1).sum())
    @torch.no_grad()
    def render_states(self,states,anchor,steps=50,seed=0):
        anchor=self.render_anchor(anchor)
        states=states.unflatten(-1,(self.state_slots,self.state_width)) if states.shape[-1]==self.state_dim else states
        if self.objective=="mse":
            with self.renderer.anchored(pack(anchor)):
                return unpack(self.renderer(torch.zeros(len(states),48,1,16,16,device=states.device),torch.ones(len(states),1,device=states.device),condition=states))
        generator=torch.Generator(device=states.device).manual_seed(seed)
        return unpack(sample(self.renderer,states,pack(anchor),self.shift,steps=steps,generator=generator))
