"""Scratch RGB encoder ablation using the same factual transitions and heads."""
import types
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F


def prepare_pixels(self,batch):
    render=batch['render'].float()/127.5-1.
    assert render.shape[-2:]==(self.raw_render_size,self.raw_render_size)
    return self.normalize(batch['pixels']),render,render[:,0]


def resolution_distortion(self,batch,render,anchor,condition):
    from vforge.models.as_jepa.pixels import pack
    from vforge.models.as_jepa.flow import flow_loss
    patch=self.raw_render_size//16;w=render.shape[1]
    y=pack(render[:,1:].flatten(0,1),patch=patch)
    anchors=pack(anchor,patch=patch).repeat_interleave(w-1,dim=0)
    return flow_loss(self.renderer,y,condition.flatten(0,1),anchors,self.training_shift)


def clean_residual_velocity(self,latent,timestep,*args,**kwargs):
    """Predict clean innovation; analytically carry full-rank Gaussian noise.

    The public renderer interface still returns the original FM velocity.
    Its loss is unchanged: velocity error equals clean-innovation error / t.
    Flow sampling never evaluates its final endpoint at t=0.
    """
    assert self._anchor is not None and bool((timestep>0).all())
    t=timestep.reshape(-1,1,1,1,1)
    innovation_input=latent-(1-t)*self._anchor
    clean=self._clean_residual_backbone(innovation_input,timestep,*args,**kwargs)
    assert isinstance(clean,torch.Tensor) and clean.shape==latent.shape
    return (self._anchor+clean-latent)/t


def current_rgb_mean_skip(self,latent,timestep,*args,**kwargs):
    """Keep observed RGB exactly; learn its conditional future innovation."""
    assert self._anchor is not None
    innovation=self._mean_skip_backbone(latent,timestep,*args,**kwargs)
    assert innovation.shape==self._anchor.shape
    return self._anchor.float()+innovation.float()


class SharedVisualAnchor(nn.Module):
    """Reuse the RGB encoder for spatial decoder context and the state code."""
    def __init__(self,encoder,mean,std,width,grid,patch,detach_context):
        super().__init__()
        # Registered once at model.encoder: one parameter set and optimizer state.
        object.__setattr__(self,'_encoder',encoder)
        self.proj=nn.Linear(encoder.config.hidden_size,width)
        self.register_buffer('image_mean',mean.clone(),persistent=False)
        self.register_buffer('image_std',std.clone(),persistent=False)
        self.grid=grid;self.patch=patch;self.detach_context=detach_context
        self.context_tokens=None

    def forward(self,anchor):
        tokens=self.context_tokens
        if tokens is None:
            from vforge.models.as_jepa.pixels import unpack
            with torch.autocast(anchor.device.type,enabled=False):
                image=(unpack(anchor.float(),patch=self.patch)+1.)/2.
                image=(image-self.image_mean)/self.image_std
                tokens=self._encoder(image,interpolate_pos_encoding=True).last_hidden_state[:,1:]
        assert tokens.shape[:2]==(len(anchor),self.grid*self.grid)
        return self.proj(tokens.detach() if self.detach_context else tokens)


def shared_visual_loss(self,batch,step=0,ramp_steps=0):
    bridge=self.renderer.anchor_embed;b,w=batch['pixels'].shape[:2]
    assert bridge.context_tokens is None and w>1
    def current_tokens(module,args):
        tokens=args[0]
        assert bridge.context_tokens is None and tokens.shape[0]==b*w
        # Select the initial frame of each physical camera stream only.
        initial=tokens.reshape(b,w,*tokens.shape[1:])[:,0,1:]
        bridge.context_tokens=initial.repeat_interleave(w-1,dim=0)
    hook=self.pool.register_forward_pre_hook(current_tokens)
    try:
        return self._shared_visual_base_loss(batch,step,ramp_steps)
    finally:
        hook.remove();bridge.context_tokens=None


@torch.no_grad()
def render_future_mean(self,states,anchor,steps=1,seed=0):
    """Deterministic conditional mean; diffusion steps and seed are unused."""
    from vforge.models.as_jepa.pixels import pack,unpack
    assert self.pixel_objective=='future-mean'
    patch=self.raw_render_size//16;anchor=pack(self.render_anchor(anchor),patch=patch)
    if states.shape[-1]==self.state_dim:states=states.unflatten(-1,(self.state_slots,self.state_width))
    with self.renderer.anchored(anchor):
        value=self.renderer(torch.zeros_like(anchor),torch.ones(len(anchor),1,device=anchor.device),condition=states)
    return unpack(value,patch=patch)


def build(root,cfg):
    from experiments.robocasa_compare import train as common
    common.install_runtime(root/'code')
    from experiments.robocasa_refine.analytic_model import AnalyticResidualSPT
    torch.manual_seed(61001+cfg['seed'])
    model=AnalyticResidualSPT(mode='L2',beta=.25,sigma0=.1,window=8,anchor_gap=0,
        renderer_width=384,renderer_depth=12,renderer_heads=6,action_dim=14,
        rho=.8,forecast_fraction=.5,forecast_seed=61001+cfg['seed'],rate_prior='forecast',
        flow_logit_shift=cfg.get('flow_logit_shift',0.),state_slots=cfg.get('state_slots',4))
    model.code_chart='global'
    model.raw_render_size=cfg.get('raw_render_size',64)
    assert model.raw_render_size in (64,224)
    if model.raw_render_size!=64:
        from vforge.models.as_jepa.renderer import AnchoredWorldDiT
        # Keep the 16x16 token grid and the existing per-frame coding penalty.
        # Only input/output projection shapes grow; common random weights pair.
        original=model.renderer.state_dict()
        with torch.random.fork_rng(devices=[]):
            renderer=AnchoredWorldDiT(replace(model.renderer.config,channels=3*(model.raw_render_size//16)**2),
                state_dim=model.state_width,num_slots=model.state_slots)
        state=renderer.state_dict();assert state.keys()==original.keys()
        state.update({k:v for k,v in original.items() if state[k].shape==v.shape})
        renderer.load_state_dict(state,strict=True);renderer.sequence_parallel=False
        model.renderer=renderer
        model.distortion=types.MethodType(resolution_distortion,model)
    allocation = cfg.get('raw_encoder_heavy',False)
    if allocation or 'raw_encoder_width' in cfg or 'raw_decoder_depth' in cfg:
        assert cfg.get('raw_shared_visual_anchor') and cfg.get('pixel_objective')=='future-mean'
        from experiments.robocasa_stable_scaling.encoder_allocation import install
        install(model,cfg['seed'],encoder_width=cfg.get('raw_encoder_width',384 if allocation else 192),
            decoder_depth=cfg.get('raw_decoder_depth',6 if allocation else 12))
    if cfg.get('raw_shared_visual_anchor',False):
        assert model.raw_render_size==224 and cfg.get('pixel_objective')=='future-mean'
        assert not cfg.get('raw_clean_residual_velocity',False)
        model.renderer.anchor_embed=SharedVisualAnchor(model.encoder,model.image_mean,model.image_std,
            model.renderer.config.width,16,14,cfg.get('raw_shared_anchor_detach',False))
        model._shared_visual_base_loss=model.loss
        model.loss=types.MethodType(shared_visual_loss,model)
        model.render_states=types.MethodType(render_future_mean,model)
    if cfg.get('raw_clean_residual_velocity',False):
        model.renderer._clean_residual_backbone=model.renderer.forward
        model.renderer.forward=types.MethodType(clean_residual_velocity,model.renderer)
    if cfg.get('raw_future_mean_rgb_skip',False):
        assert cfg.get('raw_shared_visual_anchor') and cfg.get('pixel_objective')=='future-mean'
        assert not cfg.get('raw_clean_residual_velocity',False)
        model.renderer._mean_skip_backbone=model.renderer.forward
        model.renderer.forward=types.MethodType(current_rgb_mean_skip,model.renderer)
    model.prepare=types.MethodType(prepare_pixels,model)
    return model


def backbone_summary(tokens,feature_mode,has_cls=True):
    """Fixed per-image summaries before the learned stochastic bottleneck."""
    assert feature_mode in ('cls','mean','spatial-moments','spatial-flat')
    if feature_mode=='cls':
        assert has_cls
        return tokens[:,0].float()
    patches=(tokens[:,1:] if has_cls else tokens).float()
    if feature_mode=='spatial-flat':return patches.flatten(1)
    mean=patches.mean(1)
    if feature_mode=='mean':return mean
    grid=int(patches.shape[1]**.5);assert grid*grid==patches.shape[1] and grid>1
    coordinate=torch.linspace(-1,1,grid,device=patches.device,dtype=patches.dtype)
    y,x=torch.meshgrid(coordinate,coordinate,indexing='ij')
    modes=torch.stack((x,y)).flatten(1)
    modes=modes/modes.square().mean(-1,keepdim=True).sqrt()/patches.shape[1]
    moments=torch.einsum('pn,bnd->bpd',modes,patches)
    return torch.cat((mean.unsqueeze(1),moments),1).flatten(1)


class RawPixelsReadout(nn.Module):
    def __init__(self,root,cfg,weights,expected_hash,feature_mode='code'):
        super().__init__()
        assert feature_mode in ('code','cls','mean','spatial-moments','spatial-flat','renderer-mean','renderer-spatial-moments')
        self.feature_mode=feature_mode
        self.physical_joint=cfg.get('raw_physical_joint',False)
        if self.physical_joint:assert feature_mode=='code'
        from experiments.robocasa_stable_scaling.train import build as build_full
        from experiments.robocasa_compare import train as common
        model=build_full(root,cfg)
        model.load_state_dict(weights,strict=True)
        assert common.state_hash(model.state_dict())==expected_hash
        self.encoder=model.encoder;self.pool=model.pool;self.size=model.encoder_size
        if feature_mode.startswith('renderer-'):
            assert model.raw_render_size==224,'Renderer feature readout requires the native224 model'
            self.renderer=model.renderer;self.state_slots=model.state_slots;self.state_width=model.state_width
            self._renderer_ops_ready=False
        self.register_buffer('image_mean',model.image_mean.clone(),persistent=False)
        self.register_buffer('image_std',model.image_std.clone(),persistent=False)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self,pixels):
        if self.physical_joint and len(pixels)>2:
            assert len(pixels)%2==0
            return torch.cat([self.forward(pixels[i:i+2]) for i in range(0,len(pixels),2)])
        with torch.autocast('cuda',enabled=False):
            if self.feature_mode.startswith('renderer-'):
                return backbone_summary(self.renderer_tokens(pixels),self.feature_mode.removeprefix('renderer-'),has_cls=False)
            value=pixels.float()/255.
            if value.shape[-1]!=self.size:
                value=F.interpolate(value,size=(self.size,self.size),mode='bilinear',align_corners=False,antialias=True)
            value=(value-self.image_mean)/self.image_std
            tokens=self.encoder(value,interpolate_pos_encoding=True).last_hidden_state
            if self.feature_mode=='code':return self.pool(tokens).flatten(-2).float()
            return backbone_summary(tokens,self.feature_mode)

    @torch.no_grad()
    def renderer_tokens(self,pixels):
        """Unconditional current-anchor features at the pure-noise endpoint.

        Zero is the Gaussian mean. No future image or action enters this
        deterministic image feature map; the renderer's own weights are frozen.
        """
        from vforge.models.as_jepa.pixels import pack
        if not self._renderer_ops_ready:
            from vforge.ops import OpsConfig,initialize_ops
            initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=pixels.device,parallel_state=None)
            self._renderer_ops_ready=True
        assert pixels.shape[-2:]==(224,224)
        anchor=pack(pixels.float()/127.5-1.,patch=14);n=len(anchor)
        timestep=torch.ones(n,1,device=pixels.device)
        condition=torch.zeros(n,self.state_slots,self.state_width,device=pixels.device)
        drop=torch.ones(n,dtype=torch.bool,device=pixels.device);captured=[]
        hook=self.renderer.final_layer.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
        try:
            with self.renderer.anchored(anchor),torch.autocast('cuda',dtype=torch.bfloat16):
                self.renderer(torch.zeros_like(anchor),timestep,condition=condition,condition_drop=drop)
        finally:
            hook.remove()
        assert len(captured)==1 and captured[0].shape==(n,256,self.renderer.config.width)
        return captured[0].float()
