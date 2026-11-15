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
    if cfg.get('raw_clean_residual_velocity',False):
        model.renderer._clean_residual_backbone=model.renderer.forward
        model.renderer.forward=types.MethodType(clean_residual_velocity,model.renderer)
    model.prepare=types.MethodType(prepare_pixels,model)
    return model


class RawPixelsReadout(nn.Module):
    def __init__(self,root,cfg,weights,expected_hash):
        super().__init__()
        from experiments.robocasa_stable_scaling.train import build as build_full
        from experiments.robocasa_compare import train as common
        model=build_full(root,cfg)
        model.load_state_dict(weights,strict=True)
        assert common.state_hash(model.state_dict())==expected_hash
        self.encoder=model.encoder;self.pool=model.pool;self.size=model.encoder_size
        self.register_buffer('image_mean',model.image_mean.clone(),persistent=False)
        self.register_buffer('image_std',model.image_std.clone(),persistent=False)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self,pixels):
        with torch.autocast('cuda',enabled=False):
            value=pixels.float()/255.
            if value.shape[-1]!=self.size:
                value=F.interpolate(value,size=(self.size,self.size),mode='bilinear',align_corners=False,antialias=True)
            value=(value-self.image_mean)/self.image_std
            tokens=self.encoder(value,interpolate_pos_encoding=True).last_hidden_state
            return self.pool(tokens).flatten(-2).float()
