"""Scratch RGB encoder ablation using the same factual transitions and heads."""
import types

import torch
from torch import nn
from torch.nn import functional as F


def prepare_pixels(self,batch):
    render=batch['render'].float()/127.5-1.
    return self.normalize(batch['pixels']),render,render[:,0]


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
