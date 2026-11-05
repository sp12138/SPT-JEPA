"""Factual temporal metrics/inverse bridges, without optimal-action labels.

These are explicitly action-reconstruction controls, not a claim of a novel
policy objective. Local image features avoid the existing global64D bottleneck.
All normalization is per sample. The original Lightning trainer owns SGD,
optimizers, exact sample/RNG cursors, and full/portable checkpoints.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width))

    def forward(self, x):
        return F.silu(x+self.net(x))


class SpatialEncoder(nn.Module):
    def __init__(self, width=192):
        super().__init__()
        layers = []; previous = 3
        for channels in [32, 64, 128, width]:
            layers.extend([nn.Conv2d(previous, channels, 3, stride=2, padding=1),
                           nn.GroupNorm(8, channels), nn.SiLU(), Residual(channels)])
            previous = channels
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).flatten(2).transpose(1, 2)


class TemporalBridge(nn.Module):
    def __init__(self, visual_source='local', objective='flow', width=192, chunk=6,
                 action_dim=20, window=2, feature_dim=64, noise_odds=1., condition_horizon='teacher',
                 temporal_aux=True, history_context=False, freeze_visual_encoder=False, encoder_lr=None):
        super().__init__()
        assert visual_source in ['local', 'global', 'hybrid', 'patch', 'vit'] and objective in ['flow', 'mse', 'metric']
        assert condition_horizon in ['teacher', 'none']
        assert temporal_aux or (condition_horizon == 'none' and objective != 'metric')
        assert not history_context or visual_source in ['local', 'vit']
        assert objective != 'metric' or visual_source == 'global'
        self.visual_source = visual_source; self.objective = objective
        self.width = width; self.chunk = chunk; self.window = window; self.action_dim = action_dim
        self.noise_odds = noise_odds
        self.condition_horizon = condition_horizon
        self.temporal_aux = temporal_aux; self.history_context = history_context
        self.freeze_visual_encoder = freeze_visual_encoder; self.encoder_lr = encoder_lr
        # Native trainer metadata expects these interfaces. They carry no
        # parameters and do not alter this model's loss or optimizer groups.
        self.renderer = None
        self.encoder = SpatialEncoder(width) if visual_source in ['local', 'hybrid'] else nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, width))
        if visual_source == 'patch':
            self.encoder = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, width))
        if visual_source == 'vit':
            from transformers import ViTConfig, ViTModel
            assert width == 192
            self.encoder = ViTModel(ViTConfig(hidden_size=192, num_hidden_layers=12, num_attention_heads=3,
                                    intermediate_size=768, image_size=224, patch_size=14),
                                    add_pooling_layer=False, use_mask_token=False)
            if hasattr(self.encoder, 'set_attn_implementation'):
                self.encoder.set_attn_implementation('sdpa')
            self.register_buffer('pixel_mean', torch.tensor([.485,.456,.406])[None,:,None,None], persistent=False)
            self.register_buffer('pixel_std', torch.tensor([.229,.224,.225])[None,:,None,None], persistent=False)
        self.kind = nn.Parameter(torch.zeros(3 if history_context else 2, width))
        positions = 257 if visual_source == 'vit' else 256 if visual_source == 'patch' else 196 if visual_source in ['local', 'hybrid'] else 1
        self.position = nn.Parameter(torch.randn(1, positions, width)*.02)
        if temporal_aux:
            self.time_query = nn.Parameter(torch.randn(1, 1, width)*.02)
            self.horizon_predictor = nn.TransformerDecoder(nn.TransformerDecoderLayer(width, 3, width*4, dropout=0.,
                                                           activation='gelu', batch_first=True, norm_first=True), 2)
            self.time_readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        if objective != 'metric':
            self.action_input = nn.Linear(action_dim, width)
            self.action_position = nn.Parameter(torch.randn(1, chunk, width)*.02)
            self.condition = nn.Sequential(nn.Linear(18, width), nn.SiLU(), nn.Linear(width, width))
            self.predictor = nn.TransformerDecoder(nn.TransformerDecoderLayer(width, 3, width*4, dropout=0.,
                                                    activation='gelu', batch_first=True, norm_first=True), 3)
            self.action_readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))
        else:
            self.predictor = nn.Identity()
        # Append after the original modules so paired local/hybrid arms keep
        # every original initial tensor identical for a common random seed.
        if visual_source == 'hybrid':
            self.global_encoder = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, width), nn.LayerNorm(width))
        if history_context:
            self.history_encoder = nn.Sequential(nn.Linear(action_dim+1, width), nn.LayerNorm(width))
        if freeze_visual_encoder:
            self.encoder.requires_grad_(False)

    def memory(self, batch):
        if self.visual_source == 'vit':
            images = batch['pixels'].float(); b,t = images.shape[:2]
            value = (images.flatten(0,1)/255.-self.pixel_mean)/self.pixel_std
            # Keep BOTH the globalCLS and all256local contextual tokens.
            tokens = self.encoder(value, interpolate_pos_encoding=True).last_hidden_state.unflatten(0,(b,t))
        elif self.visual_source == 'patch':
            tokens = self.encoder(batch['patches'].float())
        elif self.visual_source in ['local', 'hybrid']:
            x = batch['pixels'].float()/127.5-1.
            b, t = x.shape[:2]
            tokens = self.encoder(x.flatten(0, 1)).unflatten(0, (b, t))
        else:
            tokens = self.encoder(batch['features']).unsqueeze(2)
        memory = (tokens+self.position[:, None]+self.kind[None, :, None]).flatten(1, 2)
        if self.visual_source == 'hybrid':
            global_tokens = self.global_encoder(batch['features'])+self.kind[None]
            memory = torch.cat([memory, global_tokens], 1)
        if self.history_context:
            previous = torch.cat([batch['previous_action'].float(), batch['history_valid'].float()[:,None]], -1)
            memory = torch.cat([memory, self.history_encoder(previous)[:,None]], 1)
        return memory

    def duration(self, memory):
        if not self.temporal_aux:
            return memory.new_zeros(len(memory))
        out = self.horizon_predictor(self.time_query.expand(len(memory), -1, -1), memory)
        return F.softplus(self.time_readout(out[:, 0]).squeeze(-1))

    def action_field(self, x, time, horizon, memory):
        if self.condition_horizon == 'none':
            # Integrate out the unknown logged horizon rather than substitute
            # a point estimate for a training-only known temporal offset.
            horizon = torch.zeros_like(horizon)
        frequencies = (2.**torch.arange(8, device=x.device))*math.pi
        angles = time[:, None]*frequencies
        values = torch.cat([angles.sin(), angles.cos(), time[:, None], horizon[:, None]], -1)
        condition = self.condition(values)
        queries = self.action_input(x)+self.action_position+condition[:, None]
        return self.action_readout(self.predictor(queries, memory))

    def loss(self, batch, step=0, ramp_steps=0):
        memory = self.memory(batch)
        target_horizon = torch.log1p(batch['horizon'].float())
        duration = self.duration(memory)
        time_loss = (duration-target_horizon).square().mean() if self.temporal_aux else duration.new_zeros(())
        if self.objective == 'metric':
            return time_loss, dict(loss=time_loss, temporal=time_loss)
        target = batch['action'].float()
        if self.objective == 'flow':
            u = torch.rand(len(target), device=target.device).clamp(1e-5, 1-1e-5)
            t = self.noise_odds*u/(1+(self.noise_odds-1)*u)
            noise = torch.randn_like(target)
            x = target*(1-t[:, None, None])+noise*t[:, None, None]
            prediction = self.action_field(x, t, target_horizon, memory)
            truth = noise-target
        else:
            prediction = self.action_field(torch.zeros_like(target), torch.zeros(len(target), device=target.device), target_horizon, memory)
            truth = target
        per_step = (prediction.float()-truth).square().mean(-1)
        per_sample = (per_step*batch['action_mask']).sum(-1)/batch['action_mask'].sum(-1).clamp_min(1)
        reconstruction = per_sample.mean()
        loss = reconstruction+time_loss
        return loss, dict(loss=loss, action_reconstruction=reconstruction, temporal=time_loss)

    @torch.inference_mode()
    def propose(self, batch, generator=None, steps=12):
        memory = self.memory(batch)
        horizon = self.duration(memory)
        if self.objective == 'mse':
            value = torch.zeros(len(memory), self.chunk, self.action_dim, device=memory.device)
            return self.action_field(value, value.new_zeros(len(value)), horizon, memory)
        assert self.objective == 'flow'
        value = torch.randn(len(memory), self.chunk, self.action_dim, device=memory.device, generator=generator)
        for i in range(steps):
            t = value.new_full((len(value),), 1-i/steps)
            value = value-self.action_field(value, t, horizon, memory)/steps
        return value

    def training_optimizer_config(self, max_steps):
        parameters = [p for p in self.parameters() if p.requires_grad]
        if self.encoder_lr is not None and not self.freeze_visual_encoder:
            visual = {id(p) for p in self.encoder.parameters()}
            groups = [dict(params=[p for p in parameters if id(p) in visual], lr=self.encoder_lr),
                      dict(params=[p for p in parameters if id(p) not in visual], lr=3e-4)]
            optimizer = torch.optim.AdamW(groups, lr=3e-4, weight_decay=1e-3)
        else:
            optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=1e-3)
        warmup = max(1, int(.01*max_steps))
        def schedule(step):
            if step < warmup:
                return max(1, step)/warmup
            return .5*(1+math.cos(math.pi*min(1., (step-warmup)/max(1, max_steps-warmup))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='step'))

    def audit_expected_gradients(self):
        result = dict(encoder='frozen' if self.freeze_visual_encoder else 'active')
        if self.temporal_aux:
            result.update(horizon_predictor='active', time_readout='active')
        if self.objective != 'metric':
            result.update(predictor='active', action_readout='active')
        if self.visual_source == 'hybrid':
            result['global_encoder'] = 'active'
        if self.history_context:
            result['history_encoder'] = 'active'
        return result
