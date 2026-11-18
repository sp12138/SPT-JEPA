"""Factual forward prediction from one synchronized physical observation."""
import math
import types

import torch
from torch import nn


def camera_coordinate_values(tokens):
    """Reserve value channels for each view's homogeneous patch coordinates."""
    _, views, length, width = tokens.shape
    side = math.isqrt(length - 1)
    assert views == 2 and width >= 6 and side > 0 and length == 1 + side**2
    axis = (2 * torch.arange(side, device=tokens.device, dtype=tokens.dtype) + 1) / side - 1
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    xy1 = torch.stack((x, y, torch.ones_like(x)), -1).reshape(-1, 3)
    coordinates = torch.cat((tokens.new_zeros(1, 3), xy1))
    empty = torch.zeros_like(coordinates)
    charts = torch.stack((torch.cat((coordinates, empty), -1), torch.cat((empty, coordinates), -1)))
    return torch.cat((charts.expand(tokens.shape[0], -1, -1, -1), tokens[..., 6:]), -1)


class JointPool(nn.Module):
    """Keep the cache's two 64D rows as halves of one joint 128D state."""
    def __init__(self, inner, coordinate_values=False, motion_consensus=False):
        super().__init__()
        assert len(inner.queries) == 8
        self.inner = inner
        self.view_embedding = nn.Parameter(torch.zeros(2, 1, inner.queries.shape[-1]))
        self.coordinate_values = coordinate_values
        self.motion_consensus = motion_consensus
        assert not (coordinate_values and motion_consensus)

    def camera_codes(self, tokens):
        """Use the same observation map for each camera, keeping its address."""
        n, length, width = tokens.shape
        pair = tokens.reshape(n//2, 2, length, width) + self.view_embedding
        codes = self.inner(pair.flatten(0, 1))
        return codes.reshape(n//2, 2, *codes.shape[1:])

    def forward(self, tokens):
        n, length, width = tokens.shape
        assert n % 2 == 0
        if self.motion_consensus:
            joint = self.camera_codes(tokens).mean(dim=1)
            return joint.reshape(n, 4, joint.shape[-1])
        pair = tokens.reshape(n//2, 2, length, width) + self.view_embedding
        keys = pair.flatten(1, 2)
        if self.coordinate_values:
            # The attention-weighted coordinate values give per-view moments
            # before the usual learned projections, with the same query/keys.
            values = camera_coordinate_values(pair).flatten(1, 2)
            q = self.inner.queries.expand(n//2, -1, -1)
            pooled, _ = self.inner.attn(q, keys, values, need_weights=False)
            joint = self.inner.head(self.inner.norm(pooled))
        else:
            joint = self.inner(keys)
        return joint.reshape(n, 4, joint.shape[-1])


def synchronize_tokens(tokens, physical_batch, window):
    """The logged layout is physical sample, camera, time, patch, channel."""
    return tokens.reshape(physical_batch, 2, window, *tokens.shape[1:]).permute(
        0, 2, 1, 3, 4).reshape(physical_batch*window*2, *tokens.shape[1:])


def encode(self, info):
    pixels = info['pixels']
    bv, window = pixels.shape[:2]
    assert bv % 2 == 0
    b = bv//2
    with torch.autocast('cuda', enabled=False):
        tokens = self.encoder(pixels.flatten(0, 1).float(),
            interpolate_pos_encoding=True).last_hidden_state
        paired = synchronize_tokens(tokens, b, window)
        if self.pool.motion_consensus:
            codes = self.pool.camera_codes(paired).reshape(b, window, 2, self.state_dim)
            info['camera_codes'] = codes
            info['emb'] = codes.mean(dim=2)
        else:
            info['emb'] = self.pool(paired).reshape(b, window, self.state_dim)
        if 'action' in info:
            commands = info['action'].reshape(b, 2, window, -1)
            assert torch.equal(commands[:, 0], commands[:, 1])
            info['act_emb'] = self.action_encoder(commands[:, 0].float())
    return info


def install(model, cfg):
    assert cfg.get('raw_physical_joint') and cfg.get('conditional_rate_only')
    assert not cfg.get('anchor_reader') and not cfg.get('joint_views')
    assert model.state_slots == 8 and model.state_width == 16
    assert not model.batchnorm_state and not model.batchnorm_projector
    model.pool = JointPool(model.pool, coordinate_values=cfg.get('joint_coordinate_values', False),
                           motion_consensus=cfg.get('camera_motion_consensus', False))
    model.encode = types.MethodType(encode, model)
