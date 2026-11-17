"""Factual forward prediction from one synchronized physical observation."""
import types

import torch
from torch import nn


class JointPool(nn.Module):
    """Keep the cache's two 64D rows as halves of one joint 128D state."""
    def __init__(self, inner):
        super().__init__()
        assert len(inner.queries) == 8
        self.inner = inner
        self.view_embedding = nn.Parameter(torch.zeros(2, 1, inner.queries.shape[-1]))

    def forward(self, tokens):
        n, length, width = tokens.shape
        assert n % 2 == 0
        pair = tokens.reshape(n//2, 2, length, width)
        joint = self.inner((pair+self.view_embedding).flatten(1, 2))
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
    model.pool = JointPool(model.pool)
    model.encode = types.MethodType(encode, model)
