"""One physical state for synchronized RGB views; no cross-sample attention."""
import types

import torch
from torch import nn


class StereoPool(nn.Module):
    def __init__(self, pool, depth, seed, view_grouped=False):
        super().__init__()
        assert depth in (0, 2) and len(pool.queries) == 8
        self.inner = pool
        self.view_grouped = view_grouped
        width = pool.queries.shape[-1]
        self.view_embedding = nn.Parameter(torch.zeros(2, 1, width))
        self.fusion = nn.ModuleList()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(97620261116+seed)
            for _ in range(depth):
                layer = nn.TransformerEncoderLayer(width, 3, 4*width, dropout=0.,
                    activation="gelu", batch_first=True, norm_first=True)
                # Start with the same function as joint pooling. Additional
                # residual paths become trainable without changing the base NN.
                nn.init.zeros_(layer.self_attn.out_proj.weight)
                nn.init.zeros_(layer.self_attn.out_proj.bias)
                nn.init.zeros_(layer.linear2.weight)
                nn.init.zeros_(layer.linear2.bias)
                self.fusion.append(layer)

    def forward(self, tokens):
        n, length, width = tokens.shape
        assert n % 2 == 0
        pair = tokens.reshape(n//2, 2, length, width)[:, :, 1:]
        pair = (pair+self.view_embedding).flatten(1, 2)
        for layer in self.fusion:
            pair = layer(pair)
        if self.view_grouped:
            # Interaction above can use both views. Keep their source addresses
            # when compressing: each half of the queries reads one view.
            k = len(self.inner.queries)
            query_view = torch.arange(k, device=pair.device)//(k//2)
            key_view = torch.arange(pair.shape[1], device=pair.device)//(pair.shape[1]//2)
            mask = query_view[:, None] != key_view[None, :]
            q = self.inner.queries.expand(pair.shape[0], -1, -1)
            pooled, _ = self.inner.attn(q, pair, pair, attn_mask=mask, need_weights=False)
            joint = self.inner.head(self.inner.norm(pooled))
        else:
            joint = self.inner(pair)
        # Preserve the existing cache's two rows per physical observation.
        # These are halves of one joint state, not camera-specific states.
        return joint.reshape(n, 4, joint.shape[-1])


def encode(self, info):
    pixels = info["pixels"]
    bv, window = pixels.shape[:2]
    assert bv % 2 == 0
    b = bv//2
    with torch.autocast("cuda", enabled=False):
        tokens = self.encoder(pixels.flatten(0, 1).float(),
            interpolate_pos_encoding=True).last_hidden_state
        if self._stereo_capture:
            bridge = self.renderer.anchor_embed
            assert bridge.context_tokens is None
            initial = tokens.reshape(bv, window, *tokens.shape[1:])[:, 0, 1:]
            bridge.context_tokens = initial.repeat_interleave(window-1, dim=0)
        # Dataset camera streams are camera-major; pool pairs at the same time.
        paired = tokens.reshape(b, 2, window, *tokens.shape[1:])
        paired = paired.permute(0, 2, 1, 3, 4).reshape(b*window*2, *tokens.shape[1:])
        info["emb"] = self.pool(paired).reshape(b, window, self.state_dim)
        if "action" in info:
            commands = info["action"].reshape(b, 2, window, -1)
            assert torch.equal(commands[:, 0], commands[:, 1])
            info["act_emb"] = self.action_encoder(commands[:, 0].float())
    return info


def loss(self, batch, step=0, ramp_steps=0):
    from experiments.robocasa_stable_scaling.train import joint_pixel_loss
    bridge = self.renderer.anchor_embed
    assert not self._stereo_capture and bridge.context_tokens is None
    self._stereo_capture = True
    try:
        return joint_pixel_loss(self, batch, step, ramp_steps)
    finally:
        self._stereo_capture = False
        bridge.context_tokens = None


def install(model, cfg):
    from experiments.robocasa_stable_scaling.train import joint_pixel_distortion
    assert cfg["raw_shared_visual_anchor"] and not cfg.get("joint_views", False)
    assert model.raw_render_size == 224 and model.pixel_objective == "future-mean"
    assert model.state_slots == 8 and model.state_width == 16
    assert not model.batchnorm_state and not model.batchnorm_projector
    model.pool = StereoPool(model.pool, cfg["stereo_fusion_depth"], cfg["seed"],
        cfg.get("stereo_view_grouped", False))
    model.joint_views = True
    model._stereo_capture = False
    model.encode = types.MethodType(encode, model)
    model.original_pixel_distortion = model.distortion
    model.distortion = types.MethodType(joint_pixel_distortion, model)
    model.loss = types.MethodType(loss, model)
