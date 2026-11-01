"""Full native predictor blocks over64 addressed spatial tokens per frame.

The original body/conditioning tensors are transferred without reinitializing.
Only the input/output linear maps change to shared16-channel patch maps.
"""
import torch
from torch import nn
from torch.nn import functional as F

from as_jepa_lewm.patch_carrier import frame_mask
from as_jepa_lewm.spatial_carrier import SpatialCarrierLeWM


class SpatialAttention(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.heads, self.dropout = original.heads, original.dropout
        self.norm, self.to_qkv, self.to_out = original.norm, original.to_qkv, original.to_out

    def forward(self, state, mask):
        b, t, slots, width = state.shape
        tokens = self.norm(state).reshape(b, t * slots, width)
        q, k, v = [x.reshape(b, t * slots, self.heads, -1).transpose(1, 2)
                   for x in self.to_qkv(tokens).chunk(3, dim=-1)]
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0., is_causal=False)
        result = self.to_out(value.transpose(1, 2).reshape(b, t * slots, -1))
        return result.reshape(b, t, slots, width)


class SpatialBlock(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.attn = SpatialAttention(original.attn)
        self.mlp, self.norm1, self.norm2 = original.mlp, original.norm1, original.norm2
        self.adaLN_modulation = original.adaLN_modulation

    def forward(self, state, condition, mask):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = [x.unsqueeze(2)
            for x in self.adaLN_modulation(condition).chunk(6, dim=-1)]
        state = state + gate_a * self.attn(self.norm1(state) * (1 + scale_a) + shift_a, mask)
        return state + gate_m * self.mlp(self.norm2(state) * (1 + scale_m) + shift_m)


class SpatialNativePredictor(nn.Module):
    def __init__(self, original, seed):
        super().__init__()
        if (original.input_dim, original.hidden_dim, original.output_dim, original.depth,
                original.heads, original.dim_head, original.mlp_dim) != (1024, 192, 1024, 6, 16, 64, 2048):
            raise ValueError('Requires the complete original1024-state native predictor')
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError('Explicit nonnegative private projection seed required')
        self.num_frames, self.input_dim, self.hidden_dim, self.output_dim = 7, 1024, 192, 1024
        self.pos_embedding, self.dropout = original.pos_embedding, original.dropout
        self.transformer = original.transformer
        self.transformer.layers = nn.ModuleList([SpatialBlock(x) for x in self.transformer.layers])
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            self.transformer.input_proj = nn.Linear(16, 192)
            self.transformer.output_proj = nn.Linear(192, 16)
            nn.init.zeros_(self.transformer.output_proj.weight)
            nn.init.zeros_(self.transformer.output_proj.bias)

    def forward(self, state, action):
        if state.ndim != 3 or state.shape[-1] != 1024 or action.shape != state.shape or not 1 <= state.shape[1] <= 7:
            raise ValueError('Require matching[B,T,1024],1<=T<=7')
        b, t, _ = state.shape
        tokens = self.dropout(state + self.pos_embedding[:, :t]).reshape(b, t, 64, 16)
        hidden = self.transformer.input_proj(tokens)
        condition = self.transformer.cond_proj(action)
        mask = frame_mask(t, state.device)
        for block in self.transformer.layers:
            hidden = block(hidden, condition, mask)
        return self.transformer.output_proj(self.transformer.norm(hidden)).reshape(b, t, 1024)


class SpatialNativeLeWM(SpatialCarrierLeWM):
    def __init__(self, spatial_predictor_seed=1, **kwargs):
        if (kwargs.get('spatial_mode', 'local') != 'local' or kwargs.get('spatial_basis', 'identity') != 'identity'
                or kwargs.get('spatial_chart', 'none') != 'none' or kwargs.get('spatial_readout_mode', 'none') != 'none'
                or kwargs.get('spatial_state_interface', 'chart') != 'chart' or kwargs.get('carry', True) is not True
                or kwargs.get('window', 8) != 8):
            raise ValueError('Requires unchanged local1024 carrier,window8,native carry and goal coordinates')
        super().__init__(**kwargs)
        self.predictor = SpatialNativePredictor(self.predictor, spatial_predictor_seed)

    def audit_expected_gradients(self):
        return dict(super().audit_expected_gradients(), action_encoder='active')
