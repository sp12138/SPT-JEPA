"""Local residual readout on the existing full-capacity forecast predictor.

The observation carrier, native transformer body, loss, renderer and training
implementation are unchanged. This replaces only the final global linear map.
"""
import torch
from torch import nn
from torch.nn import functional as F

from as_jepa_lewm.spatial_carrier import SpatialCarrierLeWM


class LocalReadoutPredictor(nn.Module):
    def __init__(self, original, seed, direct_condition=False):
        super().__init__()
        if (original.input_dim, original.hidden_dim, original.output_dim) != (1024, 192, 1024):
            raise ValueError("Requires the reviewed1024-state/192-hidden native predictor")
        self.num_frames = original.num_frames
        self.input_dim, self.hidden_dim, self.output_dim = 1024, 192, 1024
        self.direct_condition = bool(direct_condition)
        self.pos_embedding = original.pos_embedding
        self.dropout = original.dropout
        self.transformer = original.transformer
        self.transformer.output_proj = nn.Identity()
        # Construct after the complete parent model using a private CPU stream.
        # Renderer/action/trunk initialization and the outer RNG stay unchanged.
        with torch.random.fork_rng(devices=[]):
            # torch.manual_seed also reseeds CUDA, which a CPU-only fork does
            # not restore. Set only the CPU state used by these CPU layers.
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(int(seed)).get_state())
            width = 9 * 16 + 192 + (192 if self.direct_condition else 0)
            self.readout = nn.Sequential(nn.Linear(width, 512), nn.GELU(), nn.Linear(512, 16))
            nn.init.zeros_(self.readout[-1].weight)
            nn.init.zeros_(self.readout[-1].bias)

    @staticmethod
    def neighborhoods(state):
        b, t, d = state.shape
        if d != 1024:
            raise ValueError("Requires the unchanged8x8 addressed16-channel state")
        grid = state.reshape(b * t, 8, 8, 16).permute(0, 3, 1, 2)
        return F.unfold(grid, kernel_size=3, padding=1).transpose(1, 2).reshape(b, t, 64, 144)

    def forward(self, state, action):
        hidden = self.transformer(self.dropout(state + self.pos_embedding[:, :state.shape[1]]), action)
        local = self.neighborhoods(state)
        context = hidden.unsqueeze(-2).expand(*hidden.shape[:2], 64, 192)
        parts = [local, context]
        if self.direct_condition:
            # The already available action condition reaches the shared local
            # readout without first passing through the zero-initialized gates.
            condition = self.transformer.cond_proj(action)
            parts.append(condition.unsqueeze(-2).expand(*condition.shape[:2], 64, 192))
        return self.readout(torch.cat(parts, dim=-1)).flatten(-2)


class LocalReadoutLeWM(SpatialCarrierLeWM):
    def __init__(self, local_readout_seed=1, local_readout_condition=False, **kwargs):
        if not isinstance(local_readout_condition, bool):
            raise ValueError('local_readout_condition must be boolean')
        if kwargs.get("spatial_mode", "local") != "local":
            raise ValueError("Local readout requires the original local carrier")
        if (kwargs.get("spatial_basis", "identity") != "identity"
                or kwargs.get("spatial_chart", "none") != "none"
                or kwargs.get("spatial_readout_mode", "none") != "none"
                or kwargs.get("spatial_state_interface", "chart") != "chart"
                or not kwargs.get("carry", True)):
            raise ValueError("Use unchanged raw local state with native residual carry")
        super().__init__(**kwargs)
        self.predictor = LocalReadoutPredictor(self.predictor, local_readout_seed, local_readout_condition)

    def audit_expected_gradients(self):
        result = super().audit_expected_gradients()
        result.update({"predictor.readout": "active", "action_encoder": "active"})
        if self.predictor.direct_condition:
            result['predictor.transformer.cond_proj'] = 'active'
        return result
