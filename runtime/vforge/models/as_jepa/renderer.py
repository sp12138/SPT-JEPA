import contextlib
import torch
from torch import nn
from vforge.models.as_jepa.constants import GRID, TOKEN_CH
from vforge.models.rep_world.config import WorldConfig
from vforge.models.rep_world.layers import PatchEmbed3D
from vforge.models.rep_world.world import WorldDiT

class SlotLinear(nn.Linear):
    """Linear projection of state tokens plus a learned embedding per token slot. No normalization."""
    def __init__(self, in_features, out_features, num_slots):
        super().__init__(in_features, out_features)
        self.slot = nn.Parameter(torch.zeros(num_slots, out_features))
    def forward(self, x):                          # x: [N, num_slots, in_features]
        return super().forward(x) + self.slot[: x.shape[1]]

class AnchoredWorldDiT(WorldDiT):
    """WorldDiT whose patch embedding also reads an anchor frame: channel concatenation
    implemented as a zero-initialized added embedding. state_dim=None keeps SPT's
    condition projector (used by S1); an integer replaces it by SlotLinear."""

    def __init__(self, config, state_dim=None, num_slots=None, use_anchor=True):
        super().__init__(config)
        c = config
        self.use_anchor = use_anchor
        if use_anchor:
            self.anchor_embed = PatchEmbed3D((c.state_planes, c.grid, c.grid), (1, c.patch_size, c.patch_size), c.channels, c.width)
            self.x_embedder.register_forward_hook(self._add_anchor)
        if state_dim is not None:
            if c.condition_width != state_dim:
                raise ValueError("condition_width must equal the state token width")
            self.semantic_raw_projector = nn.Sequential(nn.Identity(), SlotLinear(state_dim, c.width, num_slots))
        self._anchor = None
        self._init_added()

    def _init_added(self):
        if getattr(self, "use_anchor", False) and hasattr(self, "anchor_embed"):
            nn.init.zeros_(self.anchor_embed.proj.weight)
            nn.init.zeros_(self.anchor_embed.proj.bias)
        proj = self.semantic_raw_projector[1]
        if isinstance(proj, SlotLinear):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
            nn.init.normal_(proj.slot, std=0.02)

    def initialize_weights(self):                  # WorldDiT.__init__ calls this before the added modules exist
        super().initialize_weights()
        if hasattr(self, "use_anchor"):
            self._init_added()

    def _add_anchor(self, module, inputs, output):
        if self._anchor is None:
            raise RuntimeError("AnchoredWorldDiT used outside renderer.anchored(...)")
        return output + self.anchor_embed(self._anchor)

    @contextlib.contextmanager
    def anchored(self, anchor):                    # anchor: [N, TOKEN_CH, 1, GRID, GRID], one per rendered target
        self._anchor = anchor if self.use_anchor else None
        try:
            yield self
        finally:
            self._anchor = None

def renderer_config(state_width, condition_width=None, cross_attention_3d_rope=False):
    """The renderer for L1, L2 and S2 (S0 and S1 override the two keyword arguments, Sections 6.3 and 6.4).
    DiT-S size, plain output layer (no pixel refiner), cross-attention in blocks 0, 2, 4, 6, 8, 10."""
    return WorldConfig(channels=TOKEN_CH, state_planes=1, grid=GRID, width=384, depth=12, heads=6, patch_size=1,
                       mlp_ratio=4.0, qk_norm=True, adaln_mode="adaln_lora", adaln_lora_dim=256,
                       condition_width=condition_width or state_width, condition_dropout=0.1,
                       cross_attention_every=2, cross_attention_position="before_self",
                       cross_attention_scope="global", cross_attention_3d_rope=cross_attention_3d_rope,
                       condition_mode="cross_attention", pixel_depth=0)
