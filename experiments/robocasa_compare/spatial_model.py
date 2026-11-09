"""A spatially addressed SPT carrier, retaining the original sample-level loss."""
import math

import torch

from as_jepa_lewm.robust import RobustLeWM
from vforge.models.as_jepa.pooling import QueryPool


class GridPool(QueryPool):
    def __init__(self, width=192, heads=3, slots=16, channels=16, patch_side=16):
        super().__init__(width, heads, slots, channels)
        side = math.isqrt(slots)
        assert side * side == slots and patch_side % side == 0
        indices = torch.arange(patch_side * patch_side)
        labels = (indices // patch_side // (patch_side // side)) * side
        labels += indices % patch_side // (patch_side // side)
        mask = torch.ones(slots, 1 + patch_side * patch_side, dtype=torch.bool)
        # CLS is deliberately excluded: every slot reads its addressed region.
        mask[:, 1:] = labels[None] != torch.arange(slots)[:, None]
        self.register_buffer("spatial_mask", mask, persistent=False)
        self.initialize()

    def forward(self, tokens):
        assert tokens.shape[1] == self.spatial_mask.shape[1]
        query = self.queries.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(query, tokens, tokens, attn_mask=self.spatial_mask, need_weights=False)
        return self.head(self.norm(pooled))


class SpatialForecast(RobustLeWM):
    def __init__(self, **kwargs):
        assert kwargs.get("state_slots") == 16 and kwargs.get("state_width") == 16
        super().__init__(**kwargs)
        self.pool = GridPool()
