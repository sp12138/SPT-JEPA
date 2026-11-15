"""Move predictive capacity before the frozen representation readout."""
from dataclasses import replace

import torch
from transformers import ViTConfig, ViTModel


def install(model, seed):
    from vforge.models.as_jepa.pooling import QueryPool

    assert model.encoder.config.hidden_size == 192
    assert len(model.renderer.blocks) == model.renderer.config.depth == 12
    config = model.encoder.config.to_dict()
    config.update(hidden_size=384, intermediate_size=1536, num_attention_heads=6)
    # This is a fresh architecture, not a migration of trained weights.
    # Preserve the caller's RNG and all unchanged random decoder parameters.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(98720261116+seed)
        encoder = ViTModel(ViTConfig(**config), add_pooling_layer=False,
            use_mask_token=False)
        encoder.config.interpolate_pos_encoding = True
        if hasattr(encoder, "set_attn_implementation"):
            encoder.set_attn_implementation("sdpa")
        pool = QueryPool(384, 6, model.state_slots, model.state_width)
        pool.initialize()
    model.encoder, model.pool = encoder, pool
    model.renderer.blocks = model.renderer.blocks[:6]
    model.renderer.config = replace(model.renderer.config, depth=6)
