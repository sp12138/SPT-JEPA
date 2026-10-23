from __future__ import annotations

import torch
import torch.nn as nn

from vforge.models.video_diffusion.wan import WanFlowMatchPipelineBase


def test_wan_flow_match_base_prepares_seeded_latents() -> None:
    transformer = nn.Linear(1, 1).to(dtype=torch.bfloat16)
    pipeline = WanFlowMatchPipelineBase(
        tokenizer=None,
        text_encoder=None,
        vae=None,
        transformer=transformer,
        scheduler=object(),
        vae_scale_factor_temporal=4,
        vae_scale_factor_spatial=8,
    )

    first = pipeline.prepare_latents(
        1,
        2,
        16,
        24,
        9,
        torch.float32,
        torch.device("cpu"),
        generator=torch.Generator().manual_seed(17),
    )
    second = pipeline.prepare_latents(
        1,
        2,
        16,
        24,
        9,
        torch.float32,
        torch.device("cpu"),
        generator=torch.Generator().manual_seed(17),
    )

    assert first.shape == (1, 2, 3, 2, 3)
    assert first.dtype is torch.float32
    assert torch.equal(first, second)
