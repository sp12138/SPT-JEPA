"""Native vforge wrapper around the teacher-forcing Wan 2.2 TI2V transformer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from vforge.models.causal import CausalKVCache
from vforge.models.wan.model import WanSingleTransformerModel, build_meta_module
from vforge.models.wan.v22.ti2v_5b.teacher_forcing.transformer import (
    WanTeacherForcingTransformer3DModel,
)

DEFAULT_FRAME_PER_BLOCK = 3


class Wan22Ti2v5bTfModel(WanSingleTransformerModel):
    """Checkpoint-wrapper-friendly teacher-forcing Wan 2.2 TI2V transformer.

    Shares the bidirectional model's channel layout and weight shapes, so it
    initialises from the same published checkpoint and the same widened
    ``patch_embedding``. Only the causal block layout is extra.
    """

    @classmethod
    def build_empty(
        cls,
        config: Mapping[str, Any],
        *,
        parameter_dtype: torch.dtype | None = None,
        frame_per_block: int = DEFAULT_FRAME_PER_BLOCK,
        first_frame_own_block: bool = True,
    ) -> Wan22Ti2v5bTfModel:
        """Materialize only model structure; weights are loaded after FSDP."""
        transformer = build_meta_module(
            WanTeacherForcingTransformer3DModel,
            dict(config),
            parameter_dtype=parameter_dtype,
            frame_per_block=frame_per_block,
            first_frame_own_block=first_frame_own_block,
        )
        return cls(transformer)

    def init_kv_cache(
        self,
        *,
        batch_size: int,
        num_cache_tokens: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> CausalKVCache:
        return self.transformer.init_kv_cache(
            batch_size=batch_size,
            num_cache_tokens=num_cache_tokens,
            device=device,
            dtype=dtype,
        )

    def full_clip_rotary_emb(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transformer.full_clip_rotary_emb(hidden_states)

    def forward_kv_block(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.transformer.forward_kv_block(*args, **kwargs)


__all__ = ["DEFAULT_FRAME_PER_BLOCK", "Wan22Ti2v5bTfModel"]
