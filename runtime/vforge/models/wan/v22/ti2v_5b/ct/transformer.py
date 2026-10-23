"""Wan 2.2 TI2V transformer.

Same Diffusers DiT skeleton as ``vforge.models.wan.v21.transformer`` — the two
files are kept separate so each stays diffable against its own upstream, and so
the TI2V conditioning contract cannot silently regress into the I2V one. The
deltas that matter:

* ``in_channels=96`` (noisy 48 ‖ normal 48), ``out_channels=48``. The first
  frame is injected by temporal in-painting on the latent, not by a mask
  channel, so there is no ``patch_embedding_image`` side branch.
* No CLIP / DINO / ``added_kv_proj_dim`` image cross-attention. Text is the
  only cross-attention context.
* ``timestep`` is per token (``[B, seq]``), which makes AdaLN modulation
  sequence-shaped. Under Ulysses the modulation tensors therefore have to be
  padded and sharded alongside ``hidden_states``; the Wan2.1 scalar path has no
  sequence axis and needs none.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import AttentionMixin
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import logging

from vforge.distributed.parallel_state import get_parallel_state
from vforge.distributed.parallelize import FSDPPrecisionContract
from vforge.models.wan.attention import (
    WanAttention as _WanAttention,
)
from vforge.models.wan.attention import (
    WanVforgeSelfAttentionProcessor,
)
from vforge.models.wan.attention import (
    apply_rotary_emb as _apply_rotary_emb,
)
from vforge.models.wan.attention import (
    get_qkv_projections as _get_qkv_projections,
)
from vforge.models.wan.transformer import (
    WanRotaryPosEmbed,
    apply_lora_scale,
    gather_ulysses_output,
    prepare_ulysses_inputs,
    project_output,
    run_transformer_blocks,
    unpatchify,
    wan_fsdp_precision_contract,
)
from vforge.models.wan.transformer import (
    WanTransformerBlock as _WanTransformerBlock,
)
from vforge.ops import attention as vforge_attention

logger = logging.get_logger(__name__)


class WanAttnProcessor:
    """Text cross-attention. TI2V never carries image context here."""

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query, key, value = _get_qkv_projections(
            attn,
            hidden_states,
            encoder_hidden_states,
        )
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)
        hidden_states = vforge_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=False,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


class WanAttention(_WanAttention):
    _default_processor_cls = WanAttnProcessor
    _available_processors = [WanAttnProcessor]


class WanTimeTextEmbedding(nn.Module):
    """Timestep + text embedding. Accepts scalar or per-token timesteps."""

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
    ) -> None:
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim,
            time_embed_dim=dim,
        )
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim,
            dim,
            act_fn="gelu_tanh",
        )

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_seq_len: int | None = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        return temb, timestep_proj, encoder_hidden_states


class WanTransformerBlock(_WanTransformerBlock):
    """TI2V block with text-only cross-attention."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            dim,
            ffn_dim,
            num_heads,
            qk_norm,
            cross_attn_norm,
            eps,
            attention_type=WanAttention,
            cross_attention_processor=WanAttnProcessor(),
        )


class WanTransformer3DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    CacheMixin,
    AttentionMixin,
):
    r"""Wan 2.2 TI2V transformer with channel-concatenated normal conditioning.

    Args:
        in_channels (`int`, defaults to `96`):
            Noisy latent channels plus normal-condition latent channels.
        out_channels (`int`, defaults to `48`):
            VAE ``z_dim`` of the Wan2.2 VAE.
        image_dim / added_kv_proj_dim / pos_embed_seq_len:
            Accepted only as ``None``. TI2V has no image cross-attention; a
            non-null value means a Wan2.1 config reached this model.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        "patch_embedding",
        "condition_embedder",
        "norm",
    ]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        in_channels: int = 96,
        out_channels: int = 48,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 14336,
        num_layers: int = 30,
        cross_attn_norm: bool = True,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        rope_max_seq_len: int = 1024,
        image_dim: int | None = None,
        dino_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        pos_embed_seq_len: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        unsupported = {
            "image_dim": image_dim,
            "dino_dim": dino_dim,
            "added_kv_proj_dim": added_kv_proj_dim,
            "pos_embed_seq_len": pos_embed_seq_len,
        }
        enabled = sorted(key for key, value in unsupported.items() if value)
        if enabled:
            raise ValueError(
                "Wan22 TI2V has no image cross-attention branch; "
                f"got non-null {enabled}. Use vforge.models.wan.v21 for I2V "
                "configs that carry CLIP/DINO/VAE image conditions."
            )

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.rope = WanRotaryPosEmbed(
            attention_head_dim,
            patch_size,
            rope_max_seq_len,
        )
        self.patch_embedding = nn.Conv3d(
            in_channels,
            inner_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.condition_embedder = WanTimeTextEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
        )
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def get_fsdp_precision_contract(self) -> FSDPPrecisionContract:
        return wan_fsdp_precision_contract()

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        apply_lora_scale(self, attention_kwargs, logger=logger)

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        patch_grid = (num_frames // p_t, height // p_h, width // p_w)
        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)

        # [B, seq] is the TI2V per-token timestep; [B] is the scalar fallback.
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None
        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
            timestep,
            encoder_hidden_states,
            timestep_seq_len=ts_seq_len,
        )
        timestep_proj = timestep_proj.unflatten(
            2 if ts_seq_len is not None else 1,
            (6, -1),
        )

        sequence_tensors = (timestep_proj, temb) if ts_seq_len is not None else ()
        hidden_states, rotary_emb, sequence_tensors, sp_context = prepare_ulysses_inputs(
            hidden_states,
            rotary_emb,
            blocks=self.blocks,
            parallel_state=get_parallel_state(),
            sequence_tensors=sequence_tensors,
        )
        if ts_seq_len is not None:
            timestep_proj, temb = sequence_tensors
        hidden_states = run_transformer_blocks(
            self,
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
        )
        hidden_states = project_output(
            hidden_states,
            temb,
            scale_shift_table=self.scale_shift_table,
            norm_out=self.norm_out,
            proj_out=self.proj_out,
        )
        hidden_states = gather_ulysses_output(hidden_states, sp_context)
        return unpatchify(
            hidden_states,
            batch_size=batch_size,
            patch_grid=patch_grid,
            patch_size=(p_t, p_h, p_w),
        )


__all__ = [
    "WanAttention",
    "WanAttnProcessor",
    "WanRotaryPosEmbed",
    "WanTimeTextEmbedding",
    "WanTransformer3DModel",
    "WanTransformerBlock",
    "WanVforgeSelfAttentionProcessor",
]
