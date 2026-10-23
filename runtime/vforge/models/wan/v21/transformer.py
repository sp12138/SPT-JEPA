# =============================================================================
# This file is a fork of diffusers' wan_transformers.py.
# All deltas vs. upstream are tagged with searchable markers:
#   [VS-UPSTREAM] high-level pointer to a diff region
#   [ADDED]       code that does not exist upstream
#   [CHANGED]     code that exists upstream but was modified here
#   [REMOVED-UPSTREAM] note that some upstream construct is intentionally gone
#   [BUG]         real defect (behavior-affecting), needs a fix
#   [STYLE]       cosmetic-only drift (imports / typing style / blank lines)
# Grep `[VS-UPSTREAM]` to enumerate every divergence at a glance.
# =============================================================================

import math

# [STYLE] upstream uses PEP604 (`int | None`, `tuple[...]`); this file mixes
# old-style typing imports below with new-style unions elsewhere.
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# [STYLE] upstream uses relative imports (`from ...configuration_utils ...`)
# because it lives inside diffusers; here we import diffusers as a 3rd party.
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import (
    AttentionMixin,
    FeedForward,
)
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

# [CHANGED] upstream imports only `apply_lora_scale` (used as decorator).
# Here we import the manual PEFT helpers because the decorator was dropped.
from diffusers.utils import (
    deprecate,
    logging,
)

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
    get_added_kv_projections as _get_added_kv_projections,
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

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanAttnProcessor:
    """Wan I2V cross-attention with optional image added-KV context."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self) -> None:
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade "
                "PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None and encoder_hidden_states is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img).unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))
            hidden_states_img = vforge_attention(
                query, key_img, value_img, attn_mask=None, is_causal=False
            )
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(query)

        hidden_states = vforge_attention(
            query, key, value, attn_mask=attention_mask, is_causal=False
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


class WanAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = (
            "The WanAttnProcessor2_0 class is deprecated and will be removed in a future "
            "version. Please use WanAttnProcessor instead. "
        )
        deprecate(
            "WanAttnProcessor2_0",
            "1.0.0",
            deprecation_message,
            standard_warn=False,
        )
        return WanAttnProcessor(*args, **kwargs)


class WanAttention(_WanAttention):
    _default_processor_cls = WanAttnProcessor
    _available_processors = [WanAttnProcessor]


# Temporary import compatibility for native tests and downstream forks.
WanUlyssesSelfAttentionProcessor = WanVforgeSelfAttentionProcessor


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(
                -1, 2 * seq_len, embed_dim
            )
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: int | None = None,
        dino_embed_dim: int | None = None,
        pos_embed_seq_len: int | None = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(
                image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len
            )

        self.dino_embedder = None
        if dino_embed_dim is not None:
            self.dino_embedder = WanImageEmbedding(
                dino_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len
            )

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        encoder_hidden_states_image_dino: torch.Tensor | None = None,
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
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        if encoder_hidden_states_image_dino is not None:
            encoder_hidden_states_image_dino = self.dino_embedder(encoder_hidden_states_image_dino)

        return (
            temb,
            timestep_proj,
            encoder_hidden_states,
            encoder_hidden_states_image,
            encoder_hidden_states_image_dino,
        )


class WanTransformerBlock(_WanTransformerBlock):
    """Wan21 block with optional image added-KV cross-attention."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
    ) -> None:
        super().__init__(
            dim,
            ffn_dim,
            num_heads,
            qk_norm,
            cross_attn_norm,
            eps,
            added_kv_proj_dim,
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
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections.
            If `None`, no projection is used.
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
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: int | None = None,
        dino_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: int | None = None,
        in_channels_image=16,
        hidden_size=5120,
        num_key_value_heads: int = 40,
        num_hidden_layers: int = 40,
        use_vae_condition: bool = True,
        use_clip_condition: bool = False,
        use_dino_condition: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels
        self.hidden_size = hidden_size
        self.use_vae_condition = use_vae_condition
        self.use_clip_condition = use_clip_condition
        self.use_dino_condition = use_dino_condition
        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )

        self.patch_embedding_image = None
        if use_vae_condition:
            self.patch_embedding_image = nn.Conv3d(
                in_channels_image, inner_dim, kernel_size=patch_size, stride=patch_size
            )

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            dino_embed_dim=dino_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def get_fsdp_precision_contract(self) -> FSDPPrecisionContract:
        return wan_fsdp_precision_contract()

    # [REMOVED-UPSTREAM] upstream has `@apply_lora_scale("attention_kwargs")`
    # here; we took the manual `scale_lora_layers` path below instead.
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image_vae: torch.Tensor | None = None,
        encoder_hidden_states_image_clip: torch.Tensor | None = None,
        encoder_hidden_states_image_dino: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        y=None,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        apply_lora_scale(self, attention_kwargs, logger=logger)

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        patch_grid = (num_frames // p_t, height // p_h, width // p_w)
        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states)
        if encoder_hidden_states_image_vae is not None:
            encoder_hidden_states_image_vae = (
                self.patch_embedding_image(encoder_hidden_states_image_vae)
                .flatten(2)
                .transpose(1, 2)
            )
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None
        (
            temb,
            timestep_proj,
            encoder_hidden_states,
            encoder_hidden_states_image_clip,
            encoder_hidden_states_image_dino,
        ) = self.condition_embedder(
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image_clip,
            encoder_hidden_states_image_dino,
            timestep_seq_len=ts_seq_len,
        )
        timestep_proj = timestep_proj.unflatten(
            2 if ts_seq_len is not None else 1,
            (6, -1),
        )

        context_parts = [
            part
            for part in (
                encoder_hidden_states_image_vae,
                encoder_hidden_states_image_clip,
                encoder_hidden_states_image_dino,
                encoder_hidden_states,
            )
            if part is not None
        ]
        encoder_hidden_states = torch.cat(context_parts, dim=1) if context_parts else None
        hidden_states, rotary_emb, _, sp_context = prepare_ulysses_inputs(
            hidden_states,
            rotary_emb,
            blocks=self.blocks,
            parallel_state=get_parallel_state(),
        )
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
