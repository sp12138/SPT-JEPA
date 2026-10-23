"""Wan Taichu's Diffusers-compatible model and Native ownership wrapper.

The model deliberately reuses Diffusers' public modeling primitives.  This
package only owns the Wan structure and forward paths that Taichu changes for
vforge attention, conditioning, expert routing, and teacher forcing; it is not
maintained as a line-by-line vendored Diffusers fork.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from diffusers.utils import (
    deprecate,
    logging,
)
from torch.utils.checkpoint import checkpoint

from vforge.distributed.parallel_state import get_parallel_state
from vforge.distributed.parallelize import FSDPPrecisionContract
from vforge.models.causal.context import get_incremental_kv_cache
from vforge.models.factory import patch_diffusers_getattr
from vforge.models.loading import cast_module_parameters
from vforge.models.wan_taichu.attention import (
    WanAttention as _WanAttention,
)
from vforge.models.wan_taichu.attention import (
    WanVforgeSelfAttentionProcessor,
)
from vforge.models.wan_taichu.attention import (
    apply_rotary_emb as _apply_rotary_emb,
)
from vforge.models.wan_taichu.attention import (
    get_added_kv_projections as _get_added_kv_projections,
)
from vforge.models.wan_taichu.attention import (
    get_qkv_projections as _get_qkv_projections,
)
from vforge.models.wan_taichu.attention import (
    narrow_qk_to_value as _narrow_qk_to_value,
)
from vforge.models.wan_taichu.audio import (
    AudioInjectorWAN,
    CausalAudioEncoder,
    build_local_audio_context,
)
from vforge.models.wan_taichu.teacher_forcing import WanTeacherForcingForwardMixin
from vforge.models.wan_taichu.transformer import (
    WanRotaryPosEmbed,
    apply_lora_scale,
    gather_ulysses_output,
    prepare_ulysses_inputs,
    project_output,
    run_transformer_blocks,
    unpatchify,
    wan_fsdp_precision_contract,
)
from vforge.models.wan_taichu.transformer import (
    WanTransformerBlock as _WanTransformerBlock,
)
from vforge.ops import attention as vforge_attention

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

WanTaichuArchitecture = Literal["wan21", "wan22_a14b", "wan22_ti2v_5b"]
WanTaichuAttention = Literal["bidirectional", "block_causal_teacher"]


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
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if (
            attn.add_k_proj is not None
            and encoder_hidden_states is not None
            and encoder_hidden_states.shape[1] > 512
        ):
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
            attend_query_img, attend_key_img = _narrow_qk_to_value(query, key_img, value_img)
            hidden_states_img = vforge_attention(
                attend_query_img, attend_key_img, value_img, attn_mask=None, is_causal=False
            )
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(query)

        attend_query, attend_key = _narrow_qk_to_value(query, key, value)
        hidden_states = vforge_attention(
            attend_query, attend_key, value, attn_mask=attention_mask, is_causal=False
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
        upcast_shared_text_context: bool = True,
    ):
        super().__init__()

        self.upcast_shared_text_context = bool(upcast_shared_text_context)

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
        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep_seq_len is not None and timestep.ndim == 1:
            # Per-token timesteps repeat heavily: a whole frame shares one value
            # and the anchored frame's is zero, so there are a couple of distinct
            # values behind tens of thousands of tokens. The reference embeds only
            # those and indexes them back out, which is not only the memory win it
            # is written for: it moves the token-axis reduction of this module's
            # backward into an index_add over the distinct rows instead of leaving
            # it inside the embedder's matmul. These gradients cancel to five
            # orders of magnitude, so the two reductions do not round alike.
            unique, inverse = torch.unique(timestep, sorted=True, return_inverse=True)
            frequencies = self.timesteps_proj(unique)
            if frequencies.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
                frequencies = frequencies.to(time_embedder_dtype)
            unique_temb = self.time_embedder(frequencies).type_as(encoder_hidden_states)
            unique_proj = self.time_proj(self.act_fn(unique_temb))
            temb = unique_temb.index_select(0, inverse).unflatten(0, (-1, timestep_seq_len))
            timestep_proj = unique_proj.index_select(0, inverse).unflatten(
                0, (-1, timestep_seq_len)
            )
        else:
            timestep = self.timesteps_proj(timestep)
            if timestep_seq_len is not None:
                timestep = timestep.unflatten(0, (-1, timestep_seq_len))

            if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
                timestep = timestep.to(time_embedder_dtype)
            temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
            timestep_proj = self.time_proj(self.act_fn(temb))

        # Every Transformer block consumes the same text context. Keeping that
        # shared fan-in edge in FP32 makes the gradients arriving from all the
        # cross-attention blocks accumulate with the same rounding semantics in
        # both the 5B and A14B production variants; each block casts its local
        # attention view back to the compute dtype immediately before
        # cross-attention, so the forward is unaffected either way. The
        # reference accumulates that fan-in at the compute dtype instead, which
        # is what bit-level comparisons against it have to reproduce.
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if self.upcast_shared_text_context:
            encoder_hidden_states = encoder_hidden_states.float()
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


class WanTaichuTransformer3DModel(
    WanTeacherForcingForwardMixin,
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
    # Taichu trains with one BF16 working copy of every model parameter.  FP32
    # master parameters belong to the optimizer, while numerically sensitive
    # operations (for example FP32LayerNorm) promote internally.  Do not turn
    # Diffusers' inference-oriented keep-in-FP32 hints into parameter-storage or
    # FSDP mixed-precision exemptions for this training model.
    _keep_in_fp32_modules: list[str] = []
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
        use_vae_condition: bool | None = None,
        use_clip_condition: bool = False,
        use_dino_condition: bool = False,
        use_ref_token: bool = False,
        in_channels_ref: int = 48,
        use_audio_condition: bool = False,
        audio_dim: int = 1024,
        audio_num_layers: int = 25,
        num_audio_token: int = 4,
        audio_inject_layers_s: Sequence[int] | None = None,
        audio_inject_layers_l: Sequence[int] | None = None,
        audio_window_s: int = 0,
        audio_window_l: int = -1,
        audio_zero_init: bool = False,
        architecture: WanTaichuArchitecture = "wan21",
        attention: WanTaichuAttention = "bidirectional",
        frame_per_block: int = 3,
        first_frame_own_block: bool = True,
        local_attn_size: int = -1,
        sink_size: int = 0,
        use_fsdp_precision_contract: bool = True,
        upcast_shared_text_context: bool = True,
        **kwargs,
    ) -> None:
        if architecture not in {"wan21", "wan22_a14b", "wan22_ti2v_5b"}:
            raise ValueError(f"unsupported Wan Taichu architecture {architecture!r}")
        if attention not in {"bidirectional", "block_causal_teacher"}:
            raise ValueError(f"unsupported Wan Taichu attention {attention!r}")
        if attention == "block_causal_teacher" and architecture not in {
            "wan22_a14b",
            "wan22_ti2v_5b",
        }:
            raise ValueError(
                "block_causal_teacher is only supported by wan22_a14b and wan22_ti2v_5b"
            )
        if architecture == "wan21" and attention != "bidirectional":
            raise ValueError(f"{architecture} does not support attention={attention!r}")
        if use_ref_token and architecture != "wan22_ti2v_5b":
            raise ValueError("ref tokens are only supported by wan22_ti2v_5b")
        if use_ref_token and attention != "bidirectional":
            raise ValueError("ref tokens require bidirectional attention")
        if use_audio_condition and architecture != "wan21":
            raise ValueError("audio conditioning is only supported by Wan21")
        if use_audio_condition and attention != "bidirectional":
            raise ValueError("audio conditioning requires bidirectional attention")
        if use_audio_condition and use_ref_token:
            raise ValueError("audio conditioning and ref tokens cannot be enabled together")
        if in_channels_ref < 1:
            raise ValueError(f"in_channels_ref must be positive; got {in_channels_ref}")

        speak_layers = tuple(int(index) for index in (audio_inject_layers_s or ()))
        listen_layers = tuple(int(index) for index in (audio_inject_layers_l or ()))
        if use_audio_condition:
            if audio_dim < 1 or audio_num_layers < 1 or num_audio_token < 1:
                raise ValueError(
                    "audio_dim, audio_num_layers, and num_audio_token must be positive"
                )
            if not speak_layers and not listen_layers:
                raise ValueError("audio conditioning requires speak or listen injection layers")
            invalid_layers = sorted(
                index
                for index in (*speak_layers, *listen_layers)
                if index < 0 or index >= num_layers
            )
            if invalid_layers:
                raise ValueError(
                    f"audio injection layers must be in [0, {num_layers}); got {invalid_layers}"
                )
            if len(set(speak_layers)) != len(speak_layers):
                raise ValueError("audio_inject_layers_s must not contain duplicates")
            if len(set(listen_layers)) != len(listen_layers):
                raise ValueError("audio_inject_layers_l must not contain duplicates")

        if use_vae_condition is None:
            use_vae_condition = architecture == "wan21"
        if architecture != "wan21":
            image_features = {
                "use_vae_condition": use_vae_condition,
                "use_clip_condition": use_clip_condition,
                "use_dino_condition": use_dino_condition,
                "image_dim": image_dim,
                "dino_dim": dino_dim,
                "added_kv_proj_dim": added_kv_proj_dim,
                "pos_embed_seq_len": pos_embed_seq_len,
            }
            enabled = sorted(key for key, value in image_features.items() if value)
            if enabled:
                raise ValueError(
                    f"{architecture} has no Wan21 image side branch; got non-null {enabled}"
                )

        super().__init__()

        self.taichu_architecture = architecture
        self.taichu_attention = attention
        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels
        self.hidden_size = hidden_size
        self.use_vae_condition = use_vae_condition
        self.use_clip_condition = use_clip_condition
        self.use_dino_condition = use_dino_condition
        self.use_ref_token = bool(use_ref_token)
        self.use_audio_condition = bool(use_audio_condition)
        self.audio_dim = int(audio_dim)
        self.audio_num_layers = int(audio_num_layers)
        self.num_audio_token = int(num_audio_token)
        self.audio_window_s = int(audio_window_s)
        self.audio_window_l = int(audio_window_l)
        self._audio_enable_s = bool(speak_layers)
        self._audio_enable_l = bool(listen_layers)
        self._audio_zero_init = bool(audio_zero_init)
        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )
        self.patch_embedding_ref = None
        if self.use_ref_token:
            self.patch_embedding_ref = nn.Conv3d(
                in_channels_ref,
                inner_dim,
                kernel_size=patch_size,
                stride=patch_size,
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
            upcast_shared_text_context=upcast_shared_text_context,
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
        self.casual_audio_encoder = None
        self.audio_injector = None
        if self.use_audio_condition:
            self.casual_audio_encoder = CausalAudioEncoder(
                dim=self.audio_dim,
                num_layers=self.audio_num_layers,
                out_dim=inner_dim,
                num_token=self.num_audio_token,
            )
            self.audio_injector = AudioInjectorWAN(
                dim=inner_dim,
                num_heads=num_attention_heads,
                inject_layers_s=speak_layers,
                inject_layers_l=listen_layers,
            )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False
        self.use_fsdp_precision_contract = bool(use_fsdp_precision_contract)
        if attention == "block_causal_teacher":
            self._enable_teacher_forcing(
                frame_per_block=frame_per_block,
                first_frame_own_block=first_frame_own_block,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
            )

    def get_fsdp_precision_contract(self) -> FSDPPrecisionContract:
        if not self.use_fsdp_precision_contract:
            return FSDPPrecisionContract()
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
        ref_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        y=None,
        attention_kwargs: dict[str, Any] | None = None,
        global_has_ref: bool | None = None,
        audio_input_s: torch.Tensor | None = None,
        audio_input_l: torch.Tensor | None = None,
        kv_cache: Any | None = None,
        incremental_kv: bool = False,
        full_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_write_start: int = 0,
        commit_kv: bool = False,
        local_attn_size: int = -1,
        sink_size: int = 0,
        usp_size: int = 1,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.taichu_attention == "block_causal_teacher":
            if any(
                value is not None
                for value in (
                    encoder_hidden_states_image_vae,
                    encoder_hidden_states_image_clip,
                    encoder_hidden_states_image_dino,
                    ref_hidden_states,
                )
            ):
                raise ValueError("teacher forcing does not accept image or ref side conditions")
            if incremental_kv:
                if kv_cache is not None:
                    raise ValueError(
                        "incremental_kv cache must use the active call context, "
                        "not cross the FSDP forward boundary"
                    )
                kv_cache = get_incremental_kv_cache()
            if kv_cache is not None:
                if full_rotary_emb is None:
                    raise ValueError("incremental_kv requires full_rotary_emb")
                from vforge.models.wan_taichu.inference.incremental_kv import (
                    forward_incremental_kv_block,
                )

                return forward_incremental_kv_block(
                    self,
                    hidden_states,
                    timestep,
                    encoder_hidden_states,
                    kv_cache=kv_cache,
                    full_rotary_emb=full_rotary_emb,
                    kv_write_start=kv_write_start,
                    commit_kv=commit_kv,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    usp_size=usp_size,
                    attention_kwargs=attention_kwargs,
                )
            if (
                full_rotary_emb is not None
                or incremental_kv
                or kv_write_start
                or commit_kv
                or local_attn_size != -1
                or sink_size
                or usp_size != 1
            ):
                raise ValueError("incremental KV arguments require kv_cache")
            return self.forward_teacher_forcing(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
            )
        apply_lora_scale(self, attention_kwargs, logger=logger)

        if not self.use_audio_condition and (
            audio_input_s is not None or audio_input_l is not None
        ):
            raise ValueError("audio inputs require an audio-enabled Wan Taichu variant")
        if audio_input_s is not None and not self._audio_enable_s:
            raise ValueError("audio_input_s was provided but no speak injection layers are enabled")
        if audio_input_l is not None and not self._audio_enable_l:
            raise ValueError(
                "audio_input_l was provided but no listen injection layers are enabled"
            )

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        patch_grid = (num_frames // p_t, height // p_h, width // p_w)
        rotary_emb = self.rope(hidden_states)

        num_ref_tokens = 0
        ref_tokens = None
        ref_zero_dependency = None
        local_has_ref = ref_hidden_states is not None
        if global_has_ref is None:
            global_has_ref = local_has_ref
        elif local_has_ref and not global_has_ref:
            raise RuntimeError("global_has_ref=False but this rank received ref_hidden_states")
        if global_has_ref:
            projection = self.patch_embedding_ref
            if projection is None:
                raise RuntimeError(
                    "reference tokens are active but patch_embedding_ref is missing; "
                    "select the wan22_ti2v_5b_ref variant"
                )
            if local_has_ref:
                if ref_hidden_states.ndim != 5:
                    raise ValueError(
                        "ref_hidden_states must be [B,C,T,H,W]; "
                        f"got {tuple(ref_hidden_states.shape)}"
                    )
                expected = (batch_size, projection.in_channels)
                actual = (
                    int(ref_hidden_states.shape[0]),
                    int(ref_hidden_states.shape[1]),
                )
                if actual != expected or any(int(size) < 1 for size in ref_hidden_states.shape[2:]):
                    raise ValueError(
                        "ref_hidden_states must have positive temporal/spatial dimensions "
                        f"and batch/channel geometry={expected}; "
                        f"got {tuple(ref_hidden_states.shape)}"
                    )
                ref_cos, ref_sin = self.rope.build_ref(ref_hidden_states)
                rotary_emb = (
                    torch.cat([ref_cos, rotary_emb[0]], dim=1),
                    torch.cat([ref_sin, rotary_emb[1]], dim=1),
                )
                ref_tokens = projection(ref_hidden_states).flatten(2).transpose(1, 2)
                num_ref_tokens = int(ref_tokens.shape[1])
            else:
                dummy_ref = hidden_states.new_zeros((1, projection.in_channels, p_t, p_h, p_w))
                ref_zero_dependency = projection(dummy_ref).sum() * 0.0
        elif self.patch_embedding_ref is not None:
            # DeepSpeed's BF16 optimizer keeps a preallocated gradient buffer
            # for every parameter in its flat group.  A no-ref step therefore
            # reaches AdamW as an explicit zero gradient and continues updating
            # the ref projection from its existing moments.  Materialize the
            # same zero dependency under FSDP; leaving ``grad=None`` would make
            # torch.optim.AdamW skip the parameter entirely.
            projection = self.patch_embedding_ref
            dummy_ref = hidden_states.new_zeros((1, projection.in_channels, p_t, p_h, p_w))
            ref_zero_dependency = projection(dummy_ref).sum() * 0.0

        hidden_states = self.patch_embedding(hidden_states)
        if encoder_hidden_states_image_vae is not None:
            encoder_hidden_states_image_vae = (
                self.patch_embedding_image(encoder_hidden_states_image_vae)
                .flatten(2)
                .transpose(1, 2)
            )
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        if ref_tokens is not None:
            hidden_states = torch.cat([ref_tokens, hidden_states], dim=1)
        # Taichu casts the model buffers with the transformer, so RoPE reaches
        # every block in the same dtype as the patch tokens.  FSDP2 can retain
        # non-persistent RoPE buffers at their construction dtype; normalize at
        # the model boundary instead of relying on the wrapping backend.
        rotary_emb = tuple(value.to(dtype=hidden_states.dtype) for value in rotary_emb)

        if num_ref_tokens and timestep.ndim == 2:
            ref_timestep = timestep.new_zeros(timestep.shape[0], num_ref_tokens)
            timestep = torch.cat([ref_timestep, timestep], dim=1)

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
        spatial_tokens = patch_grid[1] * patch_grid[2]
        audio_ctx_s, audio_ctx_l = self._build_audio_contexts(
            audio_input_s=audio_input_s,
            audio_input_l=audio_input_l,
            num_latent_frames=patch_grid[0],
            hidden_dtype=hidden_states.dtype,
        )
        sequence_tensors = (timestep_proj, temb) if ts_seq_len is not None else ()
        hidden_states, rotary_emb, sharded_sequence_tensors, sp_context = prepare_ulysses_inputs(
            hidden_states,
            rotary_emb,
            blocks=self.blocks,
            parallel_state=get_parallel_state(),
            sequence_tensors=sequence_tensors,
        )
        if sharded_sequence_tensors:
            timestep_proj, temb = sharded_sequence_tensors
        if (audio_ctx_s is not None or audio_ctx_l is not None) and sp_context.enabled:
            if sp_context.backend != "ulysses":
                raise NotImplementedError(
                    "Wan Taichu audio supports Ulysses but not context parallelism"
                )
            expected_tokens = patch_grid[0] * spatial_tokens
            if expected_tokens % sp_context.size:
                raise ValueError(
                    "Wan Taichu audio with Ulysses requires T*S to divide usp_size; "
                    f"got {expected_tokens} tokens and usp_size={sp_context.size}"
                )

        def audio_kwargs_for(block_index: int) -> dict[str, Any]:
            if self.audio_injector is None:
                return {}
            attention_s, attention_l = self.audio_injector.for_block(block_index)
            if attention_s is None and attention_l is None:
                return {}
            return {
                "audio_ctx_s": audio_ctx_s,
                "audio_ctx_l": audio_ctx_l,
                "audio_attn_s": attention_s,
                "audio_attn_l": attention_l,
                "num_latent_frames": patch_grid[0],
                "spatial_tokens": spatial_tokens,
                "audio_usp_group": sp_context.group if sp_context.backend == "ulysses" else None,
                "audio_usp_size": sp_context.size if sp_context.backend == "ulysses" else 1,
            }

        hidden_states = run_transformer_blocks(
            self,
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
            block_kwargs=audio_kwargs_for if self.audio_injector is not None else None,
        )
        hidden_states = project_output(
            hidden_states,
            temb,
            scale_shift_table=self.scale_shift_table,
            norm_out=self.norm_out,
            proj_out=self.proj_out,
        )
        hidden_states = gather_ulysses_output(
            hidden_states,
            sp_context,
            backward_mode=getattr(self, "ulysses_gather_backward", None),
        )
        if num_ref_tokens:
            hidden_states = hidden_states[:, num_ref_tokens:]
        output = unpatchify(
            hidden_states,
            batch_size=batch_size,
            patch_grid=patch_grid,
            patch_size=(p_t, p_h, p_w),
        )
        if ref_zero_dependency is not None:
            output = output + ref_zero_dependency
        return output

    def _build_audio_contexts(
        self,
        *,
        audio_input_s: torch.Tensor | None,
        audio_input_l: torch.Tensor | None,
        num_latent_frames: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        speak = self._encode_audio(audio_input_s, num_latent_frames, hidden_dtype)
        listen = self._encode_audio(audio_input_l, num_latent_frames, hidden_dtype)
        return (
            build_local_audio_context(speak, self.audio_window_s) if speak is not None else None,
            build_local_audio_context(listen, self.audio_window_l) if listen is not None else None,
        )

    def _encode_audio(
        self,
        features: torch.Tensor | None,
        num_latent_frames: int,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if features is None:
            return None
        if self.casual_audio_encoder is None:
            raise RuntimeError("audio input reached a model without casual_audio_encoder")
        if features.ndim != 4:
            raise ValueError(
                f"audio features must be [B,num_layers,audio_dim,F]; got {tuple(features.shape)}"
            )
        if int(features.shape[1]) != self.audio_num_layers:
            raise ValueError(
                f"audio features declare {features.shape[1]} layers, "
                f"expected {self.audio_num_layers}"
            )
        if int(features.shape[2]) != self.audio_dim:
            raise ValueError(f"audio feature dim={features.shape[2]}, expected {self.audio_dim}")
        encoded = self.casual_audio_encoder(features)
        if encoded.shape[1] < num_latent_frames:
            if encoded.shape[1] == 0:
                raise ValueError("causal audio encoder produced zero frames")
            padding = encoded[:, -1:].expand(
                -1,
                num_latent_frames - encoded.shape[1],
                -1,
                -1,
            )
            encoded = torch.cat([encoded, padding], dim=1)
        elif encoded.shape[1] > num_latent_frames:
            encoded = encoded[:, :num_latent_frames]
        return encoded.to(dtype=dtype)

    @torch.no_grad()
    def zero_audio_output_projections(self) -> None:
        if self.audio_injector is not None:
            self.audio_injector.zero_output_projections()


@dataclass(frozen=True)
class WanTaichuChannelLayout:
    """Published and configured patch-embedding widths for one expert."""

    z_dim: int
    pretrained_in_channels: int
    in_channels: int

    def __post_init__(self) -> None:
        for name, value in (
            ("z_dim", self.z_dim),
            ("pretrained_in_channels", self.pretrained_in_channels),
            ("in_channels", self.in_channels),
        ):
            if int(value) <= 0:
                raise ValueError(f"Wan Taichu {name} must be positive; got {value}")

    @property
    def is_widened(self) -> bool:
        return self.in_channels > self.pretrained_in_channels


def read_published_transformer_config(
    path: str | Path,
    *,
    subfolder: str = "transformer",
) -> dict[str, Any]:
    """Read a published Diffusers config without depending on another Wan model."""
    if Path(path).suffix.lower() in {".pt", ".pth"}:
        raise ValueError(
            "state-dict initialization requires explicit Wan config; use a "
            "Diffusers pretrained directory for standalone materialization"
        )
    return dict(WanTaichuTransformer3DModel.load_config(str(path), subfolder=subfolder))


def published_z_dim(config: Mapping[str, Any]) -> int:
    """Latent width denoised by one published Transformer config."""
    return int(config.get("out_channels") or config["in_channels"])


def build_meta_transformer(
    config: Mapping[str, Any],
    *,
    architecture: WanTaichuArchitecture,
    attention: WanTaichuAttention,
    parameter_dtype: torch.dtype | None = None,
    frame_per_block: int = 3,
    first_frame_own_block: bool = True,
    local_attn_size: int = -1,
    sink_size: int = 0,
    use_fsdp_precision_contract: bool = True,
    upcast_shared_text_context: bool = True,
) -> WanTaichuTransformer3DModel:
    """Build the package-local Transformer structure without allocating weights."""
    resolved_config = dict(config)
    resolved_config["use_fsdp_precision_contract"] = use_fsdp_precision_contract
    resolved_config["upcast_shared_text_context"] = upcast_shared_text_context
    with torch.device("meta"):
        transformer = WanTaichuTransformer3DModel(
            architecture=architecture,
            attention=attention,
            frame_per_block=frame_per_block,
            first_frame_own_block=first_frame_own_block,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            **resolved_config,
        )
    if parameter_dtype is not None:
        cast_module_parameters(transformer, parameter_dtype)
    return transformer


_SINGLE_EXPERT = ("",)
_A14B_EXPERTS = ("wan", "wan_2")


class WanTaichuModel(nn.Module):
    """One Native lifecycle root for every Wan Taichu variant.

    Single-expert variants register one ``transformer`` subtree. A14B registers
    two instances of the same package-local Transformer as ``wan`` and
    ``wan_2`` so its historical checkpoint paths remain stable.
    """

    def __init__(
        self,
        *,
        variant: str,
        architecture: WanTaichuArchitecture,
        attention: WanTaichuAttention,
        transformers: Mapping[str, nn.Module],
        boundary_ratio: float | None = None,
        num_train_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        self.variant = str(variant)
        self.architecture = architecture
        self.attention = attention
        self.expert_names = tuple(transformers)

        if self.expert_names == _SINGLE_EXPERT:
            self.transformer = transformers[""]
            self.wan = None
            self.wan_2 = None
            self.boundary_ratio = None
        elif self.expert_names == _A14B_EXPERTS:
            if architecture != "wan22_a14b":
                raise ValueError(f"{variant} cannot use A14B expert paths")
            if boundary_ratio is None or not 0.0 < boundary_ratio < 1.0:
                raise ValueError(f"boundary_ratio must be in (0, 1); got {boundary_ratio}")
            self.transformer = None
            self.wan = transformers["wan"]
            self.wan_2 = transformers["wan_2"]
            self.boundary_ratio = float(boundary_ratio)
        else:
            raise ValueError(
                f"{variant} has unsupported checkpoint experts {self.expert_names}; "
                f"expected {_SINGLE_EXPERT} or {_A14B_EXPERTS}"
            )

        self.num_train_timesteps = int(num_train_timesteps)
        for transformer in self.cache_transformers():
            patch_diffusers_getattr(transformer)

    @classmethod
    def build_empty(
        cls,
        *,
        variant: str,
        architecture: WanTaichuArchitecture,
        attention: WanTaichuAttention,
        transformer_configs: Mapping[str, Mapping[str, Any]],
        expert_names: Sequence[str],
        parameter_dtype: torch.dtype | None = None,
        boundary_ratio: float | None = None,
        num_train_timesteps: int = 1000,
        frame_per_block: int = 3,
        first_frame_own_block: bool = True,
        local_attn_size: int = -1,
        sink_size: int = 0,
        use_fsdp_precision_contract: bool = True,
        upcast_shared_text_context: bool = True,
    ) -> WanTaichuModel:
        names = tuple(expert_names)
        if len(transformer_configs) != len(names) or set(transformer_configs) != set(names):
            raise ValueError(
                f"{variant} transformer configs must have keys {list(names)}; "
                f"got {list(transformer_configs)}"
            )
        transformers = {
            name: build_meta_transformer(
                transformer_configs[name],
                architecture=architecture,
                attention=attention,
                parameter_dtype=parameter_dtype,
                frame_per_block=frame_per_block,
                first_frame_own_block=first_frame_own_block,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                use_fsdp_precision_contract=use_fsdp_precision_contract,
                upcast_shared_text_context=upcast_shared_text_context,
            )
            for name in names
        }
        return cls(
            variant=variant,
            architecture=architecture,
            attention=attention,
            transformers=transformers,
            boundary_ratio=boundary_ratio,
            num_train_timesteps=num_train_timesteps,
        )

    def primary_transformer(self) -> nn.Module:
        if self.transformer is not None:
            return self.transformer
        assert self.wan is not None
        return self.wan

    def cache_transformers(self) -> tuple[nn.Module, ...]:
        if self.transformer is not None:
            return (self.transformer,)
        return tuple(expert for expert in (self.wan, self.wan_2) if expert is not None)

    def fsdp_execution_roots(self) -> tuple[nn.Module, ...]:
        return self.cache_transformers()

    @property
    def config(self):
        return self.primary_transformer().config

    @property
    def dtype(self) -> torch.dtype:
        transformer = self.primary_transformer()
        declared = getattr(transformer, "dtype", None)
        if isinstance(declared, torch.dtype):
            return declared
        parameter = next(transformer.parameters(), None)
        if parameter is None:
            raise RuntimeError("WanTaichuTransformer3DModel must own parameters")
        return parameter.dtype

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    ) -> None:
        checkpoint_kwargs = (
            {"use_reentrant": False}
            if gradient_checkpointing_kwargs is None
            else dict(gradient_checkpointing_kwargs)
        )
        checkpoint_func = partial(checkpoint, **checkpoint_kwargs)
        for transformer in self.fsdp_execution_roots():
            transformer.enable_gradient_checkpointing(gradient_checkpointing_func=checkpoint_func)

    @contextmanager
    def cache_context(self, name: str):
        with ExitStack() as stack:
            for transformer in self.cache_transformers():
                context = getattr(transformer, "cache_context", None)
                if callable(context):
                    stack.enter_context(context(name))
            yield

    @property
    def boundary_timestep(self) -> float:
        if self.boundary_ratio is None:
            raise AttributeError(f"{self.variant} has no A14B expert boundary")
        return self.boundary_ratio * self.num_train_timesteps

    def select_transformer(
        self,
        timestep: torch.Tensor,
        *,
        incremental_kv: bool = False,
    ) -> nn.Module:
        if self.transformer is not None:
            return self.transformer
        if timestep.numel() == 0:
            raise ValueError("cannot route an empty timestep to a Wan Taichu expert")
        routing_timestep = timestep
        if self.attention == "block_causal_teacher" and not incremental_kv:
            if timestep.ndim != 2 or timestep.shape[1] % 2:
                raise ValueError("Wan Taichu A14B teacher forcing needs a doubled [B, 2S] timestep")
            routing_timestep = timestep[:, timestep.shape[1] // 2 :]
            active = routing_timestep[routing_timestep != 0]
            if active.numel():
                routing_timestep = active
        first = routing_timestep.reshape(-1)[0]
        if routing_timestep.numel() > 1 and not bool(torch.all(routing_timestep == first)):
            raise ValueError(
                "Wan Taichu A14B routing needs one shared active noisy timestep per micro-batch; "
                "split the batch by high/low noise before the forward"
            )
        expert = self.wan if float(first) >= self.boundary_timestep else self.wan_2
        assert expert is not None
        return expert

    @property
    def blocks(self):
        return self.primary_transformer().blocks

    @property
    def _no_split_modules(self):
        return getattr(self.primary_transformer(), "_no_split_modules", None)

    def forward(
        self,
        *args: Any,
        timestep: torch.Tensor,
        expert_name: str | None = None,
        **kwargs: Any,
    ):
        if expert_name is None:
            transformer = self.select_transformer(
                timestep,
                incremental_kv=bool(kwargs.get("incremental_kv"))
                or kwargs.get("kv_cache") is not None,
            )
        else:
            if expert_name not in self.expert_names:
                raise ValueError(
                    f"unknown Wan Taichu expert {expert_name!r}; expected {self.expert_names}"
                )
            transformer = getattr(self, expert_name) if expert_name else self.transformer
            assert transformer is not None
        return transformer(*args, timestep=timestep, **kwargs)


__all__ = [
    "WanAttention",
    "WanAttnProcessor",
    "WanTaichuArchitecture",
    "WanTaichuAttention",
    "WanTaichuChannelLayout",
    "WanTaichuModel",
    "WanTaichuTransformer3DModel",
    "WanTransformerBlock",
    "build_meta_transformer",
    "published_z_dim",
    "read_published_transformer_config",
]
