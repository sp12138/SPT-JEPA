"""Wan 2.2 TI2V transformer with a block-causal teacher-forcing forward.

Everything that carries weights -- patch embedding, condition embedder, blocks,
RoPE tables, output head -- is inherited unchanged from
``vforge.models.wan.v22.ti2v_5b.ct.transformer``. A teacher-forcing checkpoint is
therefore load-compatible with the bidirectional one; this file only changes
what the forward feeds those modules.

The teacher-forcing forward takes a doubled ``[clean | noisy]`` sequence on the
frame axis and restricts attention so that a noisy token sees its own noisy
block plus strictly-earlier *clean* frames, never its own clean answer and
never the future. That predicate reaches the kernel as a
:class:`~vforge.ops.TeacherForcingSparsity`, which the ops layer turns into
FA4 tile metadata (or, without FA4, into a dense SDPA mask).

Deltas against the bidirectional parent:

* RoPE is built for one half and tiled, so clean and noisy tokens of the same
  latent frame share a position. Position, not order, is what tells the model
  which frame a token belongs to.
* Only the noisy half survives to the output head, so the model still predicts
  ``T`` frames from a ``2T`` input.
* Ulysses shards the doubled sequence but refuses to pad it. The sparsity spec
  and its tile metadata are built for one exact sequence length, and padding
  would describe a mask the kernel is no longer computing.
"""

from __future__ import annotations

from typing import Any

import torch
from diffusers.utils import logging

from vforge.distributed.parallel_state import get_parallel_state
from vforge.models.causal import (
    CausalKVCache,
    IncrementalKVAttentionContext,
    IncrementalKVExecutor,
)
from vforge.models.wan.attention import (
    apply_rotary_emb as _apply_rotary_emb,
)
from vforge.models.wan.attention import (
    get_qkv_projections as _get_qkv_projections,
)
from vforge.models.wan.transformer import (
    apply_lora_scale,
    project_output,
    run_transformer_blocks,
    unpatchify,
)
from vforge.models.wan.v22.ti2v_5b.ct.transformer import (
    WanAttention,
    WanTransformer3DModel,
)
from vforge.ops import TeacherForcingSparsity
from vforge.ops import attention as vforge_attention

logger = logging.get_logger(__name__)


def forward_incremental_kv_block(
    transformer: WanTransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    *,
    kv_cache: CausalKVCache,
    full_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    kv_write_start: int,
    commit_kv: bool,
    usp_size: int = 1,
    attention_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Run one official Wan block against explicitly supplied KV history."""
    if int(usp_size) != 1:
        raise NotImplementedError(
            f"TI2V-5B incremental_kv has no Ulysses path; got usp_size={usp_size}"
        )
    apply_lora_scale(transformer, attention_kwargs, logger=logger)
    if len(kv_cache) != len(transformer.blocks):
        raise ValueError(
            f"KV cache has {len(kv_cache)} layers, transformer has {len(transformer.blocks)}"
        )

    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = (int(value) for value in transformer.config.patch_size)
    if num_frames % p_t or height % p_h or width % p_w:
        raise ValueError(
            f"block shape {tuple(hidden_states.shape[-3:])} must divide by "
            f"patch_size={tuple(transformer.config.patch_size)}"
        )
    patch_grid = (num_frames // p_t, height // p_h, width // p_w)
    block_tokens = patch_grid[0] * patch_grid[1] * patch_grid[2]
    if timestep.ndim != 2 or tuple(timestep.shape) != (batch_size, block_tokens):
        raise ValueError(
            "incremental_kv needs a per-token timestep matching the current block; "
            f"expected {(batch_size, block_tokens)}, got {tuple(timestep.shape)}"
        )

    rotary_start = int(kv_write_start)
    rotary_end = rotary_start + block_tokens
    full_cos, full_sin = full_rotary_emb
    if full_cos.shape[1] < rotary_end or full_sin.shape[1] < rotary_end:
        raise ValueError(
            f"full clip RoPE has {full_cos.shape[1]} tokens but block needs "
            f"[{rotary_start}, {rotary_end})"
        )
    rotary_emb = (
        full_cos[:, rotary_start:rotary_end],
        full_sin[:, rotary_start:rotary_end],
    )

    tokens = transformer.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
    temb, timestep_proj, encoder_hidden_states = transformer.condition_embedder(
        timestep.flatten(),
        encoder_hidden_states,
        timestep_seq_len=block_tokens,
    )
    timestep_proj = timestep_proj.unflatten(2, (6, -1))

    def incremental_context(layer_index: int) -> dict[str, Any]:
        return {
            "self_attention_mask": IncrementalKVAttentionContext(
                cache_layer=kv_cache[layer_index],
                write_start=rotary_start,
                commit=bool(commit_kv),
            )
        }

    tokens = run_transformer_blocks(
        transformer,
        tokens,
        encoder_hidden_states,
        timestep_proj,
        rotary_emb,
        block_kwargs=incremental_context,
    )
    tokens = project_output(
        tokens,
        temb,
        scale_shift_table=transformer.scale_shift_table,
        norm_out=transformer.norm_out,
        proj_out=transformer.proj_out,
    )
    return unpatchify(
        tokens,
        batch_size=batch_size,
        patch_grid=patch_grid,
        patch_size=(p_t, p_h, p_w),
    )


class WanTeacherForcingSelfAttentionProcessor:
    """Self-attention under a block-causal mask, routed through vforge ops.

    The mask arrives as a sparsity spec rather than a tensor: the ops layer
    picks FA4's sparse kernel when it can and materialises a dense mask for
    SDPA when it cannot, so this processor never has to know which.
    Ulysses is handled the same way as in the bidirectional processor -- the
    all-to-all restores all ``2S`` tokens on every rank, so the spec stays
    written against the global sequence.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: TeacherForcingSparsity | IncrementalKVAttentionContext | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise AssertionError("the teacher-forcing processor only serves self-attention")
        if not isinstance(
            attention_mask,
            (TeacherForcingSparsity, IncrementalKVAttentionContext),
        ):
            raise RuntimeError(
                "teacher-forcing self-attention requires an explicit sparsity spec "
                "or an incremental KV cache layer"
            )

        query, key, value = _get_qkv_projections(attn, hidden_states, None)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        if isinstance(attention_mask, IncrementalKVAttentionContext):
            output = attention_mask.cache_layer.update_and_attend(
                query,
                key,
                value,
                write_start=attention_mask.write_start,
                commit=attention_mask.commit,
                attention_fn=lambda q, k, v: vforge_attention(
                    q,
                    k,
                    v,
                    is_causal=False,
                ),
            )
        else:
            output = vforge_attention(
                query,
                key,
                value,
                ulysses=True,
                sparsity=attention_mask,
            )
        output = output.flatten(2, 3).type_as(query)
        output = attn.to_out[0](output)
        return attn.to_out[1](output)


class WanTeacherForcingTransformer3DModel(WanTransformer3DModel):
    """Wan 2.2 TI2V transformer trained by block-causal teacher forcing.

    ``frame_per_block`` and ``first_frame_own_block`` define the causal block
    layout and are kept off the Diffusers config on purpose: they describe the
    training scheme, not the weights, so a checkpoint stays interchangeable
    with the bidirectional model.
    """

    def __init__(
        self,
        *,
        frame_per_block: int = 3,
        first_frame_own_block: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.frame_per_block = int(frame_per_block)
        self.first_frame_own_block = bool(first_frame_own_block)
        for block in self.blocks:
            block.attn1.set_processor(WanTeacherForcingSelfAttentionProcessor())

    def _incremental_kv_executor(self) -> IncrementalKVExecutor:
        return IncrementalKVExecutor(
            self,
            block_forward=forward_incremental_kv_block,
        )

    def init_kv_cache(
        self,
        *,
        batch_size: int,
        num_cache_tokens: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> CausalKVCache:
        """Allocate one fixed-capacity absolute-slot cache per transformer layer."""
        return self._incremental_kv_executor().init_kv_cache(
            batch_size=batch_size,
            num_cache_tokens=num_cache_tokens,
            device=device,
            dtype=dtype,
        )

    def full_clip_rotary_emb(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build absolute RoPE once for the complete latent clip."""
        return self._incremental_kv_executor().full_clip_rotary_emb(hidden_states)

    def forward_kv_block(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        kv_cache: CausalKVCache,
        full_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        kv_write_start: int,
        commit_kv: bool,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """Run one block against committed history and overwrite its absolute KV slots."""
        return self._incremental_kv_executor().forward_kv_block(
            hidden_states,
            timestep,
            encoder_hidden_states,
            kv_cache=kv_cache,
            full_rotary_emb=full_rotary_emb,
            kv_write_start=kv_write_start,
            commit_kv=commit_kv,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        kv_cache: CausalKVCache | None = None,
        full_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_write_start: int = 0,
        commit_kv: bool = False,
    ) -> torch.Tensor:
        """``hidden_states`` is ``[B, C, 2T, h, w]``; ``timestep`` is ``[B, 2S]``."""
        if kv_cache is not None:
            if attention_mask is not None:
                raise ValueError("incremental_kv does not accept an attention mask")
            if full_rotary_emb is None:
                raise ValueError("incremental_kv requires full_rotary_emb")
            return self.forward_kv_block(
                hidden_states,
                timestep,
                encoder_hidden_states,
                kv_cache=kv_cache,
                full_rotary_emb=full_rotary_emb,
                kv_write_start=kv_write_start,
                commit_kv=commit_kv,
                attention_kwargs=attention_kwargs,
                return_dict=return_dict,
            )
        if attention_mask is not None:
            raise ValueError(
                "teacher forcing derives its own attention mask from the "
                "clean/noisy layout; an explicit mask would contradict it"
            )
        apply_lora_scale(self, attention_kwargs, logger=logger)

        batch_size, _, doubled_frames, height, width = hidden_states.shape
        if doubled_frames % 2:
            raise ValueError(
                "teacher forcing expects a doubled [clean | noisy] frame axis; "
                f"got an odd {doubled_frames}"
            )
        num_frames = doubled_frames // 2
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        tokens_per_frame = post_patch_height * post_patch_width
        tokens_per_half = post_patch_num_frames * tokens_per_frame

        # One half's RoPE, tiled: clean and noisy tokens of the same latent
        # frame must land on the same position, or the block-causal mask would
        # be comparing frames the positions disagree about.
        rope_cos, rope_sin = self.rope(hidden_states[:, :, :num_frames])
        rotary_emb = (
            torch.cat([rope_cos, rope_cos], dim=1),
            torch.cat([rope_sin, rope_sin], dim=1),
        )

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.ndim != 2:
            raise ValueError(
                "teacher forcing needs a per-token timestep [B, 2S] so the "
                f"clean half can be pinned to 0; got shape {tuple(timestep.shape)}"
            )
        ts_seq_len = timestep.shape[1]
        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
            timestep.flatten(),
            encoder_hidden_states,
            timestep_seq_len=ts_seq_len,
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        sparsity = TeacherForcingSparsity(
            num_latent_frames=post_patch_num_frames,
            tokens_per_frame=tokens_per_frame,
            frame_per_block=self.frame_per_block,
            first_frame_own_block=self.first_frame_own_block,
        )
        parallel_state = get_parallel_state()
        usp_size = int(parallel_state.config.usp_size)
        usp_group = parallel_state.usp_group
        if usp_size > 1:
            from vforge.distributed.ulysses import slice_sequence

            doubled_tokens = hidden_states.shape[1]
            if doubled_tokens % usp_size:
                raise ValueError(
                    f"teacher forcing needs the doubled sequence (2S="
                    f"{doubled_tokens} = 2 * {post_patch_num_frames} frames * "
                    f"{tokens_per_frame} tokens) to divide by usp_size="
                    f"{usp_size}; remainder {doubled_tokens % usp_size}. Padding "
                    "is refused here because the sparsity metadata is built for "
                    "one exact sequence length."
                )
            # Every rank runs the replicated pre-slice embedding, so the
            # gathered slice gradient is averaged to stop FSDP counting the
            # same full gradient once per rank.
            hidden_states = slice_sequence(
                hidden_states,
                dim=1,
                group=usp_group,
                scale_grad=True,
            )
            timestep_proj = slice_sequence(
                timestep_proj,
                dim=1,
                group=usp_group,
                scale_grad=True,
            )
            rotary_emb = (
                slice_sequence(rotary_emb[0], dim=1, group=usp_group),
                slice_sequence(rotary_emb[1], dim=1, group=usp_group),
            )
            # ``temb`` stays whole: no block reads it, and the output head that
            # does runs after the gather below.

        hidden_states = run_transformer_blocks(
            self,
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
            block_kwargs={"self_attention_mask": sparsity},
        )

        if usp_size > 1:
            from vforge.distributed.ulysses import gather_sequence

            # Rebuild 2S before splitting off the noisy half: shard boundaries
            # do not line up with the clean/noisy split (at usp_size=2 rank 0
            # holds the entire clean half), so slicing first takes the wrong
            # tokens.
            hidden_states = gather_sequence(hidden_states, dim=1, group=usp_group)

        hidden_states = hidden_states[:, tokens_per_half:]
        temb = temb[:, tokens_per_half:]

        hidden_states = project_output(
            hidden_states,
            temb,
            scale_shift_table=self.scale_shift_table,
            norm_out=self.norm_out,
            proj_out=self.proj_out,
        )
        return unpatchify(
            hidden_states,
            batch_size=batch_size,
            patch_grid=(
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
            ),
            patch_size=(p_t, p_h, p_w),
        )


__all__ = [
    "WanTeacherForcingSelfAttentionProcessor",
    "WanTeacherForcingTransformer3DModel",
    "forward_incremental_kv_block",
]
