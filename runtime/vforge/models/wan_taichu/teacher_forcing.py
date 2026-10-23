"""Block-causal teacher-forcing support owned by Wan Taichu.

This module owns no weights. It installs a Wan Taichu self-attention processor
and provides the doubled ``[clean | noisy]`` forward used by the package-local
Transformer implementation.

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
from vforge.distributed.ulysses import head_to_sequence, sequence_to_head
from vforge.models.causal import IncrementalKVAttentionContext
from vforge.models.wan_taichu.attention import WanAttention
from vforge.models.wan_taichu.attention import (
    apply_rotary_emb as _apply_rotary_emb,
)
from vforge.models.wan_taichu.attention import (
    get_qkv_projections as _get_qkv_projections,
)
from vforge.models.wan_taichu.attention import (
    narrow_qk_to_value as _narrow_qk_to_value,
)
from vforge.models.wan_taichu.transformer import (
    apply_lora_scale,
    project_output,
    run_transformer_blocks,
    slice_ulysses_sequence_locally,
    unpatchify,
)
from vforge.ops import TeacherForcingSparsity
from vforge.ops import attention as vforge_attention

logger = logging.get_logger(__name__)


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
                "or incremental KV context"
            )

        query, key, value = _get_qkv_projections(attn, hidden_states, None)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        if isinstance(attention_mask, IncrementalKVAttentionContext):
            if attention_mask.usp_size > 1:
                query = sequence_to_head(
                    query,
                    attention_mask.usp_group,
                    attention_mask.usp_size,
                )
                key = sequence_to_head(
                    key,
                    attention_mask.usp_group,
                    attention_mask.usp_size,
                )
                value = sequence_to_head(
                    value,
                    attention_mask.usp_group,
                    attention_mask.usp_size,
                )
            # Narrowing here, rather than before the all-to-all above, keeps the
            # cache's dtype check able to reject a real compute-dtype change
            # mid-rollout.
            cache_query, cache_key = _narrow_qk_to_value(query, key, value)
            output = attention_mask.resolve_cache_layer().update_and_attend(
                cache_query,
                cache_key,
                value,
                write_start=attention_mask.write_start,
                history_tokens=attention_mask.history_tokens,
                commit=attention_mask.commit,
                frame_tokens=attention_mask.frame_tokens,
                local_window_frames=attention_mask.local_window_frames,
                sink_frames=attention_mask.sink_frames,
                attention_fn=lambda q, k, v: vforge_attention(
                    q,
                    k,
                    v,
                    is_causal=False,
                ),
            )
            if attention_mask.usp_size > 1:
                output = head_to_sequence(
                    output,
                    attention_mask.usp_group,
                    attention_mask.usp_size,
                    original_heads=attention_mask.original_heads,
                )
        else:
            attend_query, attend_key = _narrow_qk_to_value(query, key, value)
            output = vforge_attention(
                attend_query,
                attend_key,
                value,
                ulysses=True,
                sparsity=attention_mask,
            )
        output = output.flatten(2, 3).type_as(query)
        output = attn.to_out[0](output)
        return attn.to_out[1](output)


class WanTeacherForcingForwardMixin:
    """Reusable block-causal teacher-forcing forward for Wan transformers.

    ``frame_per_block`` and ``first_frame_own_block`` define the causal block
    layout and are kept off the Diffusers config on purpose: they describe the
    training scheme, not the weights, so a checkpoint stays interchangeable
    with the bidirectional model.
    """

    def _enable_teacher_forcing(
        self,
        *,
        frame_per_block: int = 3,
        first_frame_own_block: bool = True,
        local_attn_size: int = -1,
        sink_size: int = 0,
    ) -> None:
        self.frame_per_block = int(frame_per_block)
        self.first_frame_own_block = bool(first_frame_own_block)
        self.local_attn_size = int(local_attn_size)
        self.sink_size = int(sink_size)
        # Validate the complete geometry once during construction. The real
        # sequence length is supplied by the forward below.
        TeacherForcingSparsity(
            num_latent_frames=max(self.frame_per_block, 1),
            tokens_per_frame=1,
            frame_per_block=self.frame_per_block,
            first_frame_own_block=self.first_frame_own_block,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
        )
        for block in self.blocks:
            block.attn1.set_processor(WanTeacherForcingSelfAttentionProcessor())

    def forward_teacher_forcing(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """``hidden_states`` is ``[B, C, 2T, h, w]``; ``timestep`` is ``[B, 2S]``."""
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
            torch.cat([rope_cos, rope_cos], dim=1).float(),
            torch.cat([rope_sin, rope_sin], dim=1).float(),
        )
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.ndim != 2:
            raise ValueError(
                "teacher forcing needs a per-token timestep [B, 2S] so the "
                f"clean half can be pinned to 0; got shape {tuple(timestep.shape)}"
            )
        doubled_post_patch_frames = 2 * post_patch_num_frames
        if timestep.shape[1] != doubled_post_patch_frames * tokens_per_frame:
            raise ValueError(
                f"per-token timestep has {timestep.shape[1]} tokens but the "
                f"doubled sequence is {doubled_post_patch_frames} frames * "
                f"{tokens_per_frame} tokens"
            )
        # The timestep is built by repeating one value per latent frame, so
        # embedding it per token projects the same ``6 * dim`` row
        # ``tokens_per_frame`` times.  At 2S=1M that intermediate is 59 GiB and
        # is what OOMs the A14B run; collapse it to one row per frame and
        # expand only where a token-shaped tensor is genuinely consumed.
        frame_timestep = timestep.unflatten(1, (doubled_post_patch_frames, tokens_per_frame))
        if not torch.equal(frame_timestep, frame_timestep[:, :, :1].expand_as(frame_timestep)):
            raise ValueError(
                "teacher forcing needs a timestep that is constant within each "
                "latent frame; got a value that varies across a frame's tokens"
            )
        frame_timestep = frame_timestep[:, :, 0]
        condition = self.condition_embedder(
            frame_timestep.flatten(),
            encoder_hidden_states,
            timestep_seq_len=doubled_post_patch_frames,
        )
        temb, timestep_proj, encoder_hidden_states, *image_conditions = condition
        if any(value is not None for value in image_conditions):
            raise ValueError("teacher forcing does not accept CLIP or DINO image conditions")
        # Both stay per-frame ([B, 2F, ...]) until expanded below.
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        sparsity = TeacherForcingSparsity(
            num_latent_frames=post_patch_num_frames,
            tokens_per_frame=tokens_per_frame,
            frame_per_block=self.frame_per_block,
            first_frame_own_block=self.first_frame_own_block,
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
        )
        parallel_state = get_parallel_state()
        usp_size = int(parallel_state.config.usp_size)
        usp_group = parallel_state.usp_group
        if usp_size > 1:
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
            hidden_states = slice_ulysses_sequence_locally(
                hidden_states,
                dim=1,
                group=usp_group,
            )
            # Expand the per-frame modulation straight onto this rank's token
            # shard. Materialising all 2S rows first and then slicing throws
            # away usp_size-1 of every usp_size rows at peak memory.
            frame_index = torch.arange(
                doubled_post_patch_frames,
                device=timestep_proj.device,
            ).repeat_interleave(tokens_per_frame)
            local_frame_index = slice_ulysses_sequence_locally(
                frame_index.unsqueeze(0),
                dim=1,
                group=usp_group,
            )[0]
            timestep_proj = timestep_proj.index_select(1, local_frame_index)
            rotary_emb = (
                slice_ulysses_sequence_locally(rotary_emb[0], dim=1, group=usp_group),
                slice_ulysses_sequence_locally(rotary_emb[1], dim=1, group=usp_group),
            )
            # ``temb`` stays per-frame: no block reads it, and the output head
            # that does runs after the gather below.
        else:
            timestep_proj = timestep_proj.repeat_interleave(tokens_per_frame, dim=1)

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
        # Only the noisy half reaches the output head, so expand those frames
        # alone instead of carrying a token-shaped ``temb`` through the blocks.
        temb = temb[:, post_patch_num_frames:].repeat_interleave(tokens_per_frame, dim=1)

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
    "WanTeacherForcingForwardMixin",
    "WanTeacherForcingSelfAttentionProcessor",
]
