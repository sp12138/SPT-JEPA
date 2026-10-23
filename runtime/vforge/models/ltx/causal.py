"""LTX-2.3 Stage 1 causal model types.

The causal variant retains the stock module and parameter names so an LTX-2.3
checkpoint loads strictly. Training-only mask semantics are added in this
separate model seam; the Stage 0 :class:`LTXModel` remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from vforge.distributed.parallelize import FSDPPrecisionContract
from vforge.models.ltx.cached_attention import cached_attention, compute_pe_for_positions
from vforge.models.ltx.kv_cache import CacheMode, LayerKVCache, ModelKVCache
from vforge.models.ltx.modality import Modality
from vforge.models.ltx.model import LTXModel
from vforge.models.ltx.model_configurator import LTXModelConfigurator
from vforge.models.ltx.transformer import (
    BasicAVTransformerBlock,
    TransformerConfig,
    TransformerOpsConfig,
    apply_cross_attention_adaln,
)
from vforge.models.ltx.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from vforge.ops._attention.segmented import SegmentedSparsity


@dataclass(frozen=True)
class CausalMasks:
    """Noisy-region masks threaded explicitly through the causal forward."""

    video_self_sparsity: SegmentedSparsity | None = None
    audio_self_sparsity: SegmentedSparsity | None = None
    a2v_sparsity: SegmentedSparsity | None = None
    v2a_sparsity: SegmentedSparsity | None = None
    video_text_start: int | None = None
    audio_text_start: int | None = None
    video_text_end: int | None = None
    audio_text_end: int | None = None

    def __post_init__(self) -> None:
        paired = (self.video_text_start is None) == (self.audio_text_start is None)
        if not paired:
            raise ValueError("video_text_start and audio_text_start must be set together")
        end_paired = (self.video_text_end is None) == (self.audio_text_end is None)
        if not end_paired:
            raise ValueError("video_text_end and audio_text_end must be set together")
        sparse_cross_paired = (self.a2v_sparsity is None) == (self.v2a_sparsity is None)
        if not sparse_cross_paired:
            raise ValueError("a2v_sparsity and v2a_sparsity must be set together")
        for name, value in (
            ("video_text_start", self.video_text_start),
            ("audio_text_start", self.audio_text_start),
            ("video_text_end", self.video_text_end),
            ("audio_text_end", self.audio_text_end),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")


class CausalTransformerArgsPreprocessor(TransformerArgsPreprocessor):
    """Use a zero prompt sigma when prompt AdaLN is present, matching upstream TF."""

    def __init__(self, *args, prompt_zero_sigma: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prompt_zero_sigma = prompt_zero_sigma

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None = None,
    ) -> TransformerArgs:
        del cross_modality
        if self.prompt_zero_sigma and self.prompt_adaln is not None:
            modality = replace(modality, sigma=torch.zeros_like(modality.sigma))
        return super().prepare(modality)


class CausalMultiModalTransformerArgsPreprocessor(MultiModalTransformerArgsPreprocessor):
    """Use each modality's per-token timestep for A/V cross-attention gates."""

    def __init__(self, *args, prompt_zero_sigma: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        simple = self.simple_preprocessor
        self.simple_preprocessor = CausalTransformerArgsPreprocessor(
            patchify_proj=simple.patchify_proj,
            adaln=simple.adaln,
            inner_dim=simple.inner_dim,
            max_pos=simple.max_pos,
            num_attention_heads=simple.num_attention_heads,
            use_middle_indices_grid=simple.use_middle_indices_grid,
            timestep_scale_multiplier=simple.timestep_scale_multiplier,
            double_precision_rope=simple.double_precision_rope,
            positional_embedding_theta=simple.positional_embedding_theta,
            rope_type=simple.rope_type,
            caption_projection=simple.caption_projection,
            prompt_adaln=simple.prompt_adaln,
            prompt_zero_sigma=prompt_zero_sigma,
        )

    def _prepare_cross_attention_timestep(
        self,
        modality_timesteps: torch.Tensor,
        cross_modality_sigma: torch.Tensor,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cross_modality_sigma
        scale_shift_timestep, _ = self.cross_scale_shift_adaln(
            (modality_timesteps * timestep_scale_multiplier).flatten(),
            hidden_dtype=hidden_dtype,
        )
        scale_shift_timestep = scale_shift_timestep.view(
            batch_size,
            -1,
            scale_shift_timestep.shape[-1],
        )

        gate_timestep, _ = self.cross_gate_adaln(
            (modality_timesteps * self.av_ca_timestep_scale_multiplier).flatten(),
            hidden_dtype=hidden_dtype,
        )
        gate_timestep = gate_timestep.view(batch_size, -1, gate_timestep.shape[-1])
        return scale_shift_timestep, gate_timestep


class CausalAVTransformerBlock(BasicAVTransformerBlock):
    """Checkpoint-compatible block seam for Stage 1 training semantics."""

    def _apply_text_cross_attention(
        self,
        x_normed: torch.Tensor,
        context: torch.Tensor,
        attn,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        cross_attention_adaln: bool = False,
        noisy_start: int | None = None,
        noisy_end: int | None = None,
    ) -> torch.Tensor:
        """Run text cross-attention only on the packed noisy suffix."""
        sequence_length = x_normed.shape[1]
        start = 0 if noisy_start is None else noisy_start
        end = sequence_length if noisy_end is None else noisy_end
        if not 0 <= start <= end <= sequence_length:
            raise ValueError("local noisy text range must be within the sequence")
        empty_local_range = start == end
        if empty_local_range:
            if sequence_length == 0:
                return torch.zeros_like(x_normed)
            # USP peers must traverse the same parameterized submodules under
            # FSDP even when this rank owns no noisy query tokens. Execute one
            # local token as a zero-gradient sentinel, then discard its value.
            # This keeps collective/autograd hook order aligned without making
            # ref, clean, or padding tokens contribute to the residual.
            execution_start = min(start, sequence_length - 1)
            execution_end = execution_start + 1
        else:
            execution_start = start
            execution_end = end
        sliced = start != 0 or end != sequence_length
        if not sliced:
            return super()._apply_text_cross_attention(
                x_normed,
                context,
                attn,
                scale_shift_table,
                prompt_scale_shift_table,
                timestep,
                prompt_timestep,
                context_mask,
                cross_attention_adaln,
            )

        query = x_normed[:, execution_start:execution_end]
        query_timestep = (
            timestep[:, execution_start:execution_end]
            if timestep is not None and timestep.shape[1] == sequence_length
            else timestep
        )
        if cross_attention_adaln:
            shift, scale, gate = self.get_ada_values(
                scale_shift_table,
                query.shape[0],
                query_timestep,
                slice(6, 9),
            )
            output = apply_cross_attention_adaln(
                query,
                context,
                attn,
                shift,
                scale,
                gate,
                prompt_scale_shift_table,
                prompt_timestep,
                context_mask,
            )
        else:
            output = attn(query, context=context, mask=context_mask)
        if empty_local_range:
            return torch.zeros_like(x_normed) + output.sum() * 0.0
        prefix = output.new_zeros(output.shape[0], start, output.shape[2])
        suffix = output.new_zeros(output.shape[0], sequence_length - end, output.shape[2])
        return torch.cat((prefix, output, suffix), dim=1)

    @staticmethod
    def _slice_tokens(
        tensor: torch.Tensor | None,
        start: int,
        full_length: int,
    ) -> torch.Tensor | None:
        if tensor is None or start == 0 or tensor.ndim < 2 or tensor.shape[1] != full_length:
            return tensor
        return tensor[:, start:]

    @staticmethod
    def _slice_cross_pe(
        pe: tuple[torch.Tensor, torch.Tensor] | None,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if pe is None or start == 0:
            return pe
        cosine, sine = pe
        return cosine[:, :, start:], sine[:, :, start:]

    @staticmethod
    def _pad_prefix(output: torch.Tensor, start: int) -> torch.Tensor:
        if start == 0:
            return output
        prefix = output.new_zeros(output.shape[0], start, output.shape[2])
        return torch.cat((prefix, output), dim=1)

    @staticmethod
    def _keep_range(output: torch.Tensor, start: int, end: int) -> torch.Tensor:
        if start == 0 and end == output.shape[1]:
            return output
        prefix = output.new_zeros(output.shape[0], start, output.shape[2])
        suffix = output.new_zeros(output.shape[0], output.shape[1] - end, output.shape[2])
        return torch.cat((prefix, output[:, start:end], suffix), dim=1)

    def forward(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        masks: CausalMasks | None = None,
        *,
        cache: LayerKVCache | None = None,
        cache_mode: CacheMode | None = None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        """Dispatch packed TF or an explicit cached block forward."""
        if cache is not None:
            if masks is not None or cache_mode is None:
                raise ValueError("cached forward requires a mode and no packed masks")
            return self.forward_cached(video, audio, cache, cache_mode)
        if masks is None:
            return super().forward(video, audio)
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None
        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0
        video_start = masks.video_text_start or 0
        audio_start = masks.audio_text_start or 0
        video_end = masks.video_text_end if masks.video_text_end is not None else vx.shape[1]
        audio_end = masks.audio_text_end if masks.audio_text_end is not None else ax.shape[1]
        if run_vx and not 0 <= video_start <= video_end <= vx.shape[1]:
            raise ValueError("video text range exceeds the video sequence length")
        if run_ax and not 0 <= audio_start <= audio_end <= ax.shape[1]:
            raise ValueError("audio text range exceeds the audio sequence length")

        run_a2v = run_vx and run_ax
        run_v2a = run_ax and run_vx

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            norm_vx = self.ada_zero_function(vx, self.norm_eps, vscale_msa, vshift_msa)
            vx_msa_out = self.attn1(
                norm_vx,
                pe=video.positional_embeddings,
                mask=(None if masks.video_self_sparsity is not None else video.self_attention_mask),
                perturbation_mask=video.self_attn_perturbation_mask,
                all_perturbed=video.self_attn_all_perturbed,
                sparsity=masks.video_self_sparsity,
            )
            vx, vx_normed = self.post_sa_function(vx, vx_msa_out, None, self.norm_eps, vgate_msa)
            vx = vx + self._apply_text_cross_attention(
                vx_normed,
                video.context,
                self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps,
                video.prompt_timestep,
                video.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
                noisy_start=masks.video_text_start,
                noisy_end=masks.video_text_end,
            )

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )
            norm_ax = self.ada_zero_function(ax, self.norm_eps, ascale_msa, ashift_msa)
            ax_msa_out = self.audio_attn1(
                norm_ax,
                pe=audio.positional_embeddings,
                mask=(None if masks.audio_self_sparsity is not None else audio.self_attention_mask),
                perturbation_mask=audio.self_attn_perturbation_mask,
                all_perturbed=audio.self_attn_all_perturbed,
                sparsity=masks.audio_self_sparsity,
            )
            ax, ax_normed = self.post_sa_function(ax, ax_msa_out, None, self.norm_eps, agate_msa)
            ax = ax + self._apply_text_cross_attention(
                ax_normed,
                audio.context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps,
                audio.prompt_timestep,
                audio.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
                noisy_start=masks.audio_text_start,
                noisy_end=masks.audio_text_end,
            )

        if run_a2v or run_v2a:
            video_length = vx.shape[1]
            audio_length = ax.shape[1]
            vx_pre_av = vx
            ax_pre_av = ax
            video_source_start = 0
            audio_source_start = 0

            if run_a2v and not video.cross_attn_skip_all:
                video_scale, video_shift, video_gate = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    self._slice_tokens(
                        video.cross_scale_shift_timestep, video_source_start, video_length
                    ),
                    self._slice_tokens(video.cross_gate_timestep, video_source_start, video_length),
                    slice(0, 2),
                )
                audio_scale, audio_shift, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    self._slice_tokens(
                        audio.cross_scale_shift_timestep, audio_source_start, audio_length
                    ),
                    self._slice_tokens(audio.cross_gate_timestep, audio_source_start, audio_length),
                    slice(0, 2),
                )
                video_query = self.ada_zero_function(
                    vx_pre_av,
                    self.norm_eps,
                    video_scale,
                    video_shift,
                )
                audio_key_value = self.ada_zero_function(
                    ax_pre_av,
                    self.norm_eps,
                    audio_scale,
                    audio_shift,
                )
                a2v_output = self.audio_to_video_attn(
                    video_query,
                    context=audio_key_value,
                    pe=self._slice_cross_pe(video.cross_positional_embeddings, video_source_start),
                    k_pe=self._slice_cross_pe(
                        audio.cross_positional_embeddings, audio_source_start
                    ),
                    sparsity=masks.a2v_sparsity,
                )
                perturbation = self._slice_tokens(
                    video.cross_attn_perturbation_mask,
                    video_source_start,
                    video_length,
                )
                a2v_residual = a2v_output * video_gate * perturbation
                vx = vx + (self._keep_range(a2v_residual, video_start, video_end))

            if run_v2a and not audio.cross_attn_skip_all:
                audio_scale, audio_shift, audio_gate = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    self._slice_tokens(
                        audio.cross_scale_shift_timestep, audio_source_start, audio_length
                    ),
                    self._slice_tokens(audio.cross_gate_timestep, audio_source_start, audio_length),
                    slice(2, 4),
                )
                video_scale, video_shift, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    self._slice_tokens(
                        video.cross_scale_shift_timestep, video_source_start, video_length
                    ),
                    self._slice_tokens(video.cross_gate_timestep, video_source_start, video_length),
                    slice(2, 4),
                )
                audio_query = self.ada_zero_function(
                    ax_pre_av,
                    self.norm_eps,
                    audio_scale,
                    audio_shift,
                )
                video_key_value = self.ada_zero_function(
                    vx_pre_av,
                    self.norm_eps,
                    video_scale,
                    video_shift,
                )
                v2a_output = self.video_to_audio_attn(
                    audio_query,
                    context=video_key_value,
                    pe=self._slice_cross_pe(audio.cross_positional_embeddings, audio_source_start),
                    k_pe=self._slice_cross_pe(
                        video.cross_positional_embeddings, video_source_start
                    ),
                    sparsity=masks.v2a_sparsity,
                )
                perturbation = self._slice_tokens(
                    audio.cross_attn_perturbation_mask,
                    audio_source_start,
                    audio_length,
                )
                v2a_residual = v2a_output * audio_gate * perturbation
                ax = ax + (self._keep_range(v2a_residual, audio_start, audio_end))

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
            )
            vx_scaled = self.ada_zero_function(vx, self.norm_eps, vscale_mlp, vshift_mlp)
            vx = vx + self.ff(vx_scaled) * vgate_mlp
        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
            )
            ax_scaled = self.ada_zero_function(ax, self.norm_eps, ascale_mlp, ashift_mlp)
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

        return replace(video, x=vx) if video is not None else None, replace(
            audio, x=ax
        ) if audio is not None else None

    def forward_cached(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        cache: LayerKVCache,
        mode: CacheMode,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        """KV-cached inference forward for ONE block.

        Uses :func:`cached_attention` for the blockwise-causal self-attentions
        (read/write this layer's cache). Text and A2V/V2A cross attention run
        ONLY in ``DENOISE`` mode (the whole current block is noisy); ref/src/clean
        prefills get neither -- matching the text-on-noisy / noisy-only-cross
        training. Causality is enforced by the cache contents, so no attention
        masks here. STG perturbations (attached to the args by the model-level
        ``block_input_processor``) are honored in DENOISE mode: self-attention
        skip/blend mirrors stock, cross-attention skip goes through
        ``cross_attn_skip_all`` / ``cross_attn_perturbation_mask``.
        Cross-modal attention is in-block (the block IS the noisy unit), so it
        needs no mask and no cross KV-cache.
        """
        lc = cache
        apply_text = mode == CacheMode.DENOISE

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = apply_text and run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = apply_text and run_ax and (video is not None and vx.numel() > 0)

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            norm_vx = self.ada_zero_function(vx, self.norm_eps, vscale_msa, vshift_msa)
            del vshift_msa, vscale_msa

            vx_msa_out = cached_attention(
                self.attn1,
                norm_vx,
                None,
                pe=video.positional_embeddings,
                k_pe=None,
                cache=lc.video_self,
                mode=mode,
                cond_pe=cache.video_cond_pe,
                perturbation_mask=video.self_attn_perturbation_mask,
                all_perturbed=video.self_attn_all_perturbed,
                sequence_length=video.global_sequence_length,
            )
            vx, vx_normed = self.post_sa_function(vx, vx_msa_out, None, self.norm_eps, vgate_msa)
            del vgate_msa, norm_vx, vx_msa_out
            if apply_text:
                vx = vx + self._apply_text_cross_attention(
                    vx_normed,
                    video.context,
                    self.attn2,
                    self.scale_shift_table,
                    getattr(self, "prompt_scale_shift_table", None),
                    video.timesteps,
                    video.prompt_timestep,
                    video.context_mask,
                    cross_attention_adaln=self.cross_attention_adaln,
                    noisy_start=None,
                )
            del vx_normed

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )
            norm_ax = self.ada_zero_function(ax, self.norm_eps, ascale_msa, ashift_msa)
            del ashift_msa, ascale_msa

            ax_msa_out = cached_attention(
                self.audio_attn1,
                norm_ax,
                None,
                pe=audio.positional_embeddings,
                k_pe=None,
                cache=lc.audio_self,
                mode=mode,
                cond_pe=cache.audio_cond_pe,
                perturbation_mask=audio.self_attn_perturbation_mask,
                all_perturbed=audio.self_attn_all_perturbed,
                sequence_length=audio.global_sequence_length,
            )
            ax, ax_normed = self.post_sa_function(ax, ax_msa_out, None, self.norm_eps, agate_msa)
            del agate_msa, norm_ax, ax_msa_out
            if apply_text:
                ax = ax + self._apply_text_cross_attention(
                    ax_normed,
                    audio.context,
                    self.audio_attn2,
                    self.audio_scale_shift_table,
                    getattr(self, "audio_prompt_scale_shift_table", None),
                    audio.timesteps,
                    audio.prompt_timestep,
                    audio.context_mask,
                    cross_attention_adaln=self.cross_attention_adaln,
                    noisy_start=None,
                )
            del ax_normed

        # Audio - Video cross attention -- DENOISE only, in-block, no mask
        # (block-diagonal degenerates to a single full block-pair).
        if run_a2v or run_v2a:
            vx_pre_av = vx
            ax_pre_av = ax
            if run_a2v and not video.cross_attn_skip_all:
                scale_ca_video_a2v, shift_ca_video_a2v, gate_out_a2v = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(0, 2),
                )
                a2v_vx_scaled = self.ada_zero_function(
                    vx_pre_av, self.norm_eps, scale_ca_video_a2v, shift_ca_video_a2v
                )
                del scale_ca_video_a2v, shift_ca_video_a2v

                scale_ca_audio_a2v, shift_ca_audio_a2v, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(0, 2),
                )
                a2v_ax_scaled = self.ada_zero_function(
                    ax_pre_av, self.norm_eps, scale_ca_audio_a2v, shift_ca_audio_a2v
                )
                del scale_ca_audio_a2v, shift_ca_audio_a2v
                a2v_out = self.audio_to_video_attn(
                    a2v_vx_scaled,
                    context=a2v_ax_scaled,
                    pe=video.cross_positional_embeddings,
                    k_pe=audio.cross_positional_embeddings,
                    query_length=video.global_sequence_length,
                    context_length=audio.global_sequence_length,
                )
                a2v_out = a2v_out * gate_out_a2v
                if video.cross_attn_perturbation_mask is not None:
                    a2v_out = a2v_out * video.cross_attn_perturbation_mask
                vx = vx + a2v_out
                del gate_out_a2v, a2v_vx_scaled, a2v_ax_scaled, a2v_out

            if run_v2a and not audio.cross_attn_skip_all:
                scale_ca_audio_v2a, shift_ca_audio_v2a, gate_out_v2a = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(2, 4),
                )
                v2a_ax_scaled = self.ada_zero_function(
                    ax_pre_av, self.norm_eps, scale_ca_audio_v2a, shift_ca_audio_v2a
                )
                del scale_ca_audio_v2a, shift_ca_audio_v2a
                scale_ca_video_v2a, shift_ca_video_v2a, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(2, 4),
                )
                v2a_vx_scaled = self.ada_zero_function(
                    vx_pre_av, self.norm_eps, scale_ca_video_v2a, shift_ca_video_v2a
                )
                del scale_ca_video_v2a, shift_ca_video_v2a
                v2a_out = self.video_to_audio_attn(
                    v2a_ax_scaled,
                    context=v2a_vx_scaled,
                    pe=audio.cross_positional_embeddings,
                    k_pe=video.cross_positional_embeddings,
                    query_length=audio.global_sequence_length,
                    context_length=video.global_sequence_length,
                )
                v2a_out = v2a_out * gate_out_v2a
                if audio.cross_attn_perturbation_mask is not None:
                    v2a_out = v2a_out * audio.cross_attn_perturbation_mask
                ax = ax + v2a_out
                del gate_out_v2a, v2a_ax_scaled, v2a_vx_scaled, v2a_out
            del vx_pre_av, ax_pre_av

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
            )
            vx_scaled = self.ada_zero_function(vx, self.norm_eps, vscale_mlp, vshift_mlp)
            vx = vx + self.ff(vx_scaled) * vgate_mlp
            del vshift_mlp, vscale_mlp, vgate_mlp, vx_scaled

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
            )
            ax_scaled = self.ada_zero_function(ax, self.norm_eps, ascale_mlp, ashift_mlp)
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp
            del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled

        return replace(video, x=vx) if video is not None else None, replace(
            audio, x=ax
        ) if audio is not None else None


class CausalLTXModel(LTXModel):
    """LTX model whose blocks accept the Stage 1 causal mask bundle."""

    _no_split_modules = ["CausalAVTransformerBlock"]

    def __init__(self, *, prompt_zero_sigma: bool = True, **kwargs) -> None:
        self.prompt_zero_sigma = prompt_zero_sigma
        super().__init__(**kwargs)

    def get_fsdp_precision_contract(self) -> FSDPPrecisionContract:
        """Keep Stage 1 positions in FP32 at the FSDP execution boundary."""
        return replace(
            super().get_fsdp_precision_contract(),
            preserve_forward_input_paths=(".", "transformer_blocks.*"),
        )

    def _init_preprocessors(self, cross_pe_max_pos: int | None = None) -> None:
        if not (self.model_type.is_video_enabled() and self.model_type.is_audio_enabled()):
            super()._init_preprocessors(cross_pe_max_pos)
            return
        super()._init_preprocessors(cross_pe_max_pos)
        self._stock_video_args_preprocessor = self.video_args_preprocessor
        self._stock_audio_args_preprocessor = self.audio_args_preprocessor
        self.video_args_preprocessor = CausalMultiModalTransformerArgsPreprocessor(
            patchify_proj=self.patchify_proj,
            adaln=self.adaln_single,
            cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
            cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
            inner_dim=self.inner_dim,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            cross_pe_max_pos=cross_pe_max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            audio_cross_attention_dim=self.audio_cross_attention_dim,
            timestep_scale_multiplier=self.timestep_scale_multiplier,
            double_precision_rope=self.double_precision_rope,
            positional_embedding_theta=self.positional_embedding_theta,
            rope_type=self.rope_type,
            av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
            caption_projection=getattr(self, "caption_projection", None),
            prompt_adaln=getattr(self, "prompt_adaln_single", None),
            prompt_zero_sigma=self.prompt_zero_sigma,
        )
        self.audio_args_preprocessor = CausalMultiModalTransformerArgsPreprocessor(
            patchify_proj=self.audio_patchify_proj,
            adaln=self.audio_adaln_single,
            cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
            cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
            inner_dim=self.audio_inner_dim,
            max_pos=self.audio_positional_embedding_max_pos,
            num_attention_heads=self.audio_num_attention_heads,
            cross_pe_max_pos=cross_pe_max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            audio_cross_attention_dim=self.audio_cross_attention_dim,
            timestep_scale_multiplier=self.timestep_scale_multiplier,
            double_precision_rope=self.double_precision_rope,
            positional_embedding_theta=self.positional_embedding_theta,
            rope_type=self.rope_type,
            av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
            caption_projection=getattr(self, "audio_caption_projection", None),
            prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            prompt_zero_sigma=self.prompt_zero_sigma,
        )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        ops: TransformerOpsConfig,
        apply_gated_attention: bool,
    ) -> None:
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                CausalAVTransformerBlock(
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    ops=ops,
                )
                for _ in range(num_layers)
            ]
        )

    def _process_causal_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        masks: CausalMasks,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        for block in self.transformer_blocks:
            if video is not None:
                video = self.block_input_processor(video)
            if audio is not None:
                audio = self.block_input_processor(audio)
            if self._enable_gradient_checkpointing and self.training:
                video, audio = torch.utils.checkpoint.checkpoint(
                    block,
                    video,
                    audio,
                    masks,
                    use_reentrant=False,
                )
            else:
                video, audio = block(video=video, audio=audio, masks=masks)
        return video, audio

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        masks: CausalMasks | None = None,
        *,
        cache: ModelKVCache | None = None,
        cache_mode: CacheMode | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run stock, packed TF, or snapshot-safe cached autoregression."""
        if cache is not None:
            if masks is not None or cache_mode is None:
                raise ValueError("cached forward requires a mode and no packed masks")
            return self.forward_cached(video, audio, cache, cache_mode)
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")
        if masks is None:
            video_args = (
                self._stock_video_args_preprocessor.prepare(video, audio)
                if video is not None
                else None
            )
            audio_args = (
                self._stock_audio_args_preprocessor.prepare(audio, video)
                if audio is not None
                else None
            )
            video_out, audio_out = self._process_transformer_blocks(
                video=video_args,
                audio=audio_args,
            )
            video_prediction = (
                self._process_output(
                    self.scale_shift_table,
                    self.norm_out,
                    self.proj_out,
                    video_out.x,
                    video_out.embedded_timestep,
                )
                if video_out is not None
                else None
            )
            audio_prediction = (
                self._process_output(
                    self.audio_scale_shift_table,
                    self.audio_norm_out,
                    self.audio_proj_out,
                    audio_out.x,
                    audio_out.embedded_timestep,
                )
                if audio_out is not None
                else None
            )
            return video_prediction, audio_prediction
        video_args = (
            self.video_args_preprocessor.prepare(video, audio) if video is not None else None
        )
        audio_args = (
            self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None
        )
        video_out, audio_out = self._process_causal_transformer_blocks(
            video_args,
            audio_args,
            masks,
        )
        video_prediction = (
            self._process_output(
                self.scale_shift_table,
                self.norm_out,
                self.proj_out,
                video_out.x,
                video_out.embedded_timestep,
            )
            if video_out is not None
            else None
        )
        audio_prediction = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return video_prediction, audio_prediction

    def compute_block_cond_pe(
        self,
        ref_video_positions: torch.Tensor | None,
        ref_audio_positions: torch.Tensor | None,
        block_start_seconds: float,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, tuple[torch.Tensor, torch.Tensor] | None]:
        """Compute the per-block ref RoPE PE, shifted by ``block_start_seconds``.

        This is the block-start RoPE rolling: the ref image/audio is kept at a
        constant relative distance from the generation frontier by advancing its
        time coordinate. ``block_start_seconds=0`` keeps the ref at its FIXED
        training position. Returns ``(video_pe, audio_pe)`` to feed
        ``cache.set_cond_pe``. (Cross-modal attention is noisy-only / in-block,
        so it needs no ref cross-PE.)
        """
        v_pe = a_pe = None
        if ref_video_positions is not None:
            v_pos = ref_video_positions.clone()
            v_pos[:, 0, ...] += block_start_seconds
            v_pe = compute_pe_for_positions(self, v_pos, "video")
        if ref_audio_positions is not None:
            a_pos = ref_audio_positions.clone() + block_start_seconds
            a_pe = compute_pe_for_positions(self, a_pos, "audio")
        return v_pe, a_pe

    def forward_cached(self, video, audio, cache: ModelKVCache, mode: CacheMode):
        if len(cache.layers) != len(self.transformer_blocks):
            raise ValueError("KV cache layer count does not match the model")
        video_args = (
            self.video_args_preprocessor.prepare(video, audio) if video is not None else None
        )
        audio_args = (
            self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None
        )
        for block, live_cache in zip(self.transformer_blocks, cache.layers, strict=True):
            # Capture immutable history and reference PE BEFORE closing over it.
            layer_cache = live_cache.snapshot() if mode == CacheMode.DENOISE else live_cache
            if self._enable_gradient_checkpointing and self.training and torch.is_grad_enabled():
                if mode != CacheMode.DENOISE:
                    raise RuntimeError("cache prefill must run without gradients")
                video_args, audio_args = torch.utils.checkpoint.checkpoint(
                    block,
                    video_args,
                    audio_args,
                    cache=layer_cache,
                    cache_mode=mode,
                    use_reentrant=False,
                )
            else:
                video_args, audio_args = block(
                    video_args,
                    audio_args,
                    cache=layer_cache,
                    cache_mode=mode,
                )
        vx = (
            self._process_output(
                self.scale_shift_table,
                self.norm_out,
                self.proj_out,
                video_args.x,
                video_args.embedded_timestep,
            )
            if video_args is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_args.x,
                audio_args.embedded_timestep,
            )
            if audio_args is not None
            else None
        )
        return vx, ax


class CausalLTXModelConfigurator(LTXModelConfigurator):
    """Build :class:`CausalLTXModel` from the stock checkpoint metadata."""

    model_class = CausalLTXModel


__all__ = [
    "CausalAVTransformerBlock",
    "CausalLTXModel",
    "CausalLTXModelConfigurator",
    "CausalMasks",
    "CausalMultiModalTransformerArgsPreprocessor",
    "CausalTransformerArgsPreprocessor",
]
