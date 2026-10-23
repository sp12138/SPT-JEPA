"""Pure sequence contracts for LTX-2.3 Stage 1 causal teacher forcing.

This module owns the logical packed layout. It deliberately has no model,
distributed, or kernel dependencies so dense reference tests, sparse kernels,
and sequence-parallel execution can consume the same token semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vforge.ops._attention.segmented import (
    INVALID_SEGMENT,
    SegmentedSparsity,
)

REFERENCE = -1
CLEAN = 0
NOISY = 1


@dataclass(frozen=True)
class CausalPackedLayout:
    """Per-token metadata for ``[ref | all clean | noisy block ...]``."""

    block_ids: torch.Tensor
    token_kinds: torch.Tensor

    def __post_init__(self) -> None:
        if self.block_ids.ndim != 1 or self.token_kinds.ndim != 1:
            raise ValueError("block_ids and token_kinds must be one-dimensional")
        if self.block_ids.shape != self.token_kinds.shape:
            raise ValueError("block_ids and token_kinds must have the same shape")

    @property
    def logical_length(self) -> int:
        return int(self.block_ids.numel())

    @property
    def noisy_mask(self) -> torch.Tensor:
        return self.token_kinds == NOISY


@dataclass(frozen=True)
class PackedSequence:
    """Packed tensors plus their shared logical token metadata."""

    latents: torch.Tensor
    targets: torch.Tensor
    timesteps: torch.Tensor
    positions: torch.Tensor
    layout: CausalPackedLayout


def causal_block_ranges(
    num_frames: int,
    *,
    first_block_size: int = 4,
    block_size: int = 3,
) -> tuple[tuple[int, int], ...]:
    """Partition latent frames into one leading block and fixed later blocks."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if first_block_size < 1 or block_size < 1:
        raise ValueError("first_block_size and block_size must be >= 1")

    ranges: list[tuple[int, int]] = []
    end = min(num_frames, first_block_size)
    ranges.append((0, end))
    start = end
    while start < num_frames:
        end = min(num_frames, start + block_size)
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def time_aligned_audio_ranges(
    video_frame_ranges: tuple[tuple[int, int], ...],
    *,
    audio_length: int,
    video_fps: float,
    video_time_factor: int = 8,
    audio_latent_fps: int = 25,
) -> tuple[tuple[int, int], ...]:
    """Map causal-VAE video block boundaries onto audio latent tokens."""
    if not video_frame_ranges:
        raise ValueError("video_frame_ranges must be non-empty")
    if audio_length < 1 or video_fps <= 0:
        raise ValueError("audio_length and video_fps must be positive")

    def boundary(frame: int) -> int:
        if frame <= 0:
            return 0
        pixel_frame = 1 + (frame - 1) * video_time_factor
        return min(int(pixel_frame / video_fps * audio_latent_fps + 0.5), audio_length)

    ranges = [(boundary(start), boundary(end)) for start, end in video_frame_ranges]
    ranges[-1] = (ranges[-1][0], audio_length)
    return tuple(ranges)


def build_packed_layout(
    *,
    ref_length: int,
    block_lengths: tuple[int, ...],
    noisy_block_indices: tuple[int, ...] | None = None,
    device: torch.device | str | None = None,
) -> CausalPackedLayout:
    """Build metadata for the upstream Stage 1 packed sequence order."""
    if ref_length < 1:
        raise ValueError(f"ref_length must be >= 1, got {ref_length}")
    if not block_lengths or any(length < 0 for length in block_lengths) or not any(block_lengths):
        raise ValueError("block_lengths must contain non-negative lengths and at least one token")
    if noisy_block_indices is None:
        noisy_block_indices = tuple(range(len(block_lengths)))
    if any(
        index < 0 or index >= len(block_lengths) or block_lengths[index] == 0
        for index in noisy_block_indices
    ):
        raise ValueError("noisy_block_indices must select non-empty known blocks")

    clean_blocks = torch.repeat_interleave(
        torch.arange(len(block_lengths), device=device, dtype=torch.int64),
        torch.tensor(block_lengths, device=device, dtype=torch.int64),
    )
    reference_blocks = torch.full((ref_length,), REFERENCE, device=device, dtype=torch.int64)
    noisy_blocks = torch.cat(
        [
            torch.full(
                (block_lengths[index],),
                index,
                device=device,
                dtype=torch.int64,
            )
            for index in noisy_block_indices
        ]
    )
    block_ids = torch.cat((reference_blocks, clean_blocks, noisy_blocks))
    token_kinds = torch.cat(
        (
            torch.full((ref_length,), REFERENCE, device=device, dtype=torch.int64),
            torch.full_like(clean_blocks, CLEAN),
            torch.full_like(noisy_blocks, NOISY),
        )
    )
    return CausalPackedLayout(block_ids=block_ids, token_kinds=token_kinds)


def build_packed_sequence(
    *,
    clean: torch.Tensor,
    positions: torch.Tensor,
    token_ranges: tuple[tuple[int, int], ...],
    block_indices: tuple[int, ...] | None = None,
    sigmas: torch.Tensor,
    ref_latents: torch.Tensor,
    ref_positions: torch.Tensor,
    noise_blocks: tuple[torch.Tensor, ...] | None = None,
) -> PackedSequence:
    """Pack one modality using upstream's per-block noise and velocity target."""
    if clean.ndim != 3:
        raise ValueError(f"clean must be [B,T,C], got {tuple(clean.shape)}")
    if positions.ndim != 4 or positions.shape[0] != clean.shape[0]:
        raise ValueError("positions must be [B,D,T,2]")
    if positions.shape[2] != clean.shape[1] or positions.shape[-1] != 2:
        raise ValueError("positions sequence length must match clean")
    if ref_latents.ndim != 3 or ref_latents.shape[0] != clean.shape[0]:
        raise ValueError("ref_latents must be [B,R,C]")
    if (
        ref_positions.shape[:2] != positions.shape[:2]
        or ref_positions.shape[2] != ref_latents.shape[1]
    ):
        raise ValueError("ref_positions must match the reference and position dimensions")
    block_lengths = tuple(end - start for start, end in token_ranges)
    if any(start < 0 or end < start or end > clean.shape[1] for start, end in token_ranges):
        raise ValueError("token_ranges must be ordered slices within clean")
    if block_indices is None:
        block_indices = tuple(range(len(token_ranges)))
    if any(
        index < 0 or index >= len(token_ranges) or block_lengths[index] == 0
        for index in block_indices
    ):
        raise ValueError("block_indices must select non-empty known blocks")
    if sigmas.shape != (clean.shape[0], len(block_indices)):
        raise ValueError("sigmas must contain one value per sample and selected block")
    if noise_blocks is not None and len(noise_blocks) != len(block_indices):
        raise ValueError("noise_blocks must contain one tensor per selected block")

    noisy_latents: list[torch.Tensor] = []
    noisy_targets: list[torch.Tensor] = []
    noisy_positions: list[torch.Tensor] = []
    noisy_timesteps: list[torch.Tensor] = []
    for local_index, block_index in enumerate(block_indices):
        start, end = token_ranges[block_index]
        block_clean = clean[:, start:end]
        noise = (
            noise_blocks[local_index].to(device=clean.device, dtype=clean.dtype)
            if noise_blocks is not None
            else torch.randn_like(block_clean)
        )
        if noise.shape != block_clean.shape:
            raise ValueError("each noise block must match its clean block")
        sigma = sigmas[:, local_index].to(clean.dtype).view(-1, 1, 1)
        noisy_latents.append((1 - sigma) * block_clean + sigma * noise)
        noisy_targets.append(noise - block_clean)
        noisy_positions.append(positions[:, :, start:end])
        noisy_timesteps.append(sigmas[:, local_index].view(-1, 1).expand(-1, end - start))

    ref_length = ref_latents.shape[1]
    prefix_length = ref_length + clean.shape[1]
    latents = torch.cat((ref_latents, clean, *noisy_latents), dim=1)
    targets = torch.cat(
        (torch.zeros_like(ref_latents), torch.zeros_like(clean), *noisy_targets), dim=1
    )
    timesteps = torch.cat(
        (
            clean.new_zeros((clean.shape[0], prefix_length)),
            *noisy_timesteps,
        ),
        dim=1,
    )
    packed_positions = torch.cat((ref_positions, positions, *noisy_positions), dim=2)
    layout = build_packed_layout(
        ref_length=ref_length,
        block_lengths=block_lengths,
        noisy_block_indices=block_indices,
        device=clean.device,
    )
    return PackedSequence(
        latents=latents,
        targets=targets,
        timesteps=timesteps,
        positions=packed_positions,
        layout=layout,
    )


def _pad_segment_ids(segment_ids: tuple[int, ...], multiple: int) -> tuple[int, ...]:
    if multiple < 1:
        raise ValueError(f"multiple must be >= 1, got {multiple}")
    padding = (-len(segment_ids)) % multiple
    return segment_ids + (INVALID_SEGMENT,) * padding


def compile_self_segmented_sparsity(
    layout: CausalPackedLayout,
    *,
    multiple: int,
) -> SegmentedSparsity:
    """Compile LTX self visibility into the model-neutral segment graph."""
    block_ids = tuple(int(value) for value in layout.block_ids.tolist())
    token_kinds = tuple(int(value) for value in layout.token_kinds.tolist())
    num_blocks = max(block_ids) + 1
    reference_segment = 0
    clean_segment = {block: 1 + block for block in range(num_blocks)}
    noisy_segment = {block: 1 + num_blocks + block for block in range(num_blocks)}
    segment_ids = tuple(
        reference_segment
        if kind == REFERENCE
        else clean_segment[block]
        if kind == CLEAN
        else noisy_segment[block]
        for kind, block in zip(token_kinds, block_ids, strict=True)
    )
    clean_blocks = sorted(
        {block for kind, block in zip(token_kinds, block_ids, strict=True) if kind == CLEAN}
    )
    noisy_blocks = sorted(
        {block for kind, block in zip(token_kinds, block_ids, strict=True) if kind == NOISY}
    )
    allowed: list[tuple[int, int]] = [(reference_segment, reference_segment)]
    for query_block in clean_blocks:
        allowed.extend(
            (clean_segment[query_block], clean_segment[key_block])
            for key_block in clean_blocks
            if key_block <= query_block
        )
    for query_block in noisy_blocks:
        allowed.append((noisy_segment[query_block], reference_segment))
        allowed.extend(
            (noisy_segment[query_block], clean_segment[key_block])
            for key_block in clean_blocks
            if key_block < query_block
        )
        allowed.append((noisy_segment[query_block], noisy_segment[query_block]))
    physical_ids = _pad_segment_ids(segment_ids, multiple)
    return SegmentedSparsity(
        q_segment_ids=physical_ids,
        kv_segment_ids=physical_ids,
        allowed_pairs=tuple(allowed),
    )


def compile_cross_segmented_sparsity(
    *,
    query: CausalPackedLayout,
    key_value: CausalPackedLayout,
    multiple: int,
) -> SegmentedSparsity:
    """Compile full-shard A/V visibility while marking unused query rows discardable."""
    query_block_ids = tuple(int(value) for value in query.block_ids.tolist())
    key_block_ids = tuple(int(value) for value in key_value.block_ids.tolist())
    num_blocks = max((*query_block_ids, *key_block_ids)) + 1

    def segment_ids(layout: CausalPackedLayout) -> tuple[int, ...]:
        return tuple(
            0 if kind == REFERENCE else 1 + block if kind == CLEAN else 1 + num_blocks + block
            for kind, block in zip(
                layout.token_kinds.tolist(),
                layout.block_ids.tolist(),
                strict=True,
            )
        )

    query_segments = segment_ids(query)
    key_segments = segment_ids(key_value)
    query_noisy_blocks = set(query.block_ids[query.noisy_mask].tolist())
    key_noisy_blocks = set(key_value.block_ids[key_value.noisy_mask].tolist())
    common_blocks = sorted(query_noisy_blocks & key_noisy_blocks)
    discarded = tuple(
        dict.fromkeys(
            segment
            for segment, kind in zip(
                query_segments,
                query.token_kinds.tolist(),
                strict=True,
            )
            if kind != NOISY
        )
    )
    return SegmentedSparsity(
        q_segment_ids=_pad_segment_ids(query_segments, multiple),
        kv_segment_ids=_pad_segment_ids(key_segments, multiple),
        allowed_pairs=tuple(
            (1 + num_blocks + block, 1 + num_blocks + block) for block in common_blocks
        ),
        discarded_query_segments=discarded,
    )


def tail_pad_to_multiple(
    tensor: torch.Tensor,
    *,
    multiple: int,
    dim: int = 1,
    value: float | int = 0,
) -> tuple[torch.Tensor, int]:
    """Append deterministic padding without changing logical token order."""
    if multiple < 1:
        raise ValueError(f"multiple must be >= 1, got {multiple}")
    dim = dim % tensor.ndim
    real_length = int(tensor.shape[dim])
    padded_length = ((real_length + multiple - 1) // multiple) * multiple
    if padded_length == real_length:
        return tensor, real_length
    padding_shape = list(tensor.shape)
    padding_shape[dim] = padded_length - real_length
    padding = torch.full(
        padding_shape,
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat((tensor, padding), dim=dim), real_length


def unpad_sequence(tensor: torch.Tensor, real_length: int, *, dim: int = 1) -> torch.Tensor:
    """Discard an appended sequence tail before parity checks or loss."""
    dim = dim % tensor.ndim
    if not 0 <= real_length <= tensor.shape[dim]:
        raise ValueError(f"real_length must be in [0, {tensor.shape[dim]}], got {real_length}")
    index = [slice(None)] * tensor.ndim
    index[dim] = slice(0, real_length)
    return tensor[tuple(index)]


def noisy_only_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    noisy_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute FP32 MSE over noisy tokens only, excluding ref/clean/padding."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if noisy_mask.shape != prediction.shape[:-1]:
        raise ValueError(
            "noisy_mask must match prediction without its channel dimension, "
            f"got {tuple(noisy_mask.shape)} and {tuple(prediction.shape)}"
        )
    weights = noisy_mask.unsqueeze(-1).to(device=prediction.device, dtype=torch.float32)
    squared = (prediction.float() - target.float()).pow(2)
    denominator = weights.mean(dim=[-2, -1]).clamp(min=1e-8)
    if not bool(noisy_mask.any()):
        raise ValueError("noisy_mask must select at least one token")
    return ((squared * weights).mean(dim=[-2, -1]) / denominator).mean()


__all__ = [
    "CLEAN",
    "NOISY",
    "REFERENCE",
    "CausalPackedLayout",
    "PackedSequence",
    "build_packed_layout",
    "build_packed_sequence",
    "causal_block_ranges",
    "compile_cross_segmented_sparsity",
    "compile_self_segmented_sparsity",
    "noisy_only_mse",
    "tail_pad_to_multiple",
    "time_aligned_audio_ranges",
    "unpad_sequence",
]
