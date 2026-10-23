"""How a widened ``patch_embedding`` is filled from published weights.

A task that declares condition streams makes the model's first conv wider than
the published weights expect, so warm-starting hits a shape mismatch on exactly
that one tensor. The task added the channels, so the task decides what goes in
them: published channels are copied into the prefix and the new tail starts at
zero, leaving the pretrained prior untouched at step 0.

Both hooks fire inside ``load_pretrained_weights`` and nowhere else -- they are
part of cold-starting from published weights, not of training snapshots, and
neither save nor resume goes through here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.nn as nn

from vforge.models.wan.model import WanChannelLayout

logger = logging.getLogger(__name__)


def patch_embedding_widener(
    layout: WanChannelLayout,
    *,
    log_label: str,
) -> Callable[[str, torch.Tensor, torch.Tensor], torch.Tensor | None]:
    """Build a shape resolver that copies published channels and zeroes the tail."""
    prior = layout.pretrained_in_channels

    def resolve(
        key: str,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor | None:
        if not key.endswith("patch_embedding.weight"):
            return None
        target_shape = tuple(target.shape)
        if source.shape[1] != prior or target_shape[1] != layout.in_channels:
            return None
        if tuple(source.shape[:1]) != target_shape[:1] or source.shape[2:] != target_shape[2:]:
            return None
        widened = source.new_zeros(target_shape)
        widened[:, :prior] = source
        logger.info(
            "%s: %s [%d->%d] input channels, tail zero-initialized",
            log_label,
            key,
            prior,
            layout.in_channels,
        )
        return widened

    return resolve


def condition_tail_initializer(
    layout: WanChannelLayout,
    *,
    transformer_of: Callable[[nn.Module], nn.Module] | None = None,
) -> Callable[[nn.Module], None]:
    """Build an initializer for widened channels missing from a checkpoint."""

    def initialize(model: nn.Module) -> None:
        transformer = transformer_of(model) if transformer_of is not None else model
        embedding = getattr(transformer, "patch_embedding", None)
        if embedding is None or not layout.is_widened:
            return
        weight = embedding.weight
        # FSDP may expose a DTensor sharded on dim 0; its local shard still
        # spans every input channel.
        local = weight.to_local() if hasattr(weight, "to_local") else weight
        with torch.no_grad():
            local[:, layout.pretrained_in_channels :].zero_()

    return initialize


__all__ = ["condition_tail_initializer", "patch_embedding_widener"]
