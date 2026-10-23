"""Wan compatibility names for neutral flow-match inference primitives."""

from vforge.models.video_diffusion.wan import (
    WAN_DEFAULT_NEGATIVE_PROMPT,
    WanFlowMatchPipelineBase,
)

DEFAULT_NEGATIVE_PROMPT = WAN_DEFAULT_NEGATIVE_PROMPT

__all__ = ["DEFAULT_NEGATIVE_PROMPT", "WanFlowMatchPipelineBase"]
