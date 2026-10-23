"""Opt-in inference components shared by compatible video diffusion models."""

from vforge.models.video_diffusion.flow_match import (
    FlowMatchPredictor,
    build_first_frame_per_token_timestep,
    run_flow_match_denoising,
)
from vforge.models.video_diffusion.wan import (
    WAN_DEFAULT_NEGATIVE_PROMPT,
    WanFlowMatchPipelineBase,
    encode_wan_negative_prompt,
)

__all__ = [
    "FlowMatchPredictor",
    "WanFlowMatchPipelineBase",
    "WAN_DEFAULT_NEGATIVE_PROMPT",
    "build_first_frame_per_token_timestep",
    "encode_wan_negative_prompt",
    "run_flow_match_denoising",
]
