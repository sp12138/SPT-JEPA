"""Wan Taichu implementation of the generic DMD critic adapter."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from vforge.algorithms.dmd.model import (
    DmdCriticAdapter,
    DmdNoiseLevel,
    LatentBundle,
    PreparedDmdBatch,
)
from vforge.tasks.wan_taichu.dmd.inputs import WanDmdInputs, wan_dmd_token_timesteps


class WanDmdCriticAdapter(DmdCriticAdapter):
    """Call one bidirectional Wan tower and return canonical frame-first velocity."""

    def __init__(self, *, compute_dtype: torch.dtype) -> None:
        self.compute_dtype = compute_dtype

    def predict_velocity(
        self,
        tower: nn.Module,
        noisy: LatentBundle,
        noise_level: DmdNoiseLevel[torch.Tensor],
        batch: PreparedDmdBatch[WanDmdInputs],
        conditioning: Literal["conditional", "unconditional"],
    ) -> dict[str, torch.Tensor]:
        latent = noisy["latent"]
        channel_first = latent.permute(0, 2, 1, 3, 4)
        hidden_states = torch.cat(
            (channel_first, batch.model_inputs.condition_latents),
            dim=1,
        )
        prompt = (
            batch.model_inputs.prompt_embeds
            if conditioning == "conditional"
            else batch.model_inputs.negative_prompt_embeds
        )
        timestep = wan_dmd_token_timesteps(
            noise_level.model_timestep,
            hidden_states,
            tuple(int(value) for value in tower.config.patch_size),
        )
        # Taichu routes on the scheduler's FP32 timestep before entering the
        # selected expert's FSDP forward. Keep that order: BF16 rounds levels
        # in (898, 900) up to 900, which can send DP replicas to different FSDP
        # units even when they sampled the same noise band. Each expert is an
        # FSDP execution root, so call the selected expert directly.
        select_transformer = getattr(tower, "select_transformer", None)
        if callable(select_transformer):
            tower = select_transformer(timestep)
        # Preserve Taichu's mixed-precision timestep inside the selected expert;
        # FP32 parameter storage does not change this conditioning precision.
        timestep = timestep.to(dtype=self.compute_dtype)
        output = tower(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=prompt,
            attention_kwargs=None,
            return_dict=False,
        )
        if isinstance(output, tuple):
            output = output[0]
        elif hasattr(output, "sample"):
            output = output.sample
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Wan score tower returned {type(output).__name__}, expected Tensor")
        return {"latent": output[:, : channel_first.shape[1]].permute(0, 2, 1, 3, 4)}


__all__ = ["WanDmdCriticAdapter"]
