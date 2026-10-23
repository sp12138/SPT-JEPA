"""First-frame-conditioned bidirectional LTX teacher/critic prediction."""

from dataclasses import replace

import torch

from vforge.algorithms.dmd.model import DmdCriticAdapter
from vforge.distributed.parallel_state import get_parallel_state
from vforge.models.ltx.scheduler import to_denoised
from vforge.tasks.ltx.dmd.inputs import gather_prediction, modality, shard_modality
from vforge.tasks.ltx.train import audio_positions, video_positions


class LtxDmdCriticAdapter(DmdCriticAdapter):
    def score_window(self, generated, batch):
        inputs = batch.model_inputs
        if inputs.prefix is None:
            if inputs.segment_chain:
                # Long-tuning reference passes the generated window explicitly as
                # clean_video, including its first frame. Preserve old DMD defaults.
                return generated, replace(
                    batch,
                    model_inputs=replace(
                        inputs,
                        reference_video=generated["video"][:, : inputs.layout.frame_tokens],
                        score_window=True,
                    ),
                )
            return generated, batch
        values = {
            key: torch.cat([inputs.prefix[f"handover_{key}"], value], dim=1)
            for key, value in generated.items()
        }
        layout = inputs.layout
        device = values["video"].device
        return values, replace(
            batch,
            clean=values,
            model_inputs=replace(
                inputs,
                reference_video=inputs.prefix["handover_video"],
                score_window=True,
                video_positions=video_positions(
                    num_frames=values["video"].shape[1] // layout.frame_tokens,
                    height=layout.height,
                    width=layout.width,
                    fps=inputs.fps,
                    batch_size=1,
                    device=device,
                ),
                audio_positions=audio_positions(
                    num_time_steps=values["audio"].shape[1], batch_size=1, device=device
                ),
            ),
        )

    def generated_region(self, values, batch):
        if batch.model_inputs.prefix is None:
            return values
        return {
            key: value[:, batch.model_inputs.prefix[f"handover_{key}"].shape[1] :]
            for key, value in values.items()
        }

    def predict_x0(self, tower, noisy, noise_level, batch, conditioning):
        inputs = batch.model_inputs
        if conditioning not in {"conditional", "unconditional"}:
            raise ValueError("unknown score conditioning")
        vc = (
            inputs.video_context
            if conditioning == "conditional"
            else inputs.negative["video_context"]
        )
        ac = (
            inputs.audio_context
            if conditioning == "conditional"
            else inputs.negative["audio_context"]
        )
        video = noisy["video"].clone()
        fs = inputs.layout.frame_tokens
        video[:, :fs] = inputs.reference_video
        sigma = noise_level.model_timestep
        vmod = modality(video, sigma, inputs.video_positions, vc, first_frame_tokens=fs)
        amod = modality(noisy["audio"], sigma, inputs.audio_positions, ac)
        state = get_parallel_state()
        vv, av = tower(video=shard_modality(vmod, state), audio=shard_modality(amod, state))
        vv = gather_prediction(vv, vmod, state)
        av = gather_prediction(av, amod, state)
        return {
            "video": to_denoised(video, vv, vmod.timesteps.unsqueeze(-1)),
            "audio": to_denoised(noisy["audio"], av, amod.timesteps.unsqueeze(-1)),
        }

    def predict_velocity(self, tower, noisy, noise_level, batch, conditioning):
        # Preserve reference BF16 x0 roundtrip in the critic regression path.
        x0 = self.predict_x0(tower, noisy, noise_level, batch, conditioning)
        sigma = noise_level.model_timestep.clamp_min(1e-8)
        return {key: (value - x0[key]) / sigma for key, value in noisy.items()}
