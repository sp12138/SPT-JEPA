"""Native evaluation assembly, device/precision boundary and atomic MP4 output."""

from __future__ import annotations

import math
import os
from functools import partial
from itertools import islice

import torch
from torch.utils.data import Subset

from vforge.data.loader import build_eval_loader
from vforge.data.video_io import write_video_mp4
from vforge.distributed.parallel_state import get_parallel_state
from vforge.evaluation.runner import (
    EvalComponents,
    EvalOutputWriter,
    _stable_sample_salt,
    sample_artifact_name,
)
from vforge.models.rep_world.inference import RepWorldInference, RolloutConfig
from vforge.tasks.rep_world.data import VideoClipDataset
from vforge.tasks.rep_world.ema import load_evaluation_ema
from vforge.tasks.rep_world.shared import CodecProvider, model_spec, parse_config, pretrained_source
from vforge.tasks.requests import validate_eval_source
from vforge.training.checkpoint import TrainingCheckpointWeightSource


def collate_sample(samples):
    if len(samples) != 1:
        raise ValueError("RepWorld evaluation requires one clip per batch")
    sample = samples[0]
    return {**sample, "rgb": sample["rgb"].unsqueeze(0)}


def write_prediction(result, output_dir, *, fps):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, sample_artifact_name(result["sample_id"], suffix=".mp4"))
    # Retain .mp4 as the suffix so the video writer can infer its container.
    temporary = path + ".tmp.mp4"
    write_video_mp4(result["prediction"], temporary, fps=fps)
    os.replace(temporary, path)
    return path


class BatchWindowEvalLoader:
    """Keep Native per-case ordering/padding while exposing bounded batch windows.

    Native still makes one collective skip decision and writes one artifact per
    case. RuntimeInference computes each window at its first non-skipped case.
    Equal-length DP sampler shards ensure every rank enters that batched forward
    on the same iteration, even when only part of a window needs regeneration.
    """

    def __init__(self, loader, batch_size):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("rollout.batch_size must be a positive integer")
        self.loader = loader
        self.batch_size = batch_size

    @property
    def global_length(self):
        return self.loader.global_length

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        window_index = 0
        while members := tuple(islice(iterator, self.batch_size)):
            for offset, sample in enumerate(members):
                yield {
                    **sample,
                    "__repworld_eval_window__": members,
                    "__repworld_eval_window_index__": window_index,
                    "__repworld_eval_offset__": offset,
                }
            window_index += 1


def batch_eval_loader(loader, batch_size):
    return loader if batch_size == 1 else BatchWindowEvalLoader(loader, batch_size)


class RuntimeInference:
    """Only transport/precision here; state rollout belongs to models/."""

    def __init__(self, model, engine_forward, provider, config, rollout):
        self.device = next(model.parameters()).device
        self.dtype = provider.dtype
        self._batch_key = None
        self._batch_results = {}
        self.inference = RepWorldInference(
            engine_forward,
            provider.get(self.device),
            config.geometry,
            config.flow,
            config.conditioner,
            model.timestep_shift,
            rollout,
        )

    def infer_sample(self, sample, *, seed):
        members = sample.get("__repworld_eval_window__")
        if members is None:
            sample = {**sample, "rgb": sample["rgb"].to(device=self.device, dtype=torch.float32)}
            with (
                torch.no_grad(),
                torch.autocast(
                    self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
                ),
            ):
                result = self.inference.infer_sample(sample, seed=seed)
                result["metadata"].update(getattr(self, "weight_metadata", {}))
                return result

        key = (
            sample["__repworld_eval_window_index__"],
            tuple(str(member["sample_id"]) for member in members),
        )
        offset = sample["__repworld_eval_offset__"]
        if key != self._batch_key:
            self._batch_results.clear()
            base_seed = int(seed) - _stable_sample_salt(str(sample["sample_id"]))
            seeds = [
                base_seed + _stable_sample_salt(str(member["sample_id"])) for member in members
            ]
            observed_frames = (
                self.inference.config.history_states * self.inference.geometry.rgb_frames
            )
            inputs = [
                {
                    "sample_id": member["sample_id"],
                    "rgb": member["rgb"][:, :, :observed_frames].to(
                        device=self.device, dtype=torch.float32
                    ),
                }
                for member in members
            ]
            with (
                torch.no_grad(),
                torch.autocast(
                    self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
                ),
            ):
                results = self.inference.infer_batch(inputs, seeds=seeds)
            self._batch_results = dict(enumerate(results))
            self._batch_key = key
            for result in results:
                result["metadata"]["sampling_batch_size"] = len(members)
        result = self._batch_results.pop(offset)
        if result["sample_id"] != str(sample["sample_id"]) or result["metadata"]["seed"] != seed:
            raise RuntimeError("Batched result does not match the Native sample identity/seed")
        result["metadata"].update(getattr(self, "weight_metadata", {}))
        return result


def build_eval_components(request):
    validate_eval_source(request)
    if request.data_source not in {"validation", "inference"} or request.sources:
        raise ValueError("RepWorld evaluation requires an inference/validation video manifest")
    if request.latent_cache_mode != "online":
        raise ValueError("RepWorld evaluation requires latent_cache.mode=online")
    if not isinstance(request.checkpoint_path, str):
        raise ValueError("RepWorld evaluation.checkpoint_path must be a single path")
    config = parse_config(request.extras)
    options = dict(config.rollout)
    fps = options.pop("fps", 16)
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps < 1:
        raise ValueError("rollout.fps must be finite and >= 1")
    rollout = RolloutConfig(
        **{"total_states": config.states, "history_window_states": config.states, **options},
        trained_states=config.states,
        steps=request.num_inference_steps if request.num_inference_steps is not None else 50,
        cfg_scale=request.guidance_scale if request.guidance_scale is not None else 2.0,
    )
    if config.flow.mode in {"pure_tf", "pure_df"} and rollout.cfg_scale != 1:
        raise ValueError("Pure TF/DF has no semantic CFG; set evaluation.guidance_scale=1")
    if request.checkpoint_path:
        source = TrainingCheckpointWeightSource(
            path=request.checkpoint_path, format=request.checkpoint_format
        )
    else:
        source = pretrained_source(request.pretrained_path, config)
        if source is None:
            raise ValueError(
                "RepWorld eval requires evaluation.checkpoint_path or model.pretrained_path"
            )
    if config.ema.enabled and not request.checkpoint_path:
        raise ValueError("EMA model evaluation requires a Native training checkpoint")
    initial_weights = source
    if config.ema.enabled:
        # The full training root also owns persistent EMA buffers. Load each
        # online tower strictly through Native component selection; EMA loading
        # below uses the same API and validates its saved step/parameter order.
        paths = ["net"]
        if config.conditioner is not None:
            paths.append("causal_semantic_encoder")
        initial_weights = tuple(
            TrainingCheckpointWeightSource(path=source.path, format=source.format, model_path=path)
            for path in paths
        )
    frames = (
        rollout.total_states if request.data_source == "validation" else rollout.history_states
    ) * config.geometry.rgb_frames
    dataset = VideoClipDataset(
        request.manifest, frames=frames, frame_interval=config.frame_interval, random_clip=False
    )
    if request.sample_index is not None:
        if request.sample_index >= len(dataset):
            raise ValueError("evaluation.sample_index is outside the video manifest")
        dataset = Subset(dataset, [request.sample_index])
    loader = build_eval_loader(
        dataset,
        parallel_state=get_parallel_state(),
        collate_fn=collate_sample,
        num_samples=request.num_samples,
        sample_selection=request.sample_selection,
    )
    provider = CodecProvider(config, request.precision)

    def build_inference(model, engine_forward):
        ema_updates = None
        if config.ema.evaluation_weights == "ema":
            ema_updates = load_evaluation_ema(model, config.ema, source)
        inference = RuntimeInference(model, engine_forward, provider, config, rollout)
        inference.weight_metadata = {
            "evaluation_weights": config.ema.evaluation_weights,
            "ema_decay": config.ema.decay if config.ema.enabled else None,
            "ema_updates": ema_updates,
        }
        return inference

    return EvalComponents(
        model=model_spec(config, request.init_device),
        dataloader=batch_eval_loader(loader, rollout.batch_size),
        initial_weights=initial_weights,
        build_inference=build_inference,
        output_writer=EvalOutputWriter(
            artifact_name=lambda sample_id: sample_artifact_name(sample_id, suffix=".mp4"),
            write_output=partial(write_prediction, fps=fps),
        ),
    )
