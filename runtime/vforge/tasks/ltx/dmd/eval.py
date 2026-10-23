"""Native DMD evaluation: cached conditioning, shared rollout and generator-only weights."""

import math
import os
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from vforge.algorithms.dmd.model import RoleModelBundle
from vforge.algorithms.dmd.runtime import DmdRolloutSchedule
from vforge.data.loader import broadcast_batch, build_eval_loader
from vforge.data.video_io import write_video_mp4
from vforge.distributed.parallel_state import get_parallel_state
from vforge.distributed.runtime import get_runtime
from vforge.evaluation.runner import EvalComponents, EvalOutputWriter, sample_artifact_name
from vforge.models.factory import ModelSpec
from vforge.models.ltx.decoding import LtxDecoder
from vforge.models.ltx.ops import dmd_transformer_ops
from vforge.models.ltx.text_encoder import encode_prompts, load_connectors
from vforge.tasks.eval_source import open_eval_dataset
from vforge.tasks.ltx.data import LtxPrecomputedDataset
from vforge.tasks.ltx.dmd.generator import ForwardView, LtxDmdGeneratorAdapter, generate
from vforge.tasks.ltx.dmd.inputs import prepare_batch
from vforge.tasks.ltx.dmd.shared import build_model, build_weight_source
from vforge.tasks.requests import validate_eval_source
from vforge.training.checkpoint.mapping import StateDictMapper
from vforge.training.checkpoint.source import TrainingCheckpointWeightSource


def latent_frames_for_seconds(seconds, fps, first=4, following=3):
    if (
        not math.isfinite(seconds)
        or not math.isfinite(fps)
        or min(seconds, fps, first, following) <= 0
    ):
        raise ValueError("DMD duration, FPS and block lengths must be positive")
    frames = round(seconds * fps) // 8 + 1
    return first + max(0, frames - first) // following * following


class GeneratorEvalModel(RoleModelBundle):
    """Preserve the training checkpoint prefix while exposing a single forward."""

    def __init__(self, generator):
        super().__init__({"generator": generator})

    def forward(self, **kwargs):
        return self.generator(**kwargs)


class PromptDataset(Dataset):
    """Use one selected reference sample for each supplied prompt."""

    def __init__(self, dataset, prompts):
        self.dataset, self.prompts = dataset, prompts

    def __len__(self):
        return len(self.dataset) * len(self.prompts)

    def __getitem__(self, index):
        sample = self.dataset[index // len(self.prompts)]
        prompt_index = index % len(self.prompts)
        return {
            **sample,
            "sample_id": f"{sample['sample_id']}_prompt_{prompt_index:04d}",
            "__prompt_index__": prompt_index,
        }


def _single_sample(samples):
    if len(samples) != 1:
        raise ValueError("LTX eval requires batch_size=1")
    return samples[0]


class LtxDmdInference:
    def __init__(self, model, forward, config, connectors, contexts, seed):
        self.model = ForwardView(model.generator, forward)
        self.config, self.connectors, self.contexts, self.seed = config, connectors, contexts, seed

    def resolve_seed(self, _sample_id, _seed):
        # Reference inference uses the explicitly supplied seed, including sample-index > 0.
        return self.seed

    @torch.no_grad()
    def infer_sample(self, sample, *, seed):
        device = get_runtime().device
        fps = float(sample["video_latents"].get("fps") or self.config["frame_rate"])
        frames = latent_frames_for_seconds(
            self.config["seconds"],
            fps,
            self.config["first_block_latent_frames"],
            self.config["block_latent_frames"],
        )
        # Different rollout counts across FSDP ranks would mismatch collectives.
        if dist.is_initialized():
            counts = [None] * dist.get_world_size()
            dist.all_gather_object(counts, frames)
            if len(set(counts)) != 1:
                raise ValueError(
                    "DMD eval batches must share FPS/block count; "
                    "select one sample or group equal-FPS samples"
                )
        contexts = self.contexts[sample["__prompt_index__"]] if self.contexts else None
        batch = prepare_batch(
            sample,
            device,
            config=self.config,
            connectors=self.connectors,
            negative={},
            latent_frames=frames,
            contexts=contexts,
        )
        values = generate(
            self.model,
            batch,
            seed=seed,
            steps=self.config["denoising_steps"],
            kv_cache_blocks=self.config["kv_cache_blocks"],
            num_sink_blocks=self.config["num_sink_blocks"],
        )
        return {
            "sample_id": sample["sample_id"],
            "latents": {k: v.cpu() for k, v in values.items()},
            "layout": asdict(batch.model_inputs.layout),
            "fps": fps,
            "metadata": {
                "duration_seconds": ((batch.model_inputs.layout.frames - 1) * 8 + 1) / fps,
                "fps": fps,
                "latent_frames": batch.model_inputs.layout.frames,
            },
        }


def generator_weight_source(request):
    path, format = request.checkpoint_path, request.checkpoint_format
    if format == "safetensors" or (format == "auto" and path.endswith(".safetensors")):
        return build_weight_source(
            pretrained_path=request.pretrained_path,
            weights_path=path,
            model_path="generator",
            causal=True,
            param_dtype=torch.float32,
            source_prefix=str(request.extras.get("source_prefix", "")),
        )
    if format == "pt" or (format == "auto" and os.path.isfile(path)):

        def select_generator(state):
            selected = {key: value for key, value in state.items() if key.startswith("generator.")}
            if not selected:
                raise ValueError("vForge PT checkpoint has no generator subtree")
            return selected

        return TrainingCheckpointWeightSource(
            path,
            format="pt",
            mapper=StateDictMapper(export=dict, import_=select_generator),
        )
    return TrainingCheckpointWeightSource(path, model_path="generator", format=format)


def build_eval_components(request):
    validate_eval_source(request)
    if request.data_source != "cache" or len(request.sources) != 1:
        raise ValueError("LTX DMD eval requires evaluation.sources with one cache_dir")
    if request.precision != "bf16":
        raise ValueError("LTX DMD eval requires BF16")
    state = get_parallel_state()
    if state.config.cp_size != 1 or state.config.tp_size != 1:
        raise ValueError("LTX DMD eval requires TP=CP=1")
    config = dict(request.extras)
    LtxDmdGeneratorAdapter(
        kv_cache_blocks=config["kv_cache_blocks"], num_sink_blocks=config["num_sink_blocks"]
    )
    DmdRolloutSchedule(tuple(step / 1000 for step in config["denoising_steps"]))
    if request.guidance_scale not in {None, 1.0}:
        raise ValueError("DMD generator inference does not use CFG")
    if config["prompt"] and config["prompt_file"]:
        raise ValueError("choose model.extras.prompt or model.extras.prompt_file")
    prompts = [config["prompt"]] if config["prompt"] else []
    if config["prompt_file"]:
        prompts = [
            line.strip()
            for line in Path(config["prompt_file"]).read_text().splitlines()
            if line.strip()
        ]
        if not prompts:
            raise ValueError("prompt_file contains no prompts")
    dataset = open_eval_dataset(
        request,
        codec=None,
        open_encoded_source=lambda source, _codec: LtxPrecomputedDataset(
            source.cache_dir, sources=("latents", "conditions")
        ),
    )
    if prompts:
        if request.sample_index is None:
            raise ValueError("custom prompts require evaluation.sample_index to select a reference")
        dataset = PromptDataset(dataset, prompts)
    device = get_runtime().device
    # Rank 0 encodes before materializing the generator, then releases Gemma.
    # Keep the prompt file's contexts on CPU rather than accumulating GPU memory.
    contexts = []
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    if prompts:
        encoded, error = None, [None]
        if is_main:
            try:
                encoded = encode_prompts(
                    request.pretrained_path,
                    config["text_encoder_path"],
                    prompts,
                    device=device,
                    max_length=config["max_text_length"],
                )
            except Exception as exc:
                error[0] = f"{type(exc).__name__}: {exc}"
        if dist.is_initialized():
            dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(f"LTX prompt encoding failed: {error[0]}")
        encoded = broadcast_batch(encoded, group=None, is_source=is_main, target_device=device)
        contexts = [{key: value.cpu() for key, value in item.items()} for item in encoded]
    connectors = None if contexts else load_connectors(request.pretrained_path, device=device)
    if not isinstance(request.checkpoint_path, str) or not request.checkpoint_path:
        raise ValueError(
            "evaluation.checkpoint_path must name a generator safetensors or vForge checkpoint"
        )
    model = ModelSpec(
        build=lambda: GeneratorEvalModel(
            build_model(
                pretrained_path=request.pretrained_path,
                init_device=request.init_device,
                causal=True,
                param_dtype=torch.float32,
                # The reference inference entry uses native BF16 norms and modulation.
                # Training keeps its default FP32 norm policy.
                ops=dmd_transformer_ops(norm_dtype=None),
            )
        )
    )
    weights = generator_weight_source(request)
    decoder = LtxDecoder(request.pretrained_path, device)

    def write(result, output_dir):
        from vforge.tasks.ltx.dmd.inputs import BlockLayout

        path = Path(output_dir) / sample_artifact_name(result["sample_id"], suffix=".mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        video, audio, rate = decoder.decode(result["latents"], BlockLayout(**result["layout"]))
        temporary = path.with_suffix(".tmp.mp4")
        try:
            write_video_mp4(
                video, str(temporary), fps=result["fps"], audio=audio, audio_sample_rate=rate
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return str(path)

    return EvalComponents(
        model=model,
        dataloader=build_eval_loader(
            dataset,
            parallel_state=state,
            collate_fn=_single_sample,
            num_samples=request.num_samples,
            sample_selection=request.sample_selection,
        ),
        initial_weights=weights,
        build_inference=lambda model, forward: LtxDmdInference(
            model, forward, config, connectors, contexts, request.seed
        ),
        output_writer=EvalOutputWriter(
            artifact_name=lambda sample_id: sample_artifact_name(sample_id, suffix=".mp4"),
            write_output=write,
        ),
    )
