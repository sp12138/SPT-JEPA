"""LTX model-owned hooks for the block_distribution_matching algorithm."""

from functools import partial

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from vforge.algorithms.dmd.config import SegmentChainDistributionMatchingConfig
from vforge.algorithms.dmd.model import DmdDataSupport, DmdFamilySpec
from vforge.data.cache import digest_sample_ids
from vforge.data.loader import broadcast_batch
from vforge.distributed.parallel_state import get_parallel_state
from vforge.distributed.runtime import get_runtime
from vforge.models.ltx.text_encoder import (
    encode_negative_prompt,
    load_connectors,
)
from vforge.tasks.ltx.data import LtxPrecomputedDataset
from vforge.tasks.ltx.dmd.critic import LtxDmdCriticAdapter
from vforge.tasks.ltx.dmd.generator import LtxDmdGeneratorAdapter
from vforge.tasks.ltx.dmd.inputs import (
    block_layout,
    loss_partitions,
    prepare_batch,
    resolve_rollout_noise,
    resolve_score_noise,
    valid_masks,
)
from vforge.tasks.ltx.dmd.sampler import BlockCountSampler
from vforge.tasks.ltx.dmd.shared import build_model, build_weight_source
from vforge.tasks.requests import single_offline_source


def validate_ltx_dmd(request, engine):
    long_tuning = request.dmd.get("algorithm") == "segment_chain_distribution_matching"
    if request.dmd.get("algorithm") not in {
        "block_distribution_matching",
        "segment_chain_distribution_matching",
    }:
        raise ValueError("LTX DMD requires block_distribution_matching")
    if engine != "fsdp2" or request.tp_size != 1:
        raise NotImplementedError("LTX DMD requires FSDP2 with TP/CP=1")
    if request.batch_size != 1 or request.latent_cache_mode != "offline":
        raise ValueError("LTX DMD requires batch_size=1 and offline latent cache")
    if request.precision != "bf16":
        raise ValueError("LTX DMD numerical contract requires bf16")
    if single_offline_source(request, recipe="LTX DMD") is None:
        raise ValueError("LTX DMD requires one offline cache source")
    for role, variant in (
        ("generator", "causal"),
        ("real_score", "bidirectional"),
        ("fake_score", "bidirectional"),
    ):
        model = request.training_models.get(role)
        if model is None or model.config.get("variant") != variant:
            raise ValueError(f"LTX DMD requires {role}.config.variant={variant}")
        if not model.pretrained_path or (
            not model.init_weights and not (long_tuning and role != "generator")
        ):
            raise ValueError(f"LTX DMD requires {role} base checkpoint and init_weights")
        if model.init_weights and not model.init_weights.endswith(".safetensors"):
            raise ValueError("LTX DMD initialization requires a flat safetensors checkpoint")
        if model.weight_loading == "distributed":
            raise ValueError(
                "LTX single-file initialization uses rank0_broadcast or replicated loading"
            )
    config = request.training_models["generator"].config
    if not config.get("text_encoder_path"):
        raise ValueError("LTX DMD requires a local Gemma text_encoder_path")
    if not 25 <= int(config["max_train_frames"]) <= 241:
        raise ValueError("LTX DMD max_train_frames must be between 25 and 241")


def build_role_model(request, role):
    model = request.training_models[role]
    # Preserve FP32 trainable masters and BF16 frozen teacher parameters.
    return build_model(
        pretrained_path=model.pretrained_path,
        init_device=model.init_device,
        causal=role == "generator",
        param_dtype=torch.bfloat16 if role == "real_score" else torch.float32,
    )


def build_weight_sources(request, roles):
    sources = []
    for role in roles:
        model = request.training_models[role]
        sources.append(
            build_weight_source(
                pretrained_path=model.pretrained_path,
                weights_path=model.init_weights,
                model_path=role,
                causal=role == "generator",
                param_dtype=torch.bfloat16 if role == "real_score" else torch.float32,
                source_prefix=str(model.config.get("source_prefix", "")),
                weight_loading=model.weight_loading,
            )
        )
    return tuple(sources)


def _single_sample(samples):
    if len(samples) != 1:
        raise ValueError("LTX DMD batch_size must be one")
    return samples[0]


def build_data_support(request):
    parallel = get_parallel_state()
    if parallel.config.cp_size != 1:
        raise NotImplementedError("LTX DMD requires CP=1")
    runtime = get_runtime()
    device = runtime.device if runtime is not None else torch.device("cpu")
    config = request.training_models["generator"].config
    base = request.training_models["generator"].pretrained_path
    distributed = dist.is_available() and dist.is_initialized()
    source_rank = not distributed or dist.get_rank() == 0
    negative = None
    error = [None]
    if source_rank:
        try:
            negative = encode_negative_prompt(
                base,
                config["text_encoder_path"],
                config["negative_prompt"],
                device=device,
                max_length=int(config["max_text_length"]),
            )
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if distributed:
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"LTX negative prompt initialization failed: {error[0]}")
    negative = broadcast_batch(negative, group=None, is_source=source_rank, target_device=device)
    # Gemma is released before any of the three train-time transformer towers.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    connectors = load_connectors(base, device=device)
    source = single_offline_source(request, recipe="LTX DMD")
    if source is None:
        raise ValueError("LTX DMD requires one offline cache source")
    dataset = LtxPrecomputedDataset(source.cache_dir, sources=("latents", "conditions"))
    # mmap reads metadata/tensor shape without allocating the full latent cache.
    # Never create an index beside the read-only input files.
    chain = (
        SegmentChainDistributionMatchingConfig.from_mapping(request.dmd).chain
        if request.dmd.get("algorithm") == "segment_chain_distribution_matching"
        else None
    )
    counts = []
    for _, paths in dataset._samples:
        video = torch.load(paths["latents"], map_location="cpu", weights_only=True, mmap=True)
        latent = video["latents"]
        frames = int(video["num_frames"]) if latent.ndim == 2 else latent.shape[1]
        frames = min(frames, int(config["max_train_frames"]) // 8 + 1)
        layout = block_layout(
            frames,
            1,
            1,
            float(config["frame_rate"]),
            int(config["first_block_latent_frames"]),
            int(config["block_latent_frames"]),
        )
        counts.append(chain.segment_slots if chain else len(layout.video_ranges))

    def build_stream(offset):
        seed = int(request.sampler_seed) + offset
        sampler = BlockCountSampler(
            counts,
            num_replicas=int(parallel.config.dp_size),
            rank=int(parallel.dp_rank),
            seed=seed,
            shuffle=request.shuffle,
            drop_last=request.drop_last,
        )
        loader = StatefulDataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=request.num_workers,
            collate_fn=_single_sample,
            pin_memory=request.pin_memory,
            persistent_workers=request.persistent_workers,
            prefetch_factor=request.prefetch_factor,
            generator=torch.Generator().manual_seed(seed),
        )
        return loader, {
            "dataset": "ltx_dmd",
            "sample_ids_digest": digest_sample_ids(dataset.sample_ids),
            "block_counts": counts,
            "sampler_seed": seed,
            "dp_size": int(parallel.config.dp_size),
            "shuffle": bool(request.shuffle),
            "drop_last": bool(request.drop_last),
        }

    return DmdDataSupport(
        build_train_stream=build_stream,
        prepare_batch=partial(
            prepare_batch,
            config=config,
            connectors=connectors,
            negative=negative,
            segment_slots=chain.segment_slots if chain else None,
        ),
        resolve_noise_level=resolve_score_noise,
        resolve_rollout_noise_level=resolve_rollout_noise,
        score_noise_shape=lambda _batch: (1,),
        valid_masks=valid_masks,
        loss_partitions=loss_partitions,
    )


LTX23_DMD = DmdFamilySpec(
    model_name="ltx23",
    validate=validate_ltx_dmd,
    build_role_model=build_role_model,
    build_data_support=build_data_support,
    create_generator_adapter=lambda request: LtxDmdGeneratorAdapter(
        chain=(
            SegmentChainDistributionMatchingConfig.from_mapping(request.dmd).chain
            if request.dmd.get("algorithm") == "segment_chain_distribution_matching"
            else None
        ),
        base_path=request.training_models["generator"].pretrained_path,
    ),
    create_critic_adapter=lambda _request, _role: LtxDmdCriticAdapter(),
    build_weight_sources=build_weight_sources,
)
