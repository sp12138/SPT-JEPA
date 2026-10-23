"""Pieces every Wan version's task assembly needs.

Scope is deliberately narrow: only helpers that carry no version-specific
content live here. Anything that encodes a particular version's data contract
-- its encoded-bundle schema, its condition streams, its condition encoder --
stays with that version, even when a later version inherits it. Wan 2.2 A14B
reuses the 2.1 contract wholesale, and it imports those pieces from
``vforge.tasks.wan.v21.shared`` precisely so that the inheritance stays legible
rather than being laundered through a neutral name.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vforge.data.cache import (
    CACHE_SCHEMA_VERSION,
    EncodedBatchCodec,
    digest_sample_ids,
    read_cache_manifest,
)
from vforge.data.dataset import (
    EncodedCacheDataset,
    RawSampleDataset,
    manifest_file_digest,
)
from vforge.data.loader import build_resumable_loader
from vforge.data.mixed_dataset import (
    build_mixed_resumable_loader,
    open_mixed_cache_dataset,
    open_mixed_raw_dataset,
    wrap_latent_bounds,
)
from vforge.data.taichu_index_cache import open_encoded_cache_dataset
from vforge.distributed.parallel_state import get_parallel_state
from vforge.tasks.requests import DataSourceSpec, TrainRequest
from vforge.training.checkpoint.mapping import StateDictMapper


@dataclass(frozen=True)
class WanConditionSpec:
    """Resolved latent-stream contract a task hands to model code.

    ``mask_channels`` is the prefix carried by every stream.  Wan 2.1-shaped
    streams use four temporal-mask channels; TI2V streams are plain VAE
    latents and therefore use zero.
    """

    streams: tuple[str, ...] = ()
    mask_channels: int = 0

    def __post_init__(self) -> None:
        streams = tuple(str(name) for name in self.streams)
        if any(not name for name in streams):
            raise ValueError("Wan condition stream names must be non-empty")
        if len(set(streams)) != len(streams):
            raise ValueError(f"Wan condition streams must be unique; got {streams!r}")
        if self.mask_channels < 0:
            raise ValueError(
                f"Wan condition mask_channels must be non-negative; got {self.mask_channels}"
            )
        object.__setattr__(self, "streams", streams)

    def channels_per_stream(self, *, z_dim: int) -> int:
        if z_dim <= 0:
            raise ValueError(f"Wan z_dim must be positive; got {z_dim}")
        return int(z_dim) + self.mask_channels

    def input_channels(self, *, z_dim: int) -> int:
        """Noisy latent plus all declared condition blocks."""
        return int(z_dim) + len(self.streams) * self.channels_per_stream(z_dim=z_dim)


def parse_condition_streams(value: Any, *, known: Sequence[str]) -> tuple[str, ...]:
    """Read a configured stream list, checked against what a version can encode.

    The order streams are listed in is the order they are concatenated onto the
    noisy latent, so it is kept as written rather than sorted. A bare string is
    taken as a single stream, which keeps ``condition_streams: normal`` working
    and turns a fused spelling into a named-stream error instead of silently
    iterating characters.
    """
    if value is None:
        return ()
    names = [value] if isinstance(value, str) else list(value)
    streams = tuple(str(name) for name in names)
    unknown = [name for name in streams if name not in set(known)]
    if unknown:
        raise ValueError(
            f"unknown condition stream(s) {unknown} in model.extras.condition_streams; "
            f"this model encodes {list(known)}"
        )
    if len(set(streams)) != len(streams):
        raise ValueError(f"model.extras.condition_streams must be unique; got {list(streams)}")
    return streams


def read_condition_streams(
    extras: Mapping[str, Any],
    *,
    known: Sequence[str],
    default: Sequence[str],
) -> tuple[str, ...]:
    """Read ``model.extras.condition_streams`` for a version's known set.

    ``extras`` is free-form, so the key this one replaced is rejected by name:
    left to fall through it would take the default width and train on the wrong
    channels without saying so.
    """
    if "condition_mode" in extras:
        raise ValueError(
            "model.extras.condition_mode was replaced by "
            "model.extras.condition_streams, which names the streams as a list "
            f"in concatenation order, e.g. [{', '.join(known)}]"
        )
    return parse_condition_streams(extras.get("condition_streams", default), known=known)


def validate_shared(*, latent_cache_mode: str, usp_size: int) -> None:
    """Check the two request fields every Wan verb reads the same way."""
    if usp_size < 1:
        raise ValueError("parallel.usp_size must be >= 1")
    if latent_cache_mode not in {"online", "offline"}:
        raise ValueError(f"unsupported latent_cache.mode={latent_cache_mode!r}")


def model_config_overrides(extras: Mapping[str, Any]) -> dict[str, Any]:
    """Forward ``model.extras.wan_*`` keys to the model constructor."""
    return {
        key.removeprefix("wan_"): value for key, value in extras.items() if key.startswith("wan_")
    }


def parameter_storage_dtype() -> torch.dtype:
    """Keep optimizer-facing weights in FP32; FSDP controls compute dtype."""
    return torch.float32


def build_state_dict_mapper() -> StateDictMapper:
    return StateDictMapper.strip_prefix("transformer.")


def _apply_source_bounds(dataset: Any, source: DataSourceSpec | None) -> Any:
    if source is None:
        return dataset
    return wrap_latent_bounds(
        dataset,
        min_latents=source.min_latents,
        max_latents=source.max_latents,
    )


def _source_bound_fields(source: DataSourceSpec | None) -> dict[str, Any]:
    if source is None:
        return {}
    if source.min_latents is None and source.max_latents is None:
        return {}
    return {
        "min_latents": source.min_latents,
        "max_latents": source.max_latents,
    }


def _raw_source_mapping(source: DataSourceSpec) -> dict[str, Any]:
    return {
        "name": source.name,
        "kind": source.kind,
        "index_files": tuple(source.index_files),
        "frames": source.frames,
        "height": source.height,
        "width": source.width,
        "max_sequence_length": source.max_sequence_length,
        "weight": source.weight,
        "min_latents": source.min_latents,
        "max_latents": source.max_latents,
        "extras": dict(source.extras or {}),
    }


def _split_sources(
    sources: Sequence[DataSourceSpec],
    *,
    latent_cache_mode: str,
) -> tuple[tuple[DataSourceSpec, ...], tuple[DataSourceSpec, ...]]:
    """Partition ``data.sources`` into raw and encoded, rejecting bad mixes."""
    raw = tuple(source for source in sources if source.kind)
    encoded = tuple(source for source in sources if not source.kind)
    if raw and encoded:
        raise ValueError(
            "data.sources mixes raw sources (kind set) with encoded caches; "
            "one training run decodes video or reads latents, not both"
        )
    if encoded and latent_cache_mode != "offline":
        raise ValueError("data.sources holds offline encoded caches; set latent_cache.mode=offline")
    if raw and latent_cache_mode != "online":
        raise ValueError(
            "raw data.sources entries decode video during training; set latent_cache.mode=online"
        )
    for source in raw:
        if source.cache_dir:
            raise ValueError(
                f"data source {source.name or source.kind!r} sets both kind and cache_dir. "
                "That pair means 'prepare encodes this source into that directory' and is "
                "ambiguous for training: drop cache_dir to train on pixels, or drop kind "
                "and the index fields to train on the cache"
            )
    return raw, encoded


def build_train_dataloader(
    request: TrainRequest,
    *,
    codec: EncodedBatchCodec,
    collate_encoded: Callable[[list[dict[str, Any]]], dict[str, Any]],
    make_raw_collate: Callable[[], Any],
    preprocess_fingerprint: dict[str, Any],
    force_mixed_encoded_cache: bool = False,
    open_mixed_encoded_cache: Callable[..., tuple[Any, dict[str, Any]]] = (
        open_mixed_cache_dataset
    ),
    open_indexed_encoded_cache: Callable[..., EncodedCacheDataset] = (open_encoded_cache_dataset),
) -> tuple[Any, dict[str, Any]]:
    """Build the resumable train loader plus the fingerprint that gates resume.

    Choosing between an offline encoded cache and raw online video, deriving the
    resume fingerprint, and wiring the sequence-parallel loader read the same way
    for every version. The two version-specific pieces -- which fields the cache
    carries and how raw samples are collated -- arrive as arguments.
    """
    state = get_parallel_state()
    loader_fingerprint = {
        "batch_size": request.batch_size,
        "num_workers": request.num_workers,
        "shuffle": request.shuffle,
        "drop_last": request.drop_last,
        "base_seed": request.base_seed,
        "sampler_seed": request.sampler_seed,
        "pin_memory": request.pin_memory,
        "persistent_workers": request.persistent_workers,
        "prefetch_factor": request.prefetch_factor,
    }
    mixed_loader = False
    raw_sources, sources = _split_sources(
        request.sources,
        latent_cache_mode=request.latent_cache_mode,
    )

    if raw_sources:
        if request.max_steps > 0:
            dataset, mix_fingerprint = open_mixed_raw_dataset(
                [_raw_source_mapping(source) for source in raw_sources],
                mode="train",
                mix_seed=request.mix_seed,
                steps_per_epoch=request.steps_per_epoch,
            )
            fingerprint = {
                **mix_fingerprint,
                **loader_fingerprint,
                "preprocess": dict(preprocess_fingerprint),
            }
            mixed_loader = True
        else:
            # Compose-only construction may not have the index files mounted.
            dataset = RawSampleDataset([])
            fingerprint = {
                "source_kind": "mixed_raw_video",
                "empty": True,
                "mix_seed": request.mix_seed,
                "steps_per_epoch": request.steps_per_epoch,
            }
    elif (
        request.latent_cache_mode == "offline"
        and sources
        and (len(sources) > 1 or force_mixed_encoded_cache)
    ):
        if request.max_steps > 0:
            dataset, mix_fingerprint = open_mixed_encoded_cache(
                [
                    {
                        "name": source.name,
                        "cache_dir": source.cache_dir,
                        "manifest": source.manifest,
                        "weight": source.weight,
                        "min_latents": source.min_latents,
                        "max_latents": source.max_latents,
                        "extras": dict(source.extras),
                    }
                    for source in sources
                ],
                codec=codec,
                mix_seed=request.mix_seed,
                steps_per_epoch=request.steps_per_epoch,
            )
            fingerprint = {**mix_fingerprint, **loader_fingerprint}
            mixed_loader = True
        else:
            # Compose-only construction may not have the caches mounted yet.
            dataset = EncodedCacheDataset(".", [], codec)
            fingerprint = {
                "source_kind": "mixed_encoded_cache",
                "empty": True,
                "mix_seed": request.mix_seed,
                "steps_per_epoch": request.steps_per_epoch,
            }
    elif request.latent_cache_mode == "offline":
        source = sources[0] if sources else None
        cache_dir = source.cache_dir if source is not None else ""
        cache_manifest = source.manifest if source is not None else ""
        manifest = read_cache_manifest(cache_dir, codec=codec) if cache_dir else None
        if manifest is None:
            if request.max_steps > 0:
                if not cache_dir:
                    raise ValueError(
                        "offline training requires a cache path in data.cache_dir or data.sources"
                    )
                dataset = _apply_source_bounds(
                    open_indexed_encoded_cache(
                        cache_dir,
                        cache_manifest,
                        codec=codec,
                    ),
                    source,
                )
                sample_ids = dataset.sample_ids
                # A scanned cache pins its ordering through sample_ids_digest
                # alone; only a supplied manifest adds a file to pin as well.
                manifest_fields = (
                    {
                        "manifest": cache_manifest,
                        "manifest_digest": manifest_file_digest(cache_manifest),
                    }
                    if cache_manifest
                    else {}
                )
                fingerprint = {
                    "source_kind": "encoded_cache",
                    "cache_dir": cache_dir,
                    "layout": "indexed",
                    "split": "train",
                    **manifest_fields,
                    "sample_ids_digest": digest_sample_ids(sample_ids),
                    "sample_count": len(sample_ids),
                    **loader_fingerprint,
                    **_source_bound_fields(source),
                }
            else:
                # Compose-only construction may not have a cache yet.
                dataset = EncodedCacheDataset(cache_dir or ".", [], codec)
                fingerprint = {
                    "source_kind": "encoded_cache",
                    "cache_dir": cache_dir,
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "empty": True,
                }
        else:
            if manifest.completed != len(manifest.sample_ids):
                raise ValueError(
                    "cache manifest is incomplete; run prepare again before "
                    "training or dataloader resume"
                )
            dataset = _apply_source_bounds(
                EncodedCacheDataset(cache_dir, manifest.sample_ids, codec),
                source,
            )
            fingerprint = {
                "source_kind": "encoded_cache",
                "cache_dir": cache_dir,
                "schema_version": manifest.schema_version,
                "bundle_version": manifest.bundle_version,
                "completed": manifest.completed,
                "encoding_fingerprint": manifest.encoding_fingerprint,
                "sample_ids_digest": (
                    manifest.sample_ids_digest or digest_sample_ids(manifest.sample_ids)
                ),
                "field_specs": manifest.field_specs or {},
                "sample_count": len(manifest.sample_ids),
                **loader_fingerprint,
                **_source_bound_fields(source),
            }
    elif not request.train_manifest:
        # Allow Hydra compose without a real manifest; fit will fail later.
        dataset = RawSampleDataset([])
        fingerprint = {
            "source_kind": "raw_video",
            "manifest": "",
            "empty": True,
        }
    else:
        raw = RawSampleDataset.from_manifest(request.train_manifest)
        dataset = raw
        fingerprint = {
            "source_kind": "raw_video",
            "manifest": request.train_manifest,
            "digest": raw.content_digest(),
            "sample_ids_digest": digest_sample_ids([sample.sample_id for sample in raw.samples]),
            "sample_count": len(raw),
            **loader_fingerprint,
            "preprocess": dict(preprocess_fingerprint),
        }

    collate_fn = collate_encoded if request.latent_cache_mode == "offline" else None
    if request.latent_cache_mode == "online" and request.pretrained_path:
        collate_fn = make_raw_collate()

    loader_kwargs = {
        "parallel_state": state,
        "batch_size": request.batch_size,
        "num_workers": request.num_workers,
        "drop_last": request.drop_last,
        "seed": request.sampler_seed,
        "shuffle": request.shuffle,
        "collate_fn": collate_fn,
        "pin_memory": request.pin_memory,
        "persistent_workers": request.persistent_workers,
        "prefetch_factor": request.prefetch_factor,
        "broadcast_batches": True,
    }
    if mixed_loader:
        loader = build_mixed_resumable_loader(dataset, **loader_kwargs)
    else:
        loader = build_resumable_loader(dataset, **loader_kwargs)
    batch_group_size = (
        int(state.config.usp_size)
        * int(getattr(state.config, "cp_size", 1))
        * int(getattr(state.config, "tp_size", 1))
    )
    if batch_group_size > 1:
        fingerprint = {
            **fingerprint,
            "batch_group_size": batch_group_size,
            "source": "leader_broadcast_loader",
        }
    return loader, fingerprint


__all__ = [
    "WanConditionSpec",
    "build_state_dict_mapper",
    "build_train_dataloader",
    "model_config_overrides",
    "parameter_storage_dtype",
    "parse_condition_streams",
    "read_condition_streams",
    "validate_shared",
]
