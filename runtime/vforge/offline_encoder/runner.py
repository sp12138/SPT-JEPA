"""Run a model's offline encoding pass and publish the cache manifest.

This is the ``encode`` verb's execution engine, kept out of the Hydra entry so
the component type lives next to the code that consumes it, the same way
``TrainComponents`` sits with the trainer and ``EvalComponents`` with the
evaluation runner.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from vforge.data.cache import (
    CACHE_SCHEMA_VERSION,
    CacheManifest,
    EncodedBatchCodec,
    cache_path_for_sample,
    digest_sample_ids,
    read_cache_manifest,
    save_latent_bundle,
    write_cache_manifest,
)
from vforge.distributed.parallel_state import get_parallel_state


@dataclass(frozen=True)
class EncodeComponents:
    dataset: Any
    sample_ids: tuple[str, ...]
    encode_condition: Callable[[Any], dict[str, Any]]
    cache_dir: str
    encoding_fingerprint: dict[str, Any]
    codec: EncodedBatchCodec
    skip_existing: bool = True
    prepare_condition: Callable[[Any], Any] | None = None


def _fingerprints_match(saved: dict[str, Any], current: dict[str, Any]) -> bool:
    saved = dict(saved)
    current = dict(current)
    compatible_hashes = set(current.pop("compatible_resolved_config_hashes", ()))
    saved.pop("compatible_resolved_config_hashes", None)
    saved_hash = saved.pop("resolved_config_hash", None)
    current_hash = current.pop("resolved_config_hash", None)
    return saved == current and (saved_hash == current_hash or saved_hash in compatible_hashes)


def run_encode(
    components: EncodeComponents,
    *,
    device: torch.device,
) -> CacheManifest:
    """Materialize a model plugin's deterministic encoded-cache components."""
    state = get_parallel_state()
    sample_ids = list(components.sample_ids)
    if len(sample_ids) != len(components.dataset):
        raise ValueError("encode sample_ids must align one-to-one with dataset")
    reusable_manifest = None
    if components.skip_existing:
        try:
            reusable_manifest = read_cache_manifest(
                components.cache_dir,
                codec=components.codec,
            )
        except ValueError:
            reusable_manifest = None
    can_reuse = bool(
        reusable_manifest is not None
        and reusable_manifest.sample_ids == sample_ids
        and reusable_manifest.completed == len(sample_ids)
        and _fingerprints_match(
            reusable_manifest.encoding_fingerprint,
            components.encoding_fingerprint,
        )
        and (not sample_ids or bool(reusable_manifest.field_specs))
    )
    if state.world_size > 1:
        reuse = torch.tensor([int(can_reuse)], dtype=torch.long, device=device)
        dist.all_reduce(reuse, op=dist.ReduceOp.MIN)
        can_reuse = bool(reuse.item())
    if can_reuse:
        assert reusable_manifest is not None
        return reusable_manifest

    completed = 0
    local_specs = None
    local_indices = [
        index
        for index in range(len(sample_ids))
        if index % int(state.world_size) == state.global_rank
    ]

    def encode_index(index: int, condition: Any) -> None:
        nonlocal completed, local_specs
        sample_id = sample_ids[index]
        path = cache_path_for_sample(components.cache_dir, sample_id)
        bundle = components.codec.canonicalize(components.encode_condition(condition))
        specs = components.codec.field_specs(bundle)
        if local_specs is None:
            local_specs = specs
        elif specs != local_specs:
            raise ValueError(
                "prepared cache contains inconsistent field shape/dtype: "
                f"sample_id={sample_id!r}, expected={local_specs!r}, actual={specs!r}"
            )
        save_latent_bundle(bundle, path)
        completed += 1

    if components.prepare_condition is None:
        for index in local_indices:
            encode_index(index, components.dataset[index])
    elif local_indices:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="encode-prefetch") as pool:
            future = pool.submit(components.prepare_condition, components.dataset[local_indices[0]])
            for offset, index in enumerate(local_indices):
                condition = future.result()
                if offset + 1 < len(local_indices):
                    next_index = local_indices[offset + 1]
                    future = pool.submit(
                        components.prepare_condition,
                        components.dataset[next_index],
                    )
                encode_index(index, condition)

    if state.world_size > 1:
        total = torch.tensor([completed], dtype=torch.long, device=device)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        completed = int(total.item())
    if completed != len(sample_ids):
        raise RuntimeError(
            "encode did not complete every expected sample: "
            f"completed={completed}, expected={len(sample_ids)}"
        )

    payload: list[CacheManifest | None] = [None]
    gathered_specs = None
    if state.world_size > 1:
        gathered_specs = [None] * int(state.world_size) if state.global_rank == 0 else None
        dist.gather_object(local_specs, gathered_specs, dst=0)
    if state.global_rank == 0:
        expected_specs = None
        for specs in gathered_specs or [local_specs]:
            if specs is None:
                continue
            if expected_specs is None:
                expected_specs = specs
            elif specs != expected_specs:
                raise ValueError(
                    "prepared cache contains inconsistent field shape/dtype: "
                    f"expected={expected_specs!r}, actual={specs!r}"
                )
        payload[0] = CacheManifest(
            schema_version=CACHE_SCHEMA_VERSION,
            sample_ids=sample_ids,
            completed=len(sample_ids),
            encoding_fingerprint=components.encoding_fingerprint,
            bundle_version=components.codec.bundle_version,
            sample_ids_digest=digest_sample_ids(sample_ids),
            field_specs=expected_specs or {},
            codec_name=components.codec.name,
            codec_schema_version=components.codec.schema_version,
        )
        write_cache_manifest(components.cache_dir, payload[0])
    if state.world_size > 1:
        dist.broadcast_object_list(payload, src=0)
    assert payload[0] is not None
    return payload[0]


__all__ = ["EncodeComponents", "run_encode"]
