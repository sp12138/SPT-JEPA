"""Hydra entrypoint for offline latent cache preparation."""

from __future__ import annotations

import logging
from dataclasses import asdict, replace

import hydra
from omegaconf import DictConfig

from vforge.distributed.runtime import destroy_runtime, initialize_runtime
from vforge.hydra_config import prepare_cache_dir, resolved_config_hash, to_plain
from vforge.offline_encoder import EncodeComponents, run_encode
from vforge.tasks.common import build_and_snapshot
from vforge.tasks.registry import resolve_builder

logger = logging.getLogger(__name__)


def _output_invariant_model_extras(config) -> tuple[str, ...]:
    extras = to_plain(config)["model"]["extras"]
    keys = ["prepare_cpu_prefetch"]
    if int(extras.get("video_vae_tile_batch_size", 1)) == 1:
        keys.append("video_vae_tile_batch_size")
    return tuple(keys)


def _prepare_config_hashes(config) -> tuple[str, list[str]]:
    output_invariant_extras = _output_invariant_model_extras(config)
    hashes = set()
    for mask in range(1 << len(output_invariant_extras)):
        payload = to_plain(config)
        extras = payload["model"]["extras"]
        for index, key in enumerate(output_invariant_extras):
            if mask & (1 << index):
                extras.pop(key, None)
        hashes.add(resolved_config_hash(payload))

    semantic_payload = to_plain(config)
    semantic_extras = semantic_payload["model"]["extras"]
    for key in output_invariant_extras:
        semantic_extras.pop(key, None)
    semantic_hash = resolved_config_hash(semantic_payload)
    return semantic_hash, sorted(hashes)


def run_prepare(cfg: DictConfig) -> None:
    runtime = build_and_snapshot(cfg)
    cache_dir = prepare_cache_dir(runtime.config.data)
    logger.info(
        "resolved prepare config hash=%s model=%s cache_dir=%s",
        runtime.resolved_config_hash,
        runtime.config.model.name,
        cache_dir,
    )
    ctx = initialize_runtime(runtime.parallel, runtime.ops)
    try:
        build = resolve_builder(runtime.config.model.name, "encode")
        components = build(runtime.config)
        if not isinstance(components, EncodeComponents):
            raise TypeError(
                f"encode builder must return EncodeComponents, got {type(components).__name__}"
            )
        config_hash, compatible_config_hashes = _prepare_config_hashes(runtime.config)
        components = replace(
            components,
            encoding_fingerprint={
                **components.encoding_fingerprint,
                "resolved_config_hash": config_hash,
                "compatible_resolved_config_hashes": compatible_config_hashes,
                "ops_policy": asdict(runtime.ops),
                "resolved_backend": ctx.resolved_backend,
                "limitations": asdict(runtime.config.limitations),
            },
        )
        manifest = run_encode(components, device=ctx.device)
        logger.info(
            "prepare finished completed=%s cache_dir=%s",
            manifest.completed,
            cache_dir,
        )
    finally:
        destroy_runtime()


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="prepare",
)
def main(cfg: DictConfig) -> None:
    run_prepare(cfg)


if __name__ == "__main__":
    main()
