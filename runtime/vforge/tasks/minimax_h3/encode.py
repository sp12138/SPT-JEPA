"""MiniMax-H3 FL2VA offline source encoding."""

from __future__ import annotations

from vforge.data.cache import BUNDLE_VERSION, CACHE_SCHEMA_VERSION, digest_sample_ids
from vforge.data.dataset import RawSampleDataset
from vforge.hydra_config import StandaloneConfig, prepare_cache_dir
from vforge.offline_encoder import EncodeComponents
from vforge.tasks.minimax_h3.shared import (
    H3_ENCODED_BATCH_CODEC,
    _torch_dtype,
    _validate_shared,
)


def build_encode_components(cfg: StandaloneConfig) -> EncodeComponents:
    _validate_shared(
        extras=cfg.model.extras,
        latent_cache_mode=cfg.latent_cache.mode,
        usp_size=cfg.parallel.usp_size,
    )
    if cfg.latent_cache.mode != "online":
        raise ValueError("MiniMax-H3 prepare requires latent_cache.mode=online")
    if not cfg.data.train_manifest:
        raise ValueError("H3 prepare requires data.train_manifest")
    cache_dir = prepare_cache_dir(cfg.data)

    from vforge.distributed.runtime import get_runtime
    from vforge.models.minimax_h3.encoder import (
        MiniMaxH3ConditionEncoder,
        resolve_qwen_patch_embed_mode,
    )

    runtime = get_runtime()
    if runtime is None:
        raise RuntimeError("H3 prepare requires an initialized distributed runtime")
    dataset = RawSampleDataset.from_manifest(cfg.data.train_manifest)
    encoder = MiniMaxH3ConditionEncoder.from_pretrained(
        cfg.model.pretrained_path,
        extras=cfg.model.extras,
        device=runtime.device,
        dtype=_torch_dtype(cfg.runtime.precision),
    )
    sample_ids = [sample.sample_id for sample in dataset.samples]
    cpu_prefetch = bool(cfg.model.extras.get("prepare_cpu_prefetch", True))

    return EncodeComponents(
        dataset=dataset,
        sample_ids=tuple(sample_ids),
        encode_condition=encoder.encode_prepared if cpu_prefetch else encoder.encode_sample,
        prepare_condition=encoder.prepare_sample if cpu_prefetch else None,
        cache_dir=cache_dir,
        encoding_fingerprint={
            "model": "minimax_h3",
            "variant": "fl2va",
            "pretrained_path": cfg.model.pretrained_path,
            "model_revision": cfg.model.extras.get("model_revision", ""),
            "qwen_patch_embed_mode": resolve_qwen_patch_embed_mode(cfg.model.extras),
            "manifest_digest": dataset.content_digest(),
            "sample_ids_digest": digest_sample_ids(sample_ids),
            "sample_count": len(sample_ids),
            "cache_schema": CACHE_SCHEMA_VERSION,
            "bundle_version": BUNDLE_VERSION,
            "codec_name": H3_ENCODED_BATCH_CODEC.name,
            "codec_schema_version": H3_ENCODED_BATCH_CODEC.schema_version,
        },
        codec=H3_ENCODED_BATCH_CODEC,
        skip_existing=bool(cfg.model.extras.get("skip_existing", True)),
    )


__all__ = ["build_encode_components"]
