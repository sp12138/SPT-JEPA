"""The offline encoding recipe every Wan version shares.

All three versions encode the same way: read a raw manifest, build the
version's condition encoder, run each sample through it, and canonicalize the
result with the version's codec. What differs is the codec, the encoder, and a
handful of cache-key entries -- so those arrive as a :class:`WanEncodeSpec` and
everything else lives here once.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from vforge.data.cache import (
    BUNDLE_VERSION,
    CACHE_SCHEMA_VERSION,
    EncodedBatchCodec,
    digest_sample_ids,
)
from vforge.data.collate import build_raw_collate_fn
from vforge.data.dataset import RawSampleDataset
from vforge.hydra_config import StandaloneConfig, prepare_cache_dir
from vforge.offline_encoder import EncodeComponents

logger = logging.getLogger(__name__)


class _ConditionEncoderBuilder(Protocol):
    def __call__(
        self,
        *,
        extras: Mapping[str, Any],
        pretrained_path: str,
        precision: str,
        require_assets: bool,
    ) -> Any: ...


@dataclass(frozen=True)
class WanEncodeSpec:
    """What one Wan version contributes to the shared encode recipe."""

    label: str
    """Version name for error messages, e.g. ``"Wan22 A14B"``."""

    codec: EncodedBatchCodec
    """Schema the encoder bundle is projected onto before it is written."""

    build_condition_encoder: _ConditionEncoderBuilder

    fingerprint_extras: Mapping[str, Any] = field(default_factory=dict)
    """Cache-key entries beyond the version-neutral ones.

    These decide when an existing cache is stale, so a version may only add
    keys that actually change its bundles. Adding or dropping one invalidates
    every cache written by that version.
    """


def assemble_encode_components(
    cfg: StandaloneConfig,
    *,
    extras: Mapping[str, Any],
    spec: WanEncodeSpec,
) -> EncodeComponents:
    """Assemble the encode verb from ``spec``.

    The caller has already run its own validation; what is checked here is only
    what every version checks the same way.
    """
    if cfg.latent_cache.mode != "online":
        raise ValueError(
            f"{spec.label} prepare requires latent_cache.mode=online, got {cfg.latent_cache.mode!r}"
        )
    if not cfg.data.train_manifest:
        raise ValueError("data.train_manifest is required for prepare")
    cache_dir = prepare_cache_dir(cfg.data)
    dataset = RawSampleDataset.from_manifest(cfg.data.train_manifest)
    encoder = spec.build_condition_encoder(
        extras=extras,
        pretrained_path=cfg.model.pretrained_path,
        precision=cfg.runtime.precision,
        require_assets=True,
    )
    collate_fn = build_raw_collate_fn(
        extras=extras,
        pretrained_path=cfg.model.pretrained_path,
    )

    def encode_condition(sample_or_batch: dict[str, Any]) -> dict[str, Any]:
        # Accept either a single raw sample dict or an already-collated batch.
        if "video_frames" in sample_or_batch:
            batch = sample_or_batch
        else:
            batch = collate_fn([sample_or_batch])
        return spec.codec.canonicalize(encoder.encode_batch(batch))

    return EncodeComponents(
        dataset=dataset,
        sample_ids=tuple(sample.sample_id for sample in dataset.samples),
        encode_condition=encode_condition,
        cache_dir=cache_dir,
        encoding_fingerprint=encoding_fingerprint(
            dataset,
            extras=extras,
            pretrained_path=cfg.model.pretrained_path,
            spec_extras=spec.fingerprint_extras,
        ),
        codec=spec.codec,
        skip_existing=bool(extras.get("skip_existing", True)),
    )


def encoding_fingerprint(
    dataset: RawSampleDataset,
    *,
    extras: Mapping[str, Any],
    pretrained_path: str,
    spec_extras: Mapping[str, Any],
) -> dict[str, Any]:
    """The cache key: everything that would make written bundles stale."""
    sample_ids = [sample.sample_id for sample in dataset.samples]
    return {
        "pretrained_path": pretrained_path,
        "model_revision": extras.get("model_revision") or pretrained_path,
        "manifest_digest": dataset.content_digest(),
        "sample_ids_digest": digest_sample_ids(sample_ids),
        "sample_count": len(dataset),
        "bundle_schema": CACHE_SCHEMA_VERSION,
        "bundle_version": BUNDLE_VERSION,
        "max_frames": extras.get("max_frames"),
        "max_sequence_length": extras.get("max_sequence_length", 512),
        "resolution": extras.get("resolution"),
        "fps": extras.get("fps"),
        **spec_extras,
    }


__all__ = ["WanEncodeSpec", "assemble_encode_components", "encoding_fingerprint"]
