"""Wan 2.2 TI2V offline encoding: the shared recipe with TI2V's codec."""

from __future__ import annotations

import logging

from vforge.hydra_config import StandaloneConfig
from vforge.offline_encoder import EncodeComponents
from vforge.tasks.wan.encode import WanEncodeSpec, assemble_encode_components
from vforge.tasks.wan.v22.ti2v_5b.shared import (
    CACHE_FLAVOR,
    _build_condition_encoder,
    _condition_streams,
    _encoder_codec,
    _validate,
)

logger = logging.getLogger(__name__)


def build_encode_components(cfg: StandaloneConfig) -> EncodeComponents:
    # Still on StandaloneConfig: this verb gets its own request type when
    # the runner moves out of the entry layer.
    extras = dict(cfg.model.extras or {})
    _validate(
        extras=extras,
        latent_cache_mode=cfg.latent_cache.mode,
        usp_size=cfg.parallel.usp_size,
    )
    return assemble_encode_components(
        cfg,
        extras=extras,
        spec=WanEncodeSpec(
            label="Wan22",
            # Which condition streams are active decides the bundle's fields,
            # so the codec is per-config rather than a module constant.
            codec=_encoder_codec(extras),
            build_condition_encoder=_build_condition_encoder,
            fingerprint_extras={
                "cache_flavor": CACHE_FLAVOR,
                # TI2V can take its VAE from a different checkpoint than the
                # transformer, and that changes every latent in the cache.
                "vae_pretrained_path": extras.get("vae_pretrained_path")
                or cfg.model.pretrained_path,
                "condition_streams": list(_condition_streams(extras)),
            },
        ),
    )


__all__ = ["build_encode_components"]
