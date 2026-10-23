"""Wan 2.2 A14B offline encoding: the shared recipe with A14B's codec."""

from __future__ import annotations

import logging

from vforge.hydra_config import StandaloneConfig
from vforge.offline_encoder import EncodeComponents
from vforge.tasks.wan.encode import WanEncodeSpec, assemble_encode_components
from vforge.tasks.wan.v22.a14b.shared import (
    CACHE_FLAVOR,
    WAN22_A14B_ENCODED_BATCH_CODEC,
    _build_condition_encoder,
    _condition_streams,
    _validate,
)

logger = logging.getLogger(__name__)


def build_encode_components(cfg: StandaloneConfig) -> EncodeComponents:
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
            label="Wan22 A14B",
            codec=WAN22_A14B_ENCODED_BATCH_CODEC,
            build_condition_encoder=_build_condition_encoder,
            fingerprint_extras={
                "condition_streams": list(_condition_streams(extras)),
                "cache_flavor": CACHE_FLAVOR,
            },
        ),
    )


__all__ = ["build_encode_components"]
