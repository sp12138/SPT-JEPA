"""Wan 2.1 offline encoding: the shared recipe with 2.1's codec and encoder."""

from __future__ import annotations

import logging

from vforge.hydra_config import StandaloneConfig
from vforge.offline_encoder import EncodeComponents
from vforge.tasks.wan.encode import WanEncodeSpec, assemble_encode_components
from vforge.tasks.wan.shared import validate_shared
from vforge.tasks.wan.v21.shared import (
    WAN21_ENCODED_BATCH_CODEC,
    _build_condition_encoder,
    _condition_streams,
)

logger = logging.getLogger(__name__)


def build_encode_components(cfg: StandaloneConfig) -> EncodeComponents:
    # Still on StandaloneConfig: this verb gets its own request type when
    # the runner moves out of the entry layer.
    validate_shared(
        latent_cache_mode=cfg.latent_cache.mode,
        usp_size=cfg.parallel.usp_size,
    )
    extras = dict(cfg.model.extras or {})
    return assemble_encode_components(
        cfg,
        extras=extras,
        spec=WanEncodeSpec(
            label="Wan21",
            codec=WAN21_ENCODED_BATCH_CODEC,
            build_condition_encoder=_build_condition_encoder,
            # The image-branch toggles change which latents land in a bundle,
            # so a cache written with one setting cannot serve another.
            fingerprint_extras={
                "condition_streams": list(_condition_streams(extras)),
                "use_vae_condition": extras.get("use_vae_condition", True),
                "use_clip_condition": extras.get("use_clip_condition", False),
                "use_dino_condition": extras.get("use_dino_condition", False),
            },
        ),
    )


__all__ = ["build_encode_components"]
