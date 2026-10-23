"""Native vforge wrapper around the Wan 2.1 transformer."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from vforge.models.wan.model import (
    WanSingleTransformerModel,
    build_meta_module,
    read_transformer_config,
)


class Wan21Model(WanSingleTransformerModel):
    """Checkpoint-wrapper-friendly native Wan 2.1 transformer."""

    @classmethod
    def build_empty(
        cls,
        config: Mapping[str, Any],
        *,
        parameter_dtype: torch.dtype | None = None,
    ) -> Wan21Model:
        """Build only model structure from an already-resolved config."""
        from vforge.models.wan.v21.transformer import WanTransformer3DModel

        transformer = build_meta_module(
            WanTransformer3DModel,
            dict(config),
            parameter_dtype=parameter_dtype,
        )
        return cls(transformer)

    @classmethod
    def build_empty_from_pretrained(
        cls,
        path: str | Path,
        *,
        parameter_dtype: torch.dtype | None = None,
        **config_kwargs: Any,
    ) -> Wan21Model:
        """Materialize only model structure; weights are loaded after FSDP."""
        # Kept local so importing the common root does not drag in this concrete
        # transformer; other plugins bring their own implementation.
        from vforge.models.wan.v21.transformer import WanTransformer3DModel

        config = read_transformer_config(WanTransformer3DModel, path)
        config.update(config_kwargs)
        return cls.build_empty(
            config,
            parameter_dtype=parameter_dtype,
        )


__all__ = ["Wan21Model"]
