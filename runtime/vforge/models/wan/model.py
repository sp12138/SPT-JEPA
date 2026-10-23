"""Checkpoint-stable model roots shared by Wan variants."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from vforge.models.factory import patch_diffusers_getattr
from vforge.models.loading import cast_module_parameters

_ModuleT = TypeVar("_ModuleT", bound=nn.Module)


@dataclass(frozen=True)
class WanChannelLayout:
    """Published and configured patch-embedding widths for one transformer."""

    z_dim: int
    pretrained_in_channels: int
    in_channels: int

    def __post_init__(self) -> None:
        for name, value in (
            ("z_dim", self.z_dim),
            ("pretrained_in_channels", self.pretrained_in_channels),
            ("in_channels", self.in_channels),
        ):
            if int(value) <= 0:
                raise ValueError(f"Wan {name} must be positive; got {value}")

    @property
    def is_widened(self) -> bool:
        return self.in_channels > self.pretrained_in_channels


def read_transformer_config(
    transformer_type: type,
    path: str | Path,
    *,
    subfolder: str = "transformer",
) -> dict[str, Any]:
    """The published Diffusers config for one transformer.

    This is the only place a checkpoint is touched for structure, and callers
    are expected to read once and pass the result around: everything downstream
    -- the latent width, the widened config, the channel layout -- is a pure
    function of what comes back here.
    """
    if Path(path).suffix.lower() in {".pt", ".pth"}:
        raise ValueError(
            "state-dict initialization requires explicit Wan config; use a "
            "Diffusers pretrained directory for standalone materialization"
        )
    return dict(transformer_type.load_config(str(path), subfolder=subfolder))


def published_z_dim(config: Mapping[str, Any]) -> int:
    """Latent width a published transformer denoises.

    ``out_channels`` is the reliable source: a checkpoint already widened for
    some condition mode carries a larger ``in_channels`` but still denoises the
    same latent. The fallback covers configs that omit it, where the two agree
    by definition.
    """
    return int(config.get("out_channels") or config["in_channels"])


def build_meta_module(
    module_type: type[_ModuleT],
    config: dict[str, Any],
    *,
    parameter_dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> _ModuleT:
    """Build only module structure, optionally fixing Parameter storage dtype.

    ``parameter_dtype`` is the default for floating parameters.
    ``_keep_in_fp32_modules`` stay float32, matching Diffusers Wan.
    """
    with torch.device("meta"):
        module = module_type(**kwargs, **config)
    if parameter_dtype is not None:
        cast_module_parameters(module, parameter_dtype)
    return module


class WanModelRootBase(nn.Module):
    """Storage-neutral framework interface for a Wan training root.

    This base registers no child modules.  Concrete roots decide whether their
    checkpoint key space starts with ``transformer.``, ``wan.``, or another
    variant-owned prefix.
    """

    def primary_transformer(self) -> nn.Module:
        """Transformer whose config and dtype describe the root."""
        raise NotImplementedError

    def cache_transformers(self) -> tuple[nn.Module, ...]:
        """Transformers whose named inference caches span one sampling loop."""
        return (self.primary_transformer(),)

    def fsdp_execution_roots(self) -> tuple[nn.Module, ...]:
        """Disjoint transformer trees that may be sharded independently."""
        return self.cache_transformers()

    @property
    def config(self):
        return self.primary_transformer().config

    @property
    def dtype(self) -> torch.dtype:
        transformer = self.primary_transformer()
        declared = getattr(transformer, "dtype", None)
        if isinstance(declared, torch.dtype):
            return declared
        parameter = next(transformer.parameters(), None)
        if parameter is None:
            raise RuntimeError(f"{type(transformer).__name__} must declare dtype or own parameters")
        return parameter.dtype

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    ) -> None:
        checkpoint_kwargs = (
            {"use_reentrant": False}
            if gradient_checkpointing_kwargs is None
            else dict(gradient_checkpointing_kwargs)
        )
        checkpoint_func = partial(checkpoint, **checkpoint_kwargs)
        for transformer in self.fsdp_execution_roots():
            transformer.enable_gradient_checkpointing(gradient_checkpointing_func=checkpoint_func)

    @contextmanager
    def cache_context(self, name: str):
        with ExitStack() as stack:
            for transformer in self.cache_transformers():
                context = getattr(transformer, "cache_context", None)
                if callable(context):
                    stack.enter_context(context(name))
            yield


class WanSingleTransformerModel(WanModelRootBase):
    """Wan root whose checkpoint owns exactly one ``transformer`` subtree."""

    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer
        patch_diffusers_getattr(self.transformer)

    def primary_transformer(self) -> nn.Module:
        return self.transformer

    @property
    def blocks(self):
        return self.transformer.blocks

    @property
    def _no_split_modules(self):
        return getattr(self.transformer, "_no_split_modules", None)

    @property
    def _gradient_checkpointing_func(self):
        return self.transformer._gradient_checkpointing_func

    @_gradient_checkpointing_func.setter
    def _gradient_checkpointing_func(self, value) -> None:
        self.transformer._gradient_checkpointing_func = value

    def forward(self, *args: Any, **kwargs: Any):
        return self.transformer(*args, **kwargs)


__all__ = [
    "WanChannelLayout",
    "WanModelRootBase",
    "WanSingleTransformerModel",
    "build_meta_module",
    "published_z_dim",
    "read_transformer_config",
]
