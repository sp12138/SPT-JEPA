# Copyright 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint lifecycle wrapper for the standalone MiniMax-H3 transformer."""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vforge.models.loading import cast_module_parameters

from .transformer import MiniMaxH3Transformer3DModel


class MiniMaxH3Model(nn.Module):
    """Checkpoint-wrapper-friendly standalone MiniMax-H3 FL2VA transformer."""

    def __init__(self, transformer: MiniMaxH3Transformer3DModel) -> None:
        super().__init__()
        self.transformer = transformer

    @property
    def blocks(self):
        return self.transformer.transformer_blocks

    @property
    def config(self):
        return self.transformer.config

    @property
    def dtype(self) -> torch.dtype:
        return next(self.transformer.transformer_blocks.parameters()).dtype

    @property
    def _no_split_modules(self):
        return self.transformer._no_split_modules

    @property
    def _gradient_checkpointing_func(self):
        return self.transformer._gradient_checkpointing_func

    @_gradient_checkpointing_func.setter
    def _gradient_checkpointing_func(self, value) -> None:
        self.transformer._gradient_checkpointing_func = value

    def cache_context(self, name: str):
        context = getattr(self.transformer, "cache_context", None)
        return context(name) if callable(context) else nullcontext()

    def fsdp_execution_roots(self) -> tuple[nn.Module, ...]:
        """Make the parameter-owning transformer the explicit FSDP root."""
        return (self.transformer,)

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Expose activation checkpointing at the model boundary."""
        self.transformer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    @classmethod
    def build_empty_from_pretrained(
        cls,
        path: str | Path,
        *,
        parameter_dtype: torch.dtype = torch.bfloat16,
        **config_kwargs: Any,
    ) -> MiniMaxH3Model:
        loaded_config = dict(
            MiniMaxH3Transformer3DModel.load_config(
                path,
                subfolder="transformer",
            )
        )
        valid_names = set(inspect.signature(MiniMaxH3Transformer3DModel.__init__).parameters) - {
            "self"
        }
        loaded_config = {
            name: value for name, value in loaded_config.items() if name in valid_names
        }
        loaded_config.update(config_kwargs)
        with torch.device("meta"):
            transformer = MiniMaxH3Transformer3DModel(**loaded_config)
        cast_module_parameters(transformer, parameter_dtype)
        return cls(transformer)

    def forward(self, *args: Any, **kwargs: Any):
        return self.transformer(*args, **kwargs)


__all__ = ["MiniMaxH3Model"]
