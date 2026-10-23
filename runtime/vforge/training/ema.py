"""Sharding-preserving exponential moving average state."""

from __future__ import annotations

import json
from collections.abc import Iterator

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, distribute_tensor


class ShardedEmaState(nn.Module):
    """FP32 EMA shadows that preserve each parameter's current tensor layout.

    Construct this after FSDP2 materialization and pretrained-weight loading.
    A DTensor parameter produces a DTensor shadow with the same placements, so
    every rank stores only its local shard. Registering shadows as persistent
    buffers makes the EMA part of the model DCP without gathering a full model.

    The state exists from step zero to keep checkpoint structure stable, but the
    first eligible generator update copies current weights exactly. This matches
    Taichu's lazy EMA construction at ``start_step`` without changing the DCP
    model fingerprint midway through a run.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        decay: float,
        start_step: int,
    ) -> None:
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        if start_step < 0:
            raise ValueError("EMA start_step must be non-negative")
        parameters = tuple(module.named_parameters())
        if not parameters:
            raise ValueError("EMA target must contain at least one parameter")
        if any(parameter.is_meta for _name, parameter in parameters):
            raise ValueError("EMA target must be materialized before shadow creation")

        self.decay = float(decay)
        self.start_step = int(start_step)
        self.parameter_names = tuple(name for name, _parameter in parameters)
        self.register_buffer("initialized", torch.tensor(False), persistent=True)
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.int64), persistent=True)
        encoded_names = json.dumps(
            self.parameter_names,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.register_buffer(
            "parameter_names_utf8",
            torch.tensor(list(encoded_names), dtype=torch.uint8),
            persistent=True,
        )
        for index, (_name, parameter) in enumerate(parameters):
            self.register_buffer(
                self._shadow_name(index),
                parameter.detach().to(dtype=torch.float32).clone(),
                persistent=True,
            )

    def update(self, module: nn.Module, global_step: int) -> bool:
        """Update after one generator optimizer step.

        Returns ``True`` when EMA state changed and ``False`` before
        ``start_step``. On the first eligible step the state is initialized by
        copying current weights and does not apply decay.
        """

        if global_step < self.start_step:
            return False
        parameters = self._validated_parameters(module)
        with torch.no_grad():
            if not bool(self.initialized.item()):
                for shadow, parameter in zip(self.shadows(), parameters, strict=True):
                    shadow.copy_(parameter.detach().to(dtype=torch.float32))
                self.initialized.fill_(True)
            else:
                for shadow, parameter in zip(self.shadows(), parameters, strict=True):
                    shadow.mul_(self.decay).add_(
                        parameter.detach().to(dtype=torch.float32),
                        alpha=1.0 - self.decay,
                    )
            self.num_updates.add_(1)
        return True

    def copy_to(self, module: nn.Module) -> None:
        """Copy initialized EMA weights to a compatible generator module."""

        if not bool(self.initialized.item()):
            raise RuntimeError("EMA state has not reached start_step")
        parameters = self._validated_parameters(module)
        with torch.no_grad():
            for parameter, shadow in zip(parameters, self.shadows(), strict=True):
                source = shadow.to(dtype=parameter.dtype)
                if isinstance(parameter, DTensor):
                    if isinstance(source, DTensor):
                        source = source.redistribute(
                            device_mesh=parameter.device_mesh,
                            placements=parameter.placements,
                        )
                    else:
                        source = distribute_tensor(
                            source,
                            device_mesh=parameter.device_mesh,
                            placements=parameter.placements,
                        )
                elif isinstance(source, DTensor):
                    # Frozen reference models are replicated plain tensors while
                    # the trainable student's EMA preserves its FSDP2 shards.
                    # Every rank enters this collective in parameter order.
                    source = source.full_tensor()
                parameter.copy_(source)

    def shadows(self) -> Iterator[torch.Tensor]:
        """Yield shadows in the deterministic ``named_parameters`` order."""

        for index in range(len(self.parameter_names)):
            yield self.get_buffer(self._shadow_name(index))

    def _validated_parameters(self, module: nn.Module) -> tuple[nn.Parameter, ...]:
        parameters = tuple(module.named_parameters())
        names = tuple(name for name, _parameter in parameters)
        if names != self.parameter_names:
            raise ValueError(
                "EMA target parameter names changed; "
                f"expected={self.parameter_names!r}, got={names!r}"
            )
        values = tuple(parameter for _name, parameter in parameters)
        for name, parameter, shadow in zip(
            names,
            values,
            self.shadows(),
            strict=True,
        ):
            if tuple(parameter.shape) != tuple(shadow.shape):
                raise ValueError(
                    f"EMA parameter shape changed for {name!r}: "
                    f"expected={tuple(shadow.shape)}, got={tuple(parameter.shape)}"
                )
        return values

    @staticmethod
    def _shadow_name(index: int) -> str:
        return f"shadow_{index:05d}"


__all__ = ["ShardedEmaState"]
