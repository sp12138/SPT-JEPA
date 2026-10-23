"""Task-owned joint ViT/DiT EMA, using Native sharded checkpoint state."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch

from vforge.distributed.parallelize import unwrap_replicated_data_parallel
from vforge.training.checkpoint import load_model_component_weights
from vforge.training.ema import ShardedEmaState


@dataclass(frozen=True)
class EmaConfig:
    enabled: bool = False
    decay: float = 0.99
    evaluation_weights: str = "online"

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("ema.enabled must be a boolean")
        if isinstance(self.decay, bool) or not math.isfinite(self.decay) or not 0 <= self.decay < 1:
            raise ValueError("ema.decay must be finite and in [0, 1)")
        if self.evaluation_weights not in {"online", "ema"}:
            raise ValueError("ema.evaluation_weights must be online or ema")
        if self.evaluation_weights == "ema" and not self.enabled:
            raise ValueError("EMA evaluation requires ema.enabled=true")


def initialize_ema(model, config):
    """Called by the optimizer factory after materialization, before full resume."""
    root = unwrap_replicated_data_parallel(model)
    if hasattr(root, "ema"):
        raise RuntimeError("RepWorld EMA has already been initialized")
    ema = ShardedEmaState(root, decay=config.decay, start_step=0)
    # Shadows already copy step-zero weights. Unlike lazy distillation EMA, the
    # first optimizer update applies the configured decay to that initial state.
    ema.initialized.fill_(True)
    device = next(root.parameters()).device
    for name in ("initialized", "num_updates", "parameter_names_utf8"):
        setattr(ema, name, getattr(ema, name).to(device))
    ema.register_buffer(
        "decay_value", torch.tensor(config.decay, dtype=torch.float64, device=device)
    )
    root.add_module("ema", ema)
    return ema


def validate_ema(root, config):
    ema = root.ema
    names = tuple(json.loads(bytes(ema.parameter_names_utf8.cpu().tolist()).decode("utf-8")))
    if names != ema.parameter_names:
        raise ValueError("Checkpoint EMA parameter order does not match the model")
    if float(ema.decay_value.item()) != config.decay:
        raise ValueError("Checkpoint EMA decay differs from the requested decay")
    if not bool(ema.initialized.item()):
        raise ValueError("Checkpoint EMA was not initialized")
    return ema


def update_ema(model, step, context, output, *, config):
    del context, output
    root = unwrap_replicated_data_parallel(model)
    ema = validate_ema(root, config)
    if int(ema.num_updates.item()) != step:
        raise ValueError("EMA update count does not match the completed optimizer steps")
    ema.update(root, step)
    return {"ema_updates": int(ema.num_updates.item()), "ema_decay": config.decay}


def load_evaluation_ema(model, config, source):
    """Load the EMA subtree with the same Native DCP loader as online weights.

    The eval runner has already strictly loaded both online model components.
    This initializes the matching sharded EMA destination, then strictly loads
    only that subtree before selecting its weights for inference.
    """
    initialize_ema(model, config)
    manifest = load_model_component_weights(
        model,
        source.path,
        component_path="ema",
        format=source.format,
        integrity=source.integrity,
        strict=True,
    )
    ema = validate_ema(model, config)
    if int(ema.num_updates.item()) <= 0:
        raise ValueError("EMA evaluation requires at least one optimizer update")
    if manifest is not None and int(ema.num_updates.item()) != manifest.completed_steps:
        raise ValueError("Checkpoint EMA count differs from the checkpoint step")
    ema.copy_to(model)
    return int(ema.num_updates.item())


def initialize_tuning_from_ema(model, optimizer, config, completed_steps):
    """Select the boundary EMA and clear optimizer moments, preserving LR and cursor."""
    root = unwrap_replicated_data_parallel(model)
    ema = validate_ema(root, config)
    if int(ema.num_updates.item()) != completed_steps:
        raise ValueError("Tuning EMA must match the phase boundary's completed steps")
    ema.copy_to(root)
    optimizer.state.clear()
    return {
        "tuning_init_ema": 1.0,
        "tuning_optimizer_reset": 1.0,
        "tuning_source_ema_updates": float(completed_steps),
    }
