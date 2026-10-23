"""Training random streams, process RNG state, and seeding."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


class RandomStreams:
    """Own a run's named generators and snapshot them with the process RNGs."""

    def __init__(self) -> None:
        self._generators: dict[str, torch.Generator] = {}

    def generator(self, name: str, *, seed: int, device: torch.device) -> torch.Generator:
        """Seed a stream on first use, then keep consuming the same generator."""
        if name not in self._generators:
            self._generators[name] = torch.Generator(device=device).manual_seed(seed)
        return self._generators[name]

    def state_dict(self) -> dict[str, Any]:
        """Capture global RNGs and every named generator at the same boundary."""
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "streams": {
                name: {"device": generator.device.type, "state": generator.get_state()}
                for name, generator in self._generators.items()
            },
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore global RNGs and named generators before training continues."""
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state:
            if not torch.cuda.is_available():
                raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
            torch.cuda.set_rng_state(state["torch_cuda"])
        # CUDA streams restore onto the current rank's device, not the saved ordinal.
        streams = state["streams"]
        for name in self._generators.keys() - streams.keys():
            del self._generators[name]
        for name, saved in streams.items():
            device = torch.device(saved["device"])
            if device.type == "cuda":
                device = torch.device("cuda", torch.cuda.current_device())
            generator = self.generator(name, seed=0, device=device)
            generator.set_state(saved["state"])


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and torch for scripts and parity experiments."""
    seed_process(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False


def seed_process(seed: int) -> None:
    """Seed all process-local application RNGs."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "RandomStreams",
    "seed_everything",
    "seed_process",
]
