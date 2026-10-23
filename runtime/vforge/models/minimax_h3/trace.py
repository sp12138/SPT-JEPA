"""Lightweight tensor traces for MiniMax-H3 inference attribution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from vforge.utils.comparison import tensor_sha256
from vforge.utils.tensor import to_local_tensor


class H3TraceRecorder:
    """Append rank-zero tensor checksums and moments to JSONL."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path and int(os.environ.get("RANK", "0")) == 0 else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self.path is not None

    @torch.no_grad()
    def tensor(
        self,
        name: str,
        value: torch.Tensor,
        *,
        step: int | None = None,
    ) -> None:
        if self.path is None:
            return
        local = to_local_tensor(value).detach()
        finite = torch.isfinite(local)
        finite_values = local[finite].float()
        payload: dict[str, Any] = {
            "dtype": str(local.dtype),
            "finite": int(finite.sum().item()),
            "max_abs": (
                float(finite_values.abs().max().item()) if finite_values.numel() else float("nan")
            ),
            "name": name,
            "numel": local.numel(),
            "sha256": tensor_sha256(local),
            "shape": list(local.shape),
            "square_sum": (
                float(finite_values.square().sum(dtype=torch.float64).item())
                if finite_values.numel()
                else float("nan")
            ),
            "step": step,
            "sum": (
                float(finite_values.sum(dtype=torch.float64).item())
                if finite_values.numel()
                else float("nan")
            ),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def scalar(self, name: str, value: Any, *, step: int | None = None) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"name": name, "step": step, "value": value},
                    sort_keys=True,
                )
                + "\n"
            )


__all__ = ["H3TraceRecorder"]
