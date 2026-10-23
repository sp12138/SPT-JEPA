"""Neutral assembly of leader-prepared evaluation samples."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import torch

from vforge.data.loader import LeaderBroadcastDataLoader, build_eval_loader
from vforge.tasks.requests import EvalRequest


@dataclass(frozen=True)
class PreparedEvalShape:
    """Pixel-space output geometry associated with one encoded sample."""

    height: int
    width: int
    num_frames: int


@dataclass(frozen=True)
class PreparedEvalBatch:
    """Complete model-owned encoding result ready for leader broadcast."""

    encoded: Mapping[str, Any]
    shape: PreparedEvalShape
    context: Mapping[str, Any] = field(default_factory=dict)


PreparedEvalBatchPreparer: TypeAlias = Callable[
    [list[dict[str, Any]], str, torch.device],
    PreparedEvalBatch,
]


def build_prepared_eval_loader(
    request: EvalRequest,
    *,
    dataset: Any,
    parallel_state: Any,
    label: str,
    prepare_batch: PreparedEvalBatchPreparer,
) -> LeaderBroadcastDataLoader:
    """Build a leader-prepared loader with one explicit model-owned callback."""

    def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        if len(samples) != 1:
            raise ValueError(f"{label} evaluation requires batch_size=1, got {len(samples)}")
        sample = samples[0]

        from vforge.distributed.runtime import get_runtime

        runtime = get_runtime()
        if runtime is None:
            raise RuntimeError(f"{label} evaluation requires an initialized runtime")
        batch = prepare_batch(samples, request.data_source, runtime.device)
        shape = batch.shape

        prepared = {
            "sample_id": str(sample.get("sample_id") or "sample"),
            "__vforge_eval_index__": int(sample.get("__vforge_eval_index__", 0)),
            "__vforge_eval_seed_offset__": int(
                sample.get(
                    "__vforge_eval_seed_offset__",
                    sample.get("__vforge_eval_index__", 0),
                )
            ),
            "__vforge_eval_padding__": bool(sample.get("__vforge_eval_padding__", False)),
            "__encoded__": dict(batch.encoded),
            "__height__": shape.height,
            "__width__": shape.width,
            "__num_frames__": shape.num_frames,
        }
        collisions = sorted(set(prepared).intersection(batch.context))
        if collisions:
            raise ValueError(
                f"{label} eval sample context overwrites reserved fields: {collisions}"
            )
        prepared.update(batch.context)
        return prepared

    return build_eval_loader(
        dataset,
        parallel_state=parallel_state,
        collate_fn=collate,
        num_samples=request.num_samples,
        sample_selection=request.sample_selection,
    )


def move_prepared_eval_tensors(
    encoded: Mapping[str, Any],
    device: torch.device | str,
) -> dict[str, Any]:
    """Move top-level encoded tensors while retaining model-private values."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }


__all__ = [
    "PreparedEvalBatch",
    "PreparedEvalBatchPreparer",
    "PreparedEvalShape",
    "build_prepared_eval_loader",
    "move_prepared_eval_tensors",
]
