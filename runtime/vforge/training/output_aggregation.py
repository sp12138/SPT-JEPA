"""Aggregate micro-batch outputs for optimizer-update reporting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from vforge.training.recipe import StepOutput


def _detach_metadata(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _detach_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_metadata(item) for item in value)
    if isinstance(value, list):
        return [_detach_metadata(item) for item in value]
    return value


class StepOutputAggregator:
    """Aggregate micro-batch outputs for reporting after an optimizer update."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.single_output: StepOutput | None = None
        self.loss: torch.Tensor | None = None
        self.metrics: dict[str, torch.Tensor | float] = {}
        self.metric_counts: dict[str, int] = {}
        self.workload: dict[str, float | int] | None = None
        self.metadata: list[Mapping[str, Any]] = []

    def add(self, output: StepOutput) -> None:
        if output.workload and not {"local_batch", "local_tokens"} <= output.workload.keys():
            raise ValueError("workload must provide local_batch and local_tokens")
        if self.count == 1:
            # Preserve the single-batch contract, including lazy loss metadata.
            self.single_output = output
            return
        loss = output.loss.detach().float()
        self.loss = loss.clone() if self.loss is None else self.loss + loss
        for key, value in output.metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().float().reshape(())
            elif isinstance(value, (int, float)):
                value = float(value)
            else:
                continue
            self.metrics[key] = self.metrics.get(key, 0.0) + value
            self.metric_counts[key] = self.metric_counts.get(key, 0) + 1
        if output.workload is not None:
            if self.workload is None:
                self.workload = {}
            for key, value in output.workload.items():
                self.workload[key] = self.workload.get(key, 0) + value
        self.metadata.append(_detach_metadata(output.metadata))

    def finish(self) -> StepOutput:
        if self.single_output is not None:
            return self.single_output
        assert self.loss is not None
        return StepOutput(
            loss=self.loss / self.count,
            metrics={key: value / self.metric_counts[key] for key, value in self.metrics.items()},
            workload=self.workload,
            metadata={"microbatches": self.metadata},
        )
