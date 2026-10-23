"""CPU coverage for vForge's MCore DDP lifecycle adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vforge.distributed.mcore_data_parallel import (
    finish_mcore_data_parallel_parameter_sync,
    prepare_mcore_data_parallel_backward,
)


class _FakeMCoreDDP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.unused = torch.nn.Parameter(torch.ones(()))
        self._vforge_data_parallel_backend = "mcore"
        self.ddp_config = SimpleNamespace(overlap_param_gather=True)
        self.bucket_group = SimpleNamespace(
            ddp_config=SimpleNamespace(overlap_grad_reduce=True),
            params={self.weight, self.unused},
            per_param_grad_ready_counts={},
        )
        self.bucket_groups = [self.bucket_group]
        self.expert_parallel_bucket_groups = []
        self.start_param_sync = Mock()

    def zero_grad_buffer(self) -> None:
        self.bucket_group.per_param_grad_ready_counts = {}


def test_prepare_backward_seeds_zero_hook_counts_for_unused_parameters() -> None:
    model = _FakeMCoreDDP()

    prepare_mcore_data_parallel_backward(model)

    assert model.bucket_group.per_param_grad_ready_counts == {
        model.weight: 0,
        model.unused: 0,
    }


def test_diagnostic_parameter_sync_forces_all_buckets_to_finish() -> None:
    model = _FakeMCoreDDP()

    finish_mcore_data_parallel_parameter_sync(model)

    model.start_param_sync.assert_called_once_with(force_sync=True)
