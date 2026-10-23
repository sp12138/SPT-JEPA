from __future__ import annotations

import torch

from vforge.utils.comparison import batch_sha256


def test_batch_sha256_supports_scalar_bfloat16_and_nested_values() -> None:
    payload = {
        "loss_weight": torch.tensor(0.5, dtype=torch.float32),
        "hidden_states": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        "nested": [None, {"timestep": torch.tensor(7, dtype=torch.long)}],
    }

    digest = batch_sha256(payload)

    assert digest == batch_sha256(payload)
    assert digest != batch_sha256({**payload, "loss_weight": torch.tensor(0.6)})
