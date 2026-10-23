"""Import smoke test for the initial vforge scaffold."""

from __future__ import annotations

import torch

from vforge import ParallelConfig, get_parallel_state, init_parallel_state
from vforge.utils.comparison import tensor_sha256
from vforge.utils.rng import seed_everything


def main() -> None:
    seed_everything(1234)
    state = init_parallel_state(ParallelConfig())
    digest = tensor_sha256(torch.arange(4, dtype=torch.float32))
    print(
        "vforge smoke ok "
        f"world_size={state.world_size} "
        f"rank={get_parallel_state().global_rank} "
        f"hash={digest[:12]}"
    )


if __name__ == "__main__":
    main()
