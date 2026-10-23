from types import SimpleNamespace

import torch

from vforge.distributed.fsdp2_patch import (
    ensure_activation_checkpoint_mixed_precision_compatibility,
)


def test_patch_is_idempotent_and_uses_mixed_precision_policy_dtype():
    import torch.distributed.fsdp._fully_shard._fsdp_state as fsdp_state

    first_install = ensure_activation_checkpoint_mixed_precision_compatibility()
    assert ensure_activation_checkpoint_mixed_precision_compatibility() is False

    # A fixed upstream build legitimately skips installation.
    if not first_install:
        assert hasattr(fsdp_state.FSDPState, "_cast_forward_inputs")
        return

    for target_dtype in (torch.bfloat16, torch.float16, torch.float32):
        state = SimpleNamespace(
            _training_state=fsdp_state.TrainingState.PRE_BACKWARD,
            _mp_policy=SimpleNamespace(
                cast_forward_inputs=True,
                param_dtype=target_dtype,
                output_dtype=target_dtype,
            ),
        )
        args, kwargs = fsdp_state.FSDPState._pre_forward(
            state,
            torch.nn.Identity(),
            (torch.ones(2, dtype=torch.float32),),
            {"value": torch.ones(2, dtype=torch.float32)},
        )
        output = fsdp_state.FSDPState._post_forward(
            state,
            torch.nn.Identity(),
            args,
            torch.ones(2, dtype=torch.float32),
        )

        assert args[0].dtype == target_dtype
        assert kwargs["value"].dtype == target_dtype
        assert output.dtype == target_dtype
