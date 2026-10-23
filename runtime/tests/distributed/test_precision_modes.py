import torch

from vforge.distributed.parallelize import (
    MixedPrecisionConfig,
    _cast_floating_buffers,
)


class _BufferModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.register_buffer("floating", torch.ones(2, dtype=torch.float64))
        self.register_buffer("integer", torch.ones(2, dtype=torch.int64))


def test_cast_floating_buffers_does_not_cast_parameters_or_integer_buffers():
    module = _BufferModule()

    _cast_floating_buffers(module, torch.bfloat16)

    assert module.weight.dtype == torch.float32
    assert module.floating.dtype == torch.bfloat16
    assert module.integer.dtype == torch.int64


def test_mixed_precision_policy_is_explicit():
    config = MixedPrecisionConfig(enable=True)

    assert config.param_dtype == "bfloat16"
    assert config.reduce_dtype == "float32"
    assert config.output_dtype is None
    assert config.buffer_dtype is None
    assert config.cast_forward_inputs


def test_mixed_precision_config_rejects_unknown_dtype_names():
    try:
        MixedPrecisionConfig(reduce_dtype="fp64")
    except ValueError as exc:
        assert "reduce_dtype" in str(exc)
    else:
        raise AssertionError("invalid dtype should fail")


def test_mixed_precision_config_carries_precision_contract_overrides():
    config = MixedPrecisionConfig(
        preserve_module_paths=("proj_in", "blocks.*.norm1"),
        preserve_forward_input_paths=(".", "blocks.*"),
        contract_mode="replace",
    )

    assert config.preserve_module_paths == ("proj_in", "blocks.*.norm1")
    assert config.preserve_forward_input_paths == (".", "blocks.*")
    assert config.contract_mode == "replace"
