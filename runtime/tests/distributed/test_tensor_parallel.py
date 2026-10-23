"""How a model declares its tensor-parallel splits, and what that costs it.

The declaration is the whole of a model's interface to tensor parallelism, so it
has properties worth pinning down separately from either backend:

* it shards nothing and needs no process group, so it works in a single process
  and a model file stays free of meshes, groups and engines;
* the vocabulary is PyTorch's own ``ParallelStyle``, so there is no vforge
  spelling to learn and none to keep in step with torch's;
* only the two styles both backends can implement are accepted, and the refusal
  lands where the declaration is written rather than at a conversion pass one
  engine away;
* at ``tp_size == 1`` what a model gets is literally ``nn.Linear``, which is why
  "tp off is the unparallelised model" is true by construction.

Everything about actually splitting a weight is in the per-backend files, which
need ranks. Nothing here does.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from vforge.distributed import build_linear, mark_tp_sharded
from vforge.distributed.tensor_parallel import (
    TP_STYLE_ATTR,
    restore_mcore_tensor_parallel_parameter_attributes,
    tensor_parallel_sharded_parameter_names,
    validate_mcore_tensor_parallel_parameter_attributes,
)


def test_a_declaration_shards_nothing_and_needs_no_group() -> None:
    """An ordinary Linear at the model's own dimensions, both ways round."""
    colwise = build_linear(16, 32, bias=True, style=ColwiseParallel())
    rowwise = build_linear(32, 16, bias=False, style=RowwiseParallel())

    assert type(colwise) is type(rowwise) is nn.Linear
    assert colwise.weight.shape == (32, 16)
    assert colwise.bias is not None
    assert rowwise.weight.shape == (16, 32)
    assert rowwise.bias is None


def test_the_declaration_is_carried_on_the_module() -> None:
    """On the module, not the parameter: ``to_empty()`` replaces parameters.

    Materializing a meta model swaps every Parameter object out, so a fact
    recorded on a tensor is gone by the time a checkpoint needs it. A module
    attribute survives.
    """
    style = ColwiseParallel()
    projection = build_linear(16, 32, style=style)

    assert getattr(projection, TP_STYLE_ATTR) is style


def test_sharded_parameter_names_follow_the_live_module_tree() -> None:
    model = nn.Module()
    model.add_module("projection", build_linear(16, 32, style=ColwiseParallel()))
    model.register_parameter("replicated", nn.Parameter(torch.ones(1)))

    assert tensor_parallel_sharded_parameter_names(model) == frozenset(
        {"projection.weight", "projection.bias"}
    )


def test_mcore_parameter_metadata_is_restored_after_parameter_replacement() -> None:
    projection = build_linear(4, 8, bias=True, style=ColwiseParallel())
    projection.weight = nn.Parameter(torch.empty_like(projection.weight))
    projection.bias = nn.Parameter(torch.empty_like(projection.bias))

    restore_mcore_tensor_parallel_parameter_attributes(projection)

    for parameter in (projection.weight, projection.bias):
        assert parameter.tensor_model_parallel is True
        assert parameter.partition_dim == 0
        assert parameter.partition_stride == 1
    validate_mcore_tensor_parallel_parameter_attributes(projection)


def test_mcore_parameter_metadata_validation_rejects_an_incomplete_layout() -> None:
    projection = build_linear(4, 8, bias=True, style=ColwiseParallel())
    projection.weight.tensor_model_parallel = True
    projection.weight.partition_dim = 1
    projection.weight.partition_stride = 1

    with pytest.raises(
        RuntimeError,
        match=r"weight has TP metadata .* expected \(True, 0, 1\)",
    ):
        validate_mcore_tensor_parallel_parameter_attributes(projection)


@pytest.mark.parametrize(
    "style",
    [SequenceParallel(), PrepareModuleInput(), "colwise", ColwiseParallel],
    ids=["sequence_parallel", "prepare_module_input", "a_string", "the_class_itself"],
)
def test_a_style_no_backend_can_honour_is_refused_where_it_is_written(style) -> None:
    """Including in a single-rank run, which never converts anything.

    Accepting one would promise a layout only torch native could implement, and
    the promise would not come due until somebody ran the megatron engine.
    """
    with pytest.raises(TypeError, match="must be one of"):
        build_linear(16, 32, style=style)
    with pytest.raises(TypeError, match="must be one of"):
        mark_tp_sharded(nn.RMSNorm(16), style)


@pytest.mark.parametrize(
    "style",
    [
        ColwiseParallel(),
        ColwiseParallel(output_layouts=Replicate()),
        RowwiseParallel(),
        RowwiseParallel(input_layouts=Replicate()),
    ],
    ids=["colwise", "colwise_gathered", "rowwise", "rowwise_scatters_input"],
)
def test_the_layouts_mcore_also_has_an_argument_for_are_accepted(style) -> None:
    """Both sides of each style's one varying axis.

    A gathered colwise output is ``gather_output=True`` over there, and a rowwise
    input that must be scattered is ``input_is_parallel=False``. Refusing them
    would hide something the other backend can do -- and the gathered output is
    not hypothetical: it is what a model needs where the next op wants the full
    width, which is why Wan's fused qkv path is still closed under tp.
    """
    assert build_linear(16, 32, style=style) is not None


@pytest.mark.parametrize(
    "style",
    [
        ColwiseParallel(use_local_output=False),
        ColwiseParallel(input_layouts=Shard(-1)),
        RowwiseParallel(input_layouts=Shard(1)),
        RowwiseParallel(output_layouts=Shard(1)),
    ],
    ids=["dtensor_output", "colwise_sharded_input", "rowwise_seq_input", "rowwise_seq_output"],
)
def test_a_layout_only_one_backend_could_do_is_refused(style) -> None:
    """The half that would otherwise go wrong in silence.

    Two different reasons, both worth refusing. A DTensor output mcore can never
    produce: its layers return ordinary tensors. A shard of the *sequence*
    dimension it does implement, but as ``config.sequence_parallel`` rather than
    per layer, so a model declaring it here would be honoured by the fsdp2 engine
    and ignored by megatron. Sequence parallelism will arrive as a run-level
    setting and generate these layouts; a model still may not ask per projection.
    """
    with pytest.raises(ValueError, match="use_local_output|must be"):
        build_linear(16, 32, style=style)
