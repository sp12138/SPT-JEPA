"""Tensor parallelism: how a model declares its splits, and how each engine
makes them real.

A model states the split once, where the projection is built, and states nothing
about who will implement it:

    self.to_q = build_linear(dim, self.inner_dim, bias=True, style=ColwiseParallel())

The vocabulary is PyTorch's own ``ParallelStyle``, not one of ours. Every
framework has the same two cases and spells them almost identically -- torch as
these classes, mcore as ``ColumnParallelLinear`` / ``RowParallelLinear``,
transformers as the strings ``"colwise"`` / ``"rowwise"`` -- so there is nothing
to gain by inventing a third spelling, and a model author who has seen any of the
others recognises this one. Only the two sharding styles are accepted;
:func:`_validate_style` refuses the rest at the point of declaration rather than
letting a style no backend can honour reach a conversion pass.

What comes back is an ordinary ``nn.Linear`` carrying that style as a note.
Converting the note into a real shard belongs to the engine, and the two engines
do it differently because their layers differ in kind:

* :func:`apply_mcore_tensor_parallel` *replaces* each noted Linear with one of
  mcore's parallel layers, whose weight is allocated already narrow and whose
  forward contains the collectives. There is no way to convert a Linear into
  one, only to build one and put it in place.
* :func:`apply_torch_tensor_parallel` keeps the Linear and replaces the tensor
  inside it with a DTensor, letting ``parallelize_module`` install the
  collectives as forward hooks.

Deferring both to the engine is what keeps the decision out of the model. The
alternative, choosing inside ``build_linear``, means the choice has to travel
down through however many upstream constructors sit between the recipe and the
projection -- and through diffusers it cannot travel as an argument at all.

Splitting a weight also decides how it is later read back, so the persistence
boundary lives here too -- see :func:`install_tp_state_dict_hooks`. Keeping all
of it in one module is what stops the pieces from drifting: the axis a projection
is cut along and the axis a checkpoint says it was cut along are the same fact,
and :func:`_tp_shard_axes` is the only place it is written down.

At ``tp_size == 1`` nothing converts anything, so what a model gets is literally
what upstream would have built. Equivalence with the unparallelised path is a
property of construction rather than something a test has to chase.

mcore is imported lazily. It reaches triton at package import, which has no
macOS wheel, and this module sits on the import path of models that must stay
loadable wherever tp is off -- including the whole torch-native path, which has
no use for mcore at all.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

# Stock torch, and cheap: none of this reaches mcore, or therefore triton.
from torch.distributed.tensor import Placement, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
)

from vforge.distributed.parallel_state import get_parallel_state

#: The two styles a model may declare. Both shard a weight and leave an ordinary
#: tensor flowing to the next module -- ``use_local_output`` defaults to True --
#: which is what a model that reshapes a projection into heads needs.
SUPPORTED_STYLES = (ColwiseParallel, RowwiseParallel)

#: Set on a module whose weight is one slice of a wider logical parameter, holding
#: the style that says which way it was cut. Applied by :func:`build_linear` for
#: projections, and by a model itself for anything it sized locally by hand.
TP_STYLE_ATTR = "_vforge_tp_style"


def tensor_parallel_enabled() -> bool:
    return get_parallel_state().tp_enabled


def mark_tp_sharded(module: nn.Module, style: ParallelStyle) -> nn.Module:
    """Record how tensor parallelism splits this module's weight.

    Sharding a model is not only about the projections an engine replaces:
    anything a model sized to the local width is a slice too, and nothing about
    its type says so. Wan's q/k norm is the case -- it normalises across every
    head at once, so column-sharding q leaves it a slice of exactly the axis it
    reduces over. Declaring it keeps that fact next to the code that made the
    module local, rather than in whatever later needs to know.
    """
    _validate_style(style)
    setattr(module, TP_STYLE_ATTR, style)
    return module


#: Layouts each style may carry, being those mcore's layers can also express.
#:
#: Not a subset chosen for caution -- it is exactly what the other backend has a
#: constructor argument for. A colwise output is either kept sharded
#: (``gather_output=False``) or gathered (``gather_output=True``); a rowwise input
#: either arrives sharded (``input_is_parallel=True``) or is scattered on the way
#: in (``False``). The two fixed sides are fixed in mcore too: a colwise input is
#: always replicated, and a rowwise output is always all-reduced back to full.
#:
#: Absent on purpose is any shard of the *sequence* dimension. That is
#: Megatron-style sequence parallelism, and mcore does not express it per layer
#: at all -- it reads ``config.sequence_parallel`` and swaps the all-reduce for a
#: reduce-scatter. A run-level setting is where a choice that applies to every
#: projection at once belongs, so when it arrives the conversion pass will
#: produce these layouts rather than a model asking for them.
_ALLOWED_LAYOUTS: dict[type[ParallelStyle], dict[str, tuple[Placement, ...]]] = {
    ColwiseParallel: {
        "input_layouts": (Replicate(),),
        "output_layouts": (Shard(-1), Replicate()),
    },
    RowwiseParallel: {
        "input_layouts": (Shard(-1), Replicate()),
        "output_layouts": (Replicate(),),
    },
}


def _validate_style(style: ParallelStyle) -> None:
    """Accept the two sharding styles, with any layout both backends can do.

    The type check is the obvious half. The layouts are the half that would go
    wrong quietly: a style asking for something only one backend implements would
    be honoured by that one and ignored by the other, and two engines would
    compute different things from the same model without anything being raised.
    """
    if not isinstance(style, SUPPORTED_STYLES):
        supported = ", ".join(cls.__name__ for cls in SUPPORTED_STYLES)
        raise TypeError(
            f"tensor parallel style must be one of {supported}; got "
            f"{type(style).__name__}. The others describe layouts that only the "
            "torch-native backend could honour, so accepting one here would "
            "promise something mcore cannot do"
        )
    if not style.use_local_output:
        raise ValueError(
            f"{type(style).__name__} must have use_local_output=True: mcore's "
            "layers return ordinary tensors and have no way to hand back a "
            "DTensor. It is also what a model reshaping a projection into heads "
            "needs, whichever backend it runs on"
        )
    allowed = _ALLOWED_LAYOUTS[type(style)]
    for attribute, permitted in allowed.items():
        placement = _sole_placement(style, attribute)
        if placement not in permitted:
            names = " or ".join(repr(option) for option in permitted)
            raise ValueError(
                f"{type(style).__name__}.{attribute} must be {names}; got "
                f"{placement!r}. mcore's layers have no constructor argument for "
                "that, so the fsdp2 engine would honour it and megatron would not"
            )


def _sole_placement(style: ParallelStyle, attribute: str) -> Placement:
    layouts = getattr(style, attribute)
    if not isinstance(layouts, tuple) or len(layouts) != 1:
        raise ValueError(
            f"{type(style).__name__}.{attribute} must describe exactly one tensor; "
            f"got {layouts!r}. A projection takes one input and returns one output"
        )
    return layouts[0]


def build_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    style: ParallelStyle,
) -> nn.Module:
    """A projection, and a note saying how tensor parallelism splits it.

    This is the model author's whole interface to tensor parallelism, alongside
    ``local_heads`` and a group-aware norm. What comes back is an ordinary
    ``nn.Linear``: nothing is sharded here, whatever ``tp_size`` is. The engine
    converts the note later, which is what keeps the choice of implementation out
    of the model.

    ``in_features`` and ``out_features`` are always the full, unsharded sizes.
    Every backend divides by the group size itself, so callers state the model's
    dimensions and never the local ones.

    The style says which way the weight is cut, and the judgement is always the
    same: a projection that *produces* a feature dimension is
    ``ColwiseParallel``, one that *consumes* it is ``RowwiseParallel``. In
    attention q/k/v are colwise and the output projection is rowwise; in a
    feed-forward the up projection is colwise and the down projection is rowwise.
    A colwise must be followed by a rowwise before the tensor is used at its full
    width again.
    """
    # Validated here rather than at conversion, so an unsupported style is caught
    # by the single-rank run that never converts anything.
    _validate_style(style)
    return mark_tp_sharded(nn.Linear(in_features, out_features, bias=bias), style)


class ShardedRMSNorm(nn.Module):
    """RMSNorm over a feature dimension that is split across the tp group.

    A norm whose feature axis is sharded cannot be computed rank-locally: the
    root mean square is over the whole vector, and each rank holds a slice of
    it. Normalising the slice on its own is not an approximation, it is a
    different function, and it produces a plausible model that does not match
    the one the weights came from.

    So the sum of squares is completed across the group. That is one all-reduce
    of a scalar per token, next to projections that move the hidden dimension
    -- the cost is in the collective's latency rather than its bytes.

    Only used when tp is on; at ``tp_size == 1`` callers build ``nn.RMSNorm``
    and this class never appears, which keeps the unsharded path exactly
    upstream's.
    """

    def __init__(
        self,
        local_features: int,
        *,
        eps: float | None = None,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        state = get_parallel_state()
        if not state.tp_enabled:
            raise RuntimeError("ShardedRMSNorm requires tensor parallelism to be enabled")
        self.local_features = int(local_features)
        self.full_features = self.local_features * int(state.config.tp_size)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.local_features)) if elementwise_affine else None
        # Colwise, because the weight is one slice of a wider logical parameter
        # cut along dim 0 -- the same shape of fact a colwise projection states,
        # even though nothing here is a projection.
        mark_tp_sharded(self, ColwiseParallel())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        group = get_parallel_state().tp_group
        source_dtype = hidden_states.dtype
        promoted = hidden_states.float()
        sum_of_squares = promoted.pow(2).sum(-1, keepdim=True)
        sum_of_squares = _all_reduce_sum(sum_of_squares, group)
        # Mirrors nn.RMSNorm: eps defaults to the *input* dtype's epsilon, not
        # the float32 one this happens to be accumulating in.
        eps = torch.finfo(source_dtype).eps if self.eps is None else self.eps
        normalized = promoted * torch.rsqrt(sum_of_squares / self.full_features + eps)
        normalized = normalized.to(source_dtype)
        if self.weight is None:
            return normalized
        return normalized * _local_view(self.weight)

    def extra_repr(self) -> str:
        return f"local_features={self.local_features}, full_features={self.full_features}"


def _local_view(tensor: torch.Tensor) -> torch.Tensor:
    """This rank's slice, whichever backend produced the parameter.

    mcore leaves an ordinary tensor of the local width. torch native turns the
    same parameter into a DTensor describing the whole width, and multiplying
    one of those by an ordinary activation is an error rather than a broadcast.
    ``to_local`` is differentiable, so the distinction stays out of the backward
    pass as well.
    """
    to_local = getattr(tensor, "to_local", None)
    return to_local() if callable(to_local) else tensor


def _all_reduce_sum(tensor: torch.Tensor, group: Any) -> torch.Tensor:
    if group is None:
        return tensor
    return _AllReduceSum.apply(tensor, group)


class _AllReduceSum(torch.autograd.Function):
    """Sum across the group, and sum the gradient back across it.

    Not the same as mcore's ``reduce_from_tensor_model_parallel_region``, whose
    backward is the identity. That holds where every rank goes on to do the
    same work on the same reduced value, so each already has the whole
    gradient. Here each rank continues on its own shard, so what comes back is
    one rank's share of the gradient and the shares have to be added.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: Any) -> torch.Tensor:
        ctx.group = group
        reduced = tensor.clone()
        torch.distributed.all_reduce(reduced, group=group)
        return reduced

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad = grad_output.clone()
        torch.distributed.all_reduce(grad, group=ctx.group)
        return grad, None


def shard_full_state_dict(module: nn.Module, full: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Cut a whole-model state dict down to this rank's tensor-parallel slice.

    ``full`` holds logical tensors: the shapes an unparallelised model would
    have, which is what a pretrained checkpoint and a saved vforge checkpoint
    both contain. This returns the same mapping with every sharded entry
    narrowed to the part this rank owns, so the result can be handed to
    ``load_state_dict`` unchanged. With tp off it is a copy and nothing else.

    Cutting by hand like this is for callers holding a whole state dict already
    in memory. The path a run takes instead is
    :func:`install_tp_state_dict_hooks`, which lets DCP do the same narrowing as
    it reads, one tensor at a time, rather than materializing the unsharded
    model first.
    """
    state = get_parallel_state()
    if not state.tp_enabled:
        return dict(full)

    rank = state.tp_rank
    tp_size = state.config.tp_size
    sharded = dict(full)
    for name, submodule in module.named_modules():
        prefix = f"{name}." if name else ""
        for key, dim in _tp_shard_axes(submodule).items():
            _narrow_entry(sharded, f"{prefix}{key}", dim, rank, tp_size)
    return sharded


def tensor_parallel_sharded_parameter_names(model: nn.Module) -> frozenset[str]:
    """Return FQNs whose local parameters are TP slices rather than replicas.

    The answer follows the live module tree, so an outer wrapper such as DDP is
    reflected in the names it returns.  Diagnostics use this distinction to
    count every shard once while counting a replicated parameter on only one TP
    rank.
    """
    names: set[str] = set()
    for module_name, submodule in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for key in _tp_shard_axes(submodule):
            if isinstance(getattr(submodule, key, None), nn.Parameter):
                names.add(prefix + key)
    return frozenset(names)


def restore_mcore_tensor_parallel_parameter_attributes(model: nn.Module) -> None:
    """Restore MCore's per-Parameter TP metadata after materialization.

    ``to_empty`` replaces Parameter objects and therefore drops the attributes
    MCore layers originally installed on them. vForge's module-level TP style
    survives, so reconstruct the metadata before MCore DDP/DistributedOptimizer
    classify TP shards for gradient norms and optimizer-state partitioning.
    """
    for submodule in model.modules():
        for parameter_name, partition_dim in _tp_shard_axes(submodule).items():
            parameter = getattr(submodule, parameter_name, None)
            if not isinstance(parameter, nn.Parameter):
                continue
            parameter.tensor_model_parallel = True
            parameter.partition_dim = partition_dim
            parameter.partition_stride = 1


def validate_mcore_tensor_parallel_parameter_attributes(model: nn.Module) -> None:
    """Require every declared MCore TP shard to carry its runtime metadata.

    The module declaration is the durable source of truth. This check pins down
    the post-materialization contract before weight loading, DDP, or the
    distributed optimizer can classify parameters using MCore's transient
    per-Parameter attributes.
    """
    from torch.distributed.tensor import DTensor

    failures: list[str] = []
    for module_name, submodule in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for parameter_name, partition_dim in _tp_shard_axes(submodule).items():
            parameter = getattr(submodule, parameter_name, None)
            if parameter is None:
                continue
            full_name = prefix + parameter_name
            if not isinstance(parameter, nn.Parameter):
                failures.append(f"{full_name} is not an nn.Parameter")
                continue
            if isinstance(parameter, DTensor):
                failures.append(f"{full_name} is a DTensor, not an MCore local parameter")
                continue
            actual = (
                getattr(parameter, "tensor_model_parallel", None),
                getattr(parameter, "partition_dim", None),
                getattr(parameter, "partition_stride", None),
            )
            expected = (True, partition_dim, 1)
            if actual != expected:
                failures.append(f"{full_name} has TP metadata {actual!r}; expected {expected!r}")
    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"invalid MCore tensor-parallel parameter metadata:\n{details}")


def _tp_shard_axes(module: nn.Module) -> dict[str, int]:
    """Which of this module's own parameters are slices, and along which axis.

    Empty for a module that holds whole parameters, which is most of them.

    Read off :data:`TP_STYLE_ATTR`, which every sharded module carries: the ones
    :func:`build_linear` returns, the mcore layers that later replace them, and
    the ones a model sized to the local width itself. Never off the
    ``tensor_model_parallel`` attributes mcore leaves on its parameters -- those
    are lost when ``to_empty()`` replaces the Parameter objects during
    materialization, while an attribute on the module survives it.

    The declared style rather than also recognising mcore's types, because asking
    about a type means importing mcore, and this is on the path of the
    torch-native backend, which has no use for it and may be running where it
    cannot load.

    The two styles do not mirror each other. Colwise splits the output, so the
    bias is cut along with the weight's rows. Rowwise splits the input, so only
    the weight's columns are: its bias is added after the all-reduce, which means
    every rank holds the whole of it and cutting it would count it once per rank.
    """
    declared = getattr(module, TP_STYLE_ATTR, None)
    if isinstance(declared, ColwiseParallel):
        return {"weight": 0, "bias": 0}
    if isinstance(declared, RowwiseParallel):
        return {"weight": 1}
    return {}


def tensor_parallel_shard_axes(module: nn.Module) -> dict[str, int]:
    """Return this module's durable parameter-to-TP-axis declaration."""
    return dict(_tp_shard_axes(module))


def apply_torch_tensor_parallel(model: nn.Module) -> nn.Module:
    """Hand the styles a model declared to ``parallelize_module``.

    Runs once the whole tree exists, which is what the declarations were waiting
    for: a plan is keyed by a module's fully qualified name, and a module cannot
    know its own name during ``__init__`` -- the name is given by the parent's
    ``self.to_q = ...``, after the child is built.

    The usual way to write this is by hand, a plan of literal names beside the
    model. Collecting it off the modules buys two things. A model states how each
    projection splits exactly once, in the call that builds it, which is also the
    only form mcore can use -- a hand-written plan would be a second copy of the
    same fact, kept in step by nothing. And because the plan is derived from the
    tree rather than written against it, renaming a projection cannot silently
    stop matching, which is a failure mode the frameworks that key plans by name
    do hit.

    Two kinds of declared module, and only one of them a style can be handed to
    ``parallelize_module``. A ``Linear`` goes into the plan as declared. Anything
    else was sized to the local width by the model itself, and no style means
    "this parameter simply is one shard", so it is declared directly. Left alone
    it would not merely be unsharded: fsdp2 would later wrap the local-width
    tensor as a DTensor over the *data* mesh, and the state dict would then report
    the local width as though it were the whole parameter, under one name on every
    tp rank.

    Called by the fsdp2 engine, before it shards over the data mesh so that the
    two compose into one DTensor placed on both.
    """
    state = get_parallel_state()
    if not state.tp_enabled:
        return model
    mesh = state.tp_mesh
    if mesh is None:
        raise RuntimeError(
            "tensor parallelism is on but there is no tp mesh to shard over; the "
            "parallel state has to be initialized before the model"
        )

    from torch.distributed.tensor.parallel import parallelize_module

    plan: dict[str, Any] = {}
    for name, submodule in model.named_modules():
        style = getattr(submodule, TP_STYLE_ATTR, None)
        if style is None:
            continue
        if isinstance(submodule, nn.Linear):
            plan[name] = style
        else:
            _declare_existing_shards(submodule, mesh)
    if plan:
        parallelize_module(model, mesh, plan)
    return model


def apply_mcore_tensor_parallel(model: nn.Module) -> nn.Module:
    """Replace each noted projection with one of mcore's parallel layers.

    Replacement rather than conversion, because mcore's layers are a different
    class whose weight is allocated already narrow: a ``[5120, 5120]``
    projection at ``tp_size=2`` comes out of ``ColumnParallelLinear.__init__``
    as ``[2560, 5120]``, and the collectives live in its own ``forward``. There
    is no operation that turns an ``nn.Linear`` into one.

    Which costs nothing worth avoiding. The Linear being discarded was built on
    the meta device, so it never held memory, and assigning over the attribute
    keeps the parameter named ``attn1.to_q.weight`` -- the name a pretrained
    checkpoint and every state-dict mapper already use.

    Two things do have to be carried across, and both would be silent if missed.
    The new layer is built under the device of the one it replaces, so a model
    still on meta stays there rather than allocating 14B parameters. And its
    dtype is taken from the same place, because Wan casts its parameters after
    construction: a layer built later would otherwise arrive in the default
    dtype and be the only one in the model that is.

    Modules the model sized locally by hand are left alone. Under mcore they
    stay ordinary tensors, which is the whole reason its parameters can be
    matmul'd directly, and the state-dict hooks are what make them legible to a
    checkpoint later.

    Called by the megatron engine.
    """
    state = get_parallel_state()
    if not state.tp_enabled:
        return model

    column_cls, row_cls = _tensor_parallel_linear_types()
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            style = getattr(child, TP_STYLE_ATTR, None)
            if style is None or not isinstance(child, nn.Linear):
                continue
            setattr(parent, name, _mcore_linear(child, style, column_cls, row_cls, state))
    return model


def _mcore_linear(
    template: nn.Linear,
    style: ParallelStyle,
    column_cls: type[nn.Module],
    row_cls: type[nn.Module],
    state: Any,
) -> nn.Module:
    """One mcore parallel layer shaped like, placed like, and told what ``style`` says.

    The one varying side of each style has a constructor argument here, so it is
    read across rather than assumed: a colwise output is gathered or not, a
    rowwise input arrives sharded or is scattered on the way in. The fixed sides
    -- a replicated colwise input, an all-reduced rowwise output -- are what these
    layers do unconditionally, which is why :func:`_validate_style` allows nothing
    else there.
    """
    config = _model_parallel_config(state.config.tp_size, template.weight.dtype)
    common = {
        "config": config,
        "init_method": _unused_init,
        "bias": template.bias is not None,
        "skip_bias_add": False,
        "tp_group": state.tp_group,
    }
    with template.weight.device:
        if isinstance(style, ColwiseParallel):
            layer = column_cls(
                template.in_features,
                template.out_features,
                gather_output=isinstance(_sole_placement(style, "output_layouts"), Replicate),
                **common,
            )
        else:
            layer = row_cls(
                template.in_features,
                template.out_features,
                input_is_parallel=isinstance(_sole_placement(style, "input_layouts"), Shard),
                **common,
            )
    # Carries the same style, though its type already says so, because reading the
    # declaration is what keeps everything downstream from importing mcore to ask.
    return mark_tp_sharded(layer, style)


def _declare_existing_shards(module: nn.Module, mesh: Any) -> None:
    """Say what a parameter the model already sized locally is a shard of."""
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor import Shard as ShardPlacement

    for key, axis in _tp_shard_axes(module).items():
        parameter = getattr(module, key, None)
        if parameter is None or isinstance(parameter, DTensor):
            continue
        # from_local, not distribute_tensor: the parameter was built at this
        # rank's width rather than the model's, so it already is the shard.
        setattr(
            module,
            key,
            nn.Parameter(
                DTensor.from_local(parameter, mesh, [ShardPlacement(axis)]),
                requires_grad=parameter.requires_grad,
            ),
        )


#: Set once hooks are on a module, so installing twice does not wrap twice.
_HOOKED_ATTR = "_vforge_tp_state_dict_hooked"
_POST_HOOK_ATTR = "_vforge_tp_state_dict_post_hook"
_POST_HOOK_HANDLE_ATTR = "_vforge_tp_state_dict_post_hook_handle"


def install_tp_state_dict_hooks(model: nn.Module) -> nn.Module:
    """Make this rank's tensor-parallel slices legible where weights are read.

    mcore hands each rank an ordinary tensor holding part of a weight, and
    nothing on it says so. Everything vforge reads or writes weights with goes
    through ``get_model_state_dict`` / ``set_model_state_dict``, and those take a
    shard's identity off ``DTensor`` and from nowhere else. Given plain tensors
    they conclude every rank holds a whole weight, which is wrong in three
    places at once: pretrained safetensors are handed over uncut and the shapes
    do not match, DCP sees one name per rank and keeps a single shard, and the PT
    export gathers nothing and writes rank zero's fraction.

    So the shards are declared where they leave the model rather than converted
    where they are stored. Going out, a ``state_dict`` hook reports the local
    slice as the ``DTensor`` it always logically was; coming in, a load hook
    takes the wrapper back off, or narrows a whole tensor that arrived without
    one. The parameters themselves stay ordinary tensors throughout the step,
    which is the point: mcore's layers matmul ``self.weight`` directly, and a
    DTensor there would have PyTorch inferring shard propagation through kernels
    written on the assumption that every collective is explicit.

    Two things follow. The checkpoint layer never learns that mcore exists, and
    what lands on disk is the unsharded tensor -- so a run saved at one
    ``tp_size`` can be read at another, or by fsdp2.
    """
    state = get_parallel_state()
    if not state.tp_enabled:
        return model
    mesh = state.tp_mesh
    if mesh is None:
        raise RuntimeError(
            "tensor parallelism is on but there is no tp mesh to describe shards "
            "against; the parallel state has to be initialized before the model"
        )

    rank = state.tp_rank
    tp_size = state.config.tp_size
    for submodule in model.modules():
        axes = _tp_shard_axes(submodule)
        if not axes or getattr(submodule, _HOOKED_ATTR, False):
            continue
        post_hook = _declare_shards(axes, mesh)
        post_hook_handle = submodule.register_state_dict_post_hook(post_hook)
        submodule.register_load_state_dict_pre_hook(_accept_shards(axes, rank, tp_size))
        setattr(submodule, _POST_HOOK_ATTR, post_hook)
        setattr(submodule, _POST_HOOK_HANDLE_ATTR, post_hook_handle)
        setattr(submodule, _HOOKED_ATTR, True)
    return model


@contextmanager
def suspend_tensor_parallel_state_dict_post_hooks(model: nn.Module):
    """Temporarily expose MCore TP parameters as plain local tensors.

    The normal vForge state-dict boundary wraps MCore TP shards as DTensors so
    Torch DCP can infer their logical TP axis. MCore native checkpointing has
    its own sharded-state builder and must see the original local tensor;
    otherwise it mistakes that DTensor for an additional FSDP axis and folds
    the DP size into the tensor's global shape.
    """
    suspended: list[tuple[nn.Module, Any]] = []
    for submodule in model.modules():
        handle = getattr(submodule, _POST_HOOK_HANDLE_ATTR, None)
        post_hook = getattr(submodule, _POST_HOOK_ATTR, None)
        if handle is None or post_hook is None:
            continue
        handle.remove()
        suspended.append((submodule, post_hook))
    try:
        yield
    finally:
        for submodule, post_hook in suspended:
            handle = submodule.register_state_dict_post_hook(post_hook)
            setattr(submodule, _POST_HOOK_HANDLE_ATTR, handle)


def _declare_shards(axes: Mapping[str, int], mesh: Any) -> Any:
    """On the way out, report each local slice as a logically whole tensor."""
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor import Shard as ShardPlacement

    def hook(
        module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        _local_metadata: Any,
    ) -> None:
        for key, dim in axes.items():
            full_key = prefix + key
            value = state_dict.get(full_key)
            if value is None:
                continue
            if isinstance(value, DTensor):
                raise NotImplementedError(
                    f"{full_key} is already a DTensor, so something else -- fsdp2, "
                    "most likely -- has sharded it too. Two axes need a "
                    "two-dimensional mesh, which this hook does not build: run "
                    "tensor parallelism without data sharding until it does"
                )
            state_dict[full_key] = DTensor.from_local(value, mesh, [ShardPlacement(dim)])

    return hook


def _accept_shards(axes: Mapping[str, int], rank: int, tp_size: int) -> Any:
    """On the way in, reduce whatever arrived to this rank's plain slice.

    Two shapes turn up. DCP and anything else that read our own state dict hand
    back a ``DTensor``, already narrowed, and only the wrapper has to go. A whole
    tensor arrives from the paths that never sharded: a PT export being used as a
    warm start, or ``set_model_state_dict`` broadcasting rank zero's copy.

    A tensor that is neither is left exactly as it came, so that
    ``load_state_dict`` reports the size mismatch itself rather than this hook
    quietly cutting a wrong-sized tensor down to a plausible one.
    """
    from torch.distributed.tensor import DTensor

    def hook(
        module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        _local_metadata: Any,
        _strict: bool,
        _missing_keys: list[str],
        _unexpected_keys: list[str],
        _error_msgs: list[str],
    ) -> None:
        for key, dim in axes.items():
            full_key = prefix + key
            value = state_dict.get(full_key)
            if value is None:
                continue
            if isinstance(value, DTensor):
                state_dict[full_key] = value.to_local()
                continue
            local = getattr(module, key, None)
            if local is None or value.shape[dim] != local.shape[dim] * tp_size:
                continue
            _narrow_entry(state_dict, full_key, dim, rank, tp_size)

    return hook


def _narrow_entry(
    state_dict: dict[str, Any],
    key: str,
    dim: int,
    rank: int,
    tp_size: int,
) -> None:
    tensor = state_dict.get(key)
    if tensor is None:
        return
    width = tensor.shape[dim]
    if width % tp_size != 0:
        raise ValueError(
            f"{key} has {width} along dim {dim}, which tp_size {tp_size} does not divide"
        )
    local = width // tp_size
    state_dict[key] = tensor.narrow(dim, rank * local, local).contiguous()


def _unused_init(weight: torch.Tensor) -> torch.Tensor:
    """Never called: ``perform_initialization`` is off, but the arg is required.

    Weights arrive from a pretrained checkpoint after the module is
    materialized, so initializing here would be work thrown away, and doing it
    on a meta tensor would fail outright.
    """
    return weight


def _model_parallel_config(tp_size: int, params_dtype: torch.dtype):
    from vforge.mcore.model_parallel_config import ModelParallelConfig

    return ModelParallelConfig(
        tensor_model_parallel_size=tp_size,
        # Despite the name this is the branch that allocates with a bare
        # torch.empty, so it honours an ambient device context. The other one
        # hardcodes torch.cuda.current_device(), which would defeat the meta
        # construction used to size a model before materializing it.
        use_cpu_initialization=True,
        perform_initialization=False,
        params_dtype=params_dtype,
    )


@functools.lru_cache(maxsize=1)
def _tensor_parallel_linear_types() -> tuple[type[nn.Module], type[nn.Module]]:
    """Subclasses that return a tensor, so a model's forward is untouched.

    mcore returns ``(output, bias)`` because it lets a caller fuse the bias into
    a later kernel. No model here has such a caller, and subclassing rather than
    wrapping keeps parameter names as ``to_q.weight``, which pretrained
    checkpoints and every state-dict mapper already expect.
    """
    from vforge.mcore.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    class TensorParallelColumnLinear(ColumnParallelLinear):
        def forward(self, input_, *args, **kwargs):
            output, _bias = super().forward(input_, *args, **kwargs)
            return output

    class TensorParallelRowLinear(RowParallelLinear):
        def forward(self, input_, *args, **kwargs):
            output, _bias = super().forward(input_, *args, **kwargs)
            return output

    return TensorParallelColumnLinear, TensorParallelRowLinear


__all__ = [
    "SUPPORTED_STYLES",
    "TP_STYLE_ATTR",
    "ShardedRMSNorm",
    "apply_mcore_tensor_parallel",
    "apply_torch_tensor_parallel",
    "build_linear",
    "install_tp_state_dict_hooks",
    "mark_tp_sharded",
    "restore_mcore_tensor_parallel_parameter_attributes",
    "shard_full_state_dict",
    "tensor_parallel_enabled",
    "tensor_parallel_shard_axes",
    "tensor_parallel_sharded_parameter_names",
    "suspend_tensor_parallel_state_dict_post_hooks",
    "validate_mcore_tensor_parallel_parameter_attributes",
]
