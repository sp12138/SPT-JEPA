"""Model parallelization entrypoints.

The public API here is intentionally one function: external training loops can
build their model as usual, then call ``parallelize_model`` before optimizer
construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard

from vforge.distributed.fsdp2_patch import (
    ensure_activation_checkpoint_mixed_precision_compatibility,
    validate_gradient_divide_factor,
)
from vforge.distributed.mcore_data_parallel import MCoreDataParallelConfig
from vforge.distributed.parallel_state import (
    ParallelConfig,
    get_parallel_state,
    init_parallel_state,
)
from vforge.models.loading import keep_in_fp32_module_paths


@dataclass(frozen=True)
class FSDPPrecisionContract:
    """Model defaults for exceptional FSDP mixed-precision groups."""

    preserve_module_paths: tuple[str, ...] = ()
    preserve_forward_input_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class FSDPModulePlan:
    """Resolved FSDP settings for a module with additional model parallelism."""

    mesh: DeviceMesh
    shard_placement: Shard
    gradient_divide_factor: float | None = None


@dataclass(frozen=True)
class MixedPrecisionConfig:
    enable: bool = False
    param_dtype: str | None = "bfloat16"
    reduce_dtype: str | None = "float32"
    output_dtype: str | None = None
    buffer_dtype: str | None = None
    cast_forward_inputs: bool = True
    apply_model_overrides: bool = True
    preserve_module_paths: tuple[str, ...] = ()
    preserve_forward_input_paths: tuple[str, ...] = ()
    contract_mode: str = "append"

    def __post_init__(self) -> None:
        for field_name in (
            "param_dtype",
            "reduce_dtype",
            "output_dtype",
            "buffer_dtype",
        ):
            _validate_dtype_name(field_name, getattr(self, field_name))
        if self.contract_mode not in {"append", "replace"}:
            raise ValueError("mixed-precision contract_mode must be 'append' or 'replace'")
        _validate_module_paths(
            "preserve_module_paths",
            self.preserve_module_paths,
        )
        _validate_module_paths(
            "preserve_forward_input_paths",
            self.preserve_forward_input_paths,
            allow_root=True,
        )


@dataclass(frozen=True)
class FSDP2Config:
    enable: bool = True
    reshard_after_forward: bool = True
    mixed_precision: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)
    basic_modules: tuple[str, ...] = ()
    leaf_module_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParallelizeConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    fsdp2: FSDP2Config = field(default_factory=FSDP2Config)
    mcore_ddp: MCoreDataParallelConfig = field(default_factory=MCoreDataParallelConfig)
    gradient_checkpointing: bool = False


def parallelize_model(
    model: nn.Module,
    config: ParallelizeConfig | None = None,
    *,
    gradient_checkpointing_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Shard ``model`` over the data mesh, and enable activation recompute.

    - initialize/reuse ``ParallelState``
    - enable gradient checkpointing when the model supports it
    - apply the model's EP plan and resolve per-module FSDP settings
    - apply FSDP2 bottom-up to target blocks and the root

    **Tensor parallelism must already have been applied.** It is not done here
    because *which* conversion to run depends on the engine -- torch native
    replaces the tensor inside a Linear, mcore replaces the Linear -- while this
    function is shared by both engines and by evaluation, so it would have to be
    told which caller it has. Each engine therefore applies its own first, which
    is also the order that matters: FSDP2 then wraps weights already carrying a tp
    axis and composes with it, rather than describing the data mesh alone and
    losing that axis without failing.

    Optimizer construction should happen after this function returns.
    """
    cfg = config or ParallelizeConfig()
    state = init_parallel_state(cfg.parallel)
    if state.ep_enabled and not cfg.fsdp2.enable:
        raise ValueError("EP requires FSDP2 enabled")

    if cfg.gradient_checkpointing:
        enable = getattr(model, "gradient_checkpointing_enable", None)
        if not callable(enable):
            raise TypeError(
                f"{type(model).__name__} must implement "
                "gradient_checkpointing_enable(gradient_checkpointing_kwargs=...)"
            )
        enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    if not cfg.fsdp2.enable or not state.fsdp_enabled:
        return model

    extra_plan = {}
    if state.ep_enabled:
        from vforge.distributed.expert_parallel import ExpertParallelPlan

        # TODO: Define a shared model parallel-plan interface and resolve it
        # outside this EP branch when other parallel strategies consume it.
        resolver = getattr(model, "get_parallel_plan", None)
        if not callable(resolver):
            raise ValueError("EP requires model.get_parallel_plan()")
        plan = resolver()
        if not isinstance(plan, ExpertParallelPlan):
            raise TypeError("get_parallel_plan() must return ExpertParallelPlan")
        extra_plan = plan.apply(model, state)

    return _apply_fsdp2(model, cfg.fsdp2, extra_plan=extra_plan)


def materialize_meta_module(
    model: nn.Module,
    device: torch.device | str,
) -> nn.Module:
    """Allocate meta buffers/parameters after FSDP2 has established shards."""
    if module_has_meta_tensors(model):
        runtime_buffer_resets = []
        for module in model.modules():
            meta_runtime_buffers = [
                name
                for name, buffer in module._buffers.items()
                if name in module._non_persistent_buffers_set
                and buffer is not None
                and buffer.is_meta
            ]
            if not meta_runtime_buffers:
                continue
            reset = getattr(module, "reset_non_persistent_buffers", None)
            if not callable(reset):
                raise RuntimeError(
                    f"{type(module).__name__} has meta non-persistent buffers "
                    f"{meta_runtime_buffers} but does not implement "
                    "reset_non_persistent_buffers()"
                )
            runtime_buffer_resets.append(reset)
        model.to_empty(device=device)
        for reset in runtime_buffer_resets:
            reset()
    return model


def apply_replicated_data_parallel(model: nn.Module) -> nn.Module:
    """Wrap a materialized model in replicated DDP over the DP x CP group.

    Tensor-parallel peers own different parameter shards, so the default world
    group is never the right DDP domain. The flattened DP x CP group instead
    contains ranks with the same TP coordinate. DP peers contribute different
    batches while CP peers contribute different sequence shards.

    This runs after pretrained initialization and before optimizer construction.
    Keeping it separate from :func:`parallelize_model` is required because DDP
    cannot wrap meta parameters, while TP and FSDP must be applied before those
    parameters are materialized.
    """
    state = get_parallel_state()
    if state.config.dp_mode != "ddp" or state.dp_cp_size <= 1:
        return model
    if module_has_meta_tensors(model):
        raise RuntimeError("replicated DDP requires a materialized model")
    if state.dp_cp_group is None:
        raise RuntimeError("replicated DDP requires a DP x CP process group")

    from torch.distributed.tensor import DTensor
    from torch.nn.parallel import DistributedDataParallel

    if any(isinstance(parameter, DTensor) for parameter in model.parameters()):
        raise NotImplementedError(
            "replicated DDP cannot wrap DTensor parameters; use FSDP2 for "
            "torch-native tensor parallelism, or mcore TP with the megatron engine"
        )

    parameter = next(model.parameters(), None)
    kwargs: dict[str, object] = {
        "process_group": state.dp_cp_group,
        # Wan's T2V path does not consume the trainable image patch embedding.
        # Without unused-parameter discovery, DDP leaves reducer buckets that
        # contain those parameters unfinished.  A one-step run then reaches
        # clipping before any following forward can raise DDP's usual error and
        # silently updates from rank-local gradients.
        "find_unused_parameters": True,
    }
    if parameter is not None and parameter.device.type == "cuda":
        kwargs["device_ids"] = [parameter.device.index]
        kwargs["output_device"] = parameter.device.index
    wrapped = DistributedDataParallel(model, **kwargs)
    wrapped._vforge_data_parallel_backend = "torch"
    return wrapped


def unwrap_replicated_data_parallel(model: nn.Module) -> nn.Module:
    """Return the logical model underneath vforge's replicated DDP wrapper."""
    if getattr(model, "_vforge_data_parallel_backend", None) in {"torch", "mcore"}:
        return model.module
    from torch.nn.parallel import DistributedDataParallel

    return model.module if isinstance(model, DistributedDataParallel) else model


def module_has_meta_tensors(model: nn.Module) -> bool:
    """Return whether any model parameter or buffer is still unmaterialized."""
    return any(tensor.is_meta for tensor in (*model.parameters(), *model.buffers()))


def _apply_fsdp2(
    model: nn.Module,
    config: FSDP2Config,
    *,
    extra_plan: Mapping[nn.Module, FSDPModulePlan] | None = None,
) -> nn.Module:
    """Wrap FSDP boundaries using resolved overrides, independent of EP semantics."""
    try:
        from torch.distributed._composable.fsdp import (
            MixedPrecisionPolicy,
            fully_shard,
            register_fsdp_forward_method,
        )
    except ImportError as exc:  # pragma: no cover - depends on torch build
        raise RuntimeError("FSDP2 requires torch.distributed._composable.fsdp") from exc

    if config.mixed_precision.enable and _uses_activation_checkpointing(model):
        ensure_activation_checkpoint_mixed_precision_compatibility()

    state = get_parallel_state()
    extra_plan = extra_plan or {}
    execution_roots = _resolve_fsdp_execution_roots(model)
    owned = {module for root in execution_roots for module in root.modules()}
    if any(module not in owned for module in extra_plan):
        raise ValueError("extra FSDP plan modules must belong to an execution root")

    def shard(module: nn.Module, kwargs: dict) -> None:
        module_plan = extra_plan.get(module)
        if module_plan is not None:
            kwargs = dict(kwargs)
            kwargs["mesh"] = module_plan.mesh
            kwargs["shard_placement_fn"] = lambda parameter: module_plan.shard_placement
            if module_plan.gradient_divide_factor is not None:
                # fully_shard's default mesh contract is [shard] or
                # [replicate, shard] (HSDP), independent of model parallelism.
                # This guard needs the shard size, excluding HSDP replicas.
                validate_gradient_divide_factor(
                    module,
                    factor=module_plan.gradient_divide_factor,
                    mesh_size=module_plan.mesh.size(-1),
                    mp_policy=kwargs.get("mp_policy"),
                )
        fully_shard(module, **kwargs)
        if module_plan is not None and module_plan.gradient_divide_factor is not None:
            module.set_gradient_divide_factor(module_plan.gradient_divide_factor)

    contracts_by_root = {
        id(execution_root): _resolve_fsdp_precision_contract(
            execution_root,
            config=config.mixed_precision,
        )
        for execution_root in execution_roots
    }
    keep_modules_by_root = {
        id(execution_root): _resolve_mixed_precision_preserve_modules(
            execution_root,
            enabled=config.mixed_precision.apply_model_overrides,
            paths=contracts_by_root[id(execution_root)].preserve_module_paths,
        )
        for execution_root in execution_roots
    }
    preserve_input_modules_by_root = {
        id(execution_root): _resolve_mixed_precision_preserve_input_modules(
            execution_root,
            enabled=config.mixed_precision.apply_model_overrides,
            paths=contracts_by_root[id(execution_root)].preserve_forward_input_paths,
        )
        for execution_root in execution_roots
    }
    keep_modules = tuple(module for modules in keep_modules_by_root.values() for module in modules)
    if config.mixed_precision.enable and config.mixed_precision.buffer_dtype is not None:
        _cast_floating_buffers(
            model,
            _resolve_dtype(config.mixed_precision.buffer_dtype),
            excluded_modules=keep_modules,
        )
    fsdp_kwargs: dict = {
        "mesh": state.fsdp_mesh,
        "reshard_after_forward": config.reshard_after_forward,
    }
    if config.mixed_precision.enable:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=_resolve_dtype(config.mixed_precision.param_dtype),
            reduce_dtype=_resolve_dtype(config.mixed_precision.reduce_dtype),
            output_dtype=_resolve_dtype(config.mixed_precision.output_dtype),
            cast_forward_inputs=config.mixed_precision.cast_forward_inputs,
        )
    for execution_root in execution_roots:
        execution_fsdp_kwargs = dict(fsdp_kwargs)
        preserve_input_kwargs = dict(execution_fsdp_kwargs)
        if config.mixed_precision.enable:
            preserve_input_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=_resolve_dtype(config.mixed_precision.param_dtype),
                reduce_dtype=_resolve_dtype(config.mixed_precision.reduce_dtype),
                output_dtype=_resolve_dtype(config.mixed_precision.output_dtype),
                cast_forward_inputs=False,
            )
        keep_modules = keep_modules_by_root[id(execution_root)]
        preserve_input_modules = preserve_input_modules_by_root[id(execution_root)]
        keep_module_ids = {id(module) for module in keep_modules}
        preserve_input_module_ids = {id(module) for module in preserve_input_modules}
        keep_tree_ids = {id(child) for module in keep_modules for child in module.modules()}
        preserve_input_leaves = tuple(
            module
            for module in preserve_input_modules
            if module is not execution_root
            and (id(module) not in keep_tree_ids or id(module) in keep_module_ids)
        )
        explicit_leaf_modules = _resolve_fsdp_leaf_modules(
            execution_root,
            paths=config.leaf_module_paths,
        )
        _validate_keep_modules_against_explicit_leaves(
            keep_modules,
            explicit_leaf_modules,
        )
        planned_modules = tuple(
            module
            for module in execution_root.modules()
            if module is not execution_root and module in extra_plan
        )
        target_names = set(config.basic_modules)
        target_names.update(getattr(execution_root, "_no_split_modules", None) or [])
        target_modules = tuple(
            module
            for _, module in _target_modules_bottom_up(execution_root, target_names)
            if module is not execution_root and id(module) not in keep_tree_ids
        )
        boundary_modules = _modules_bottom_up(
            execution_root,
            (
                *keep_modules,
                *explicit_leaf_modules,
                *preserve_input_leaves,
                *planned_modules,
                *target_modules,
            ),
        )
        for module in boundary_modules:
            if id(module) in keep_tree_ids:
                if not any(True for _ in module.parameters()):
                    continue
                module_kwargs = {
                    key: value for key, value in execution_fsdp_kwargs.items() if key != "mp_policy"
                }
            else:
                module_kwargs = dict(
                    preserve_input_kwargs
                    if id(module) in preserve_input_module_ids
                    else execution_fsdp_kwargs
                )
            shard(module, module_kwargs)

        root_kwargs = dict(
            preserve_input_kwargs
            if id(execution_root) in preserve_input_module_ids
            else execution_fsdp_kwargs
        )
        root_kwargs.pop("reshard_after_forward", None)
        shard(execution_root, root_kwargs)
        forward_methods = getattr(execution_root, "fsdp_forward_methods", None)
        if callable(forward_methods):
            for method_name in forward_methods():
                register_fsdp_forward_method(execution_root, method_name)

    return model


def _resolve_fsdp_leaf_modules(
    execution_root: nn.Module,
    *,
    paths: tuple[str, ...] = (),
    additional_modules: tuple[nn.Module, ...] = (),
) -> tuple[nn.Module, ...]:
    """Resolve configured leaves that must form independent FSDP groups."""
    leaves = (*_resolve_module_paths(execution_root, paths), *additional_modules)
    owned = {id(module) for module in execution_root.modules()}
    seen: set[int] = set()
    resolved: list[nn.Module] = []
    for module in leaves:
        if not isinstance(module, nn.Module):
            raise TypeError("FSDP leaf modules must be nn.Module instances")
        if module is execution_root or id(module) not in owned:
            raise ValueError("FSDP leaf modules must be descendants of the execution root")
        if id(module) in seen:
            continue
        descendants = {id(child) for child in module.modules()}
        if descendants & seen:
            raise ValueError("FSDP leaf module trees must not overlap")
        seen.update(descendants)
        resolved.append(module)
    return tuple(resolved)


def _modules_bottom_up(
    execution_root: nn.Module,
    modules: tuple[nn.Module, ...],
) -> tuple[nn.Module, ...]:
    """Deduplicate module boundaries and order nested groups child first."""
    depths = {id(module): name.count(".") + 1 for name, module in execution_root.named_modules()}
    seen: set[int] = set()
    unique = []
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        unique.append(module)
    return tuple(
        sorted(
            unique,
            key=lambda module: depths[id(module)],
            reverse=True,
        )
    )


def _resolve_fsdp_precision_contract(
    execution_root: nn.Module,
    *,
    config: MixedPrecisionConfig,
) -> FSDPPrecisionContract:
    resolver = getattr(execution_root, "get_fsdp_precision_contract", None)
    model_contract = resolver() if callable(resolver) else FSDPPrecisionContract()
    if not isinstance(model_contract, FSDPPrecisionContract):
        raise TypeError("get_fsdp_precision_contract() must return FSDPPrecisionContract")
    if config.contract_mode == "replace":
        return FSDPPrecisionContract(
            preserve_module_paths=config.preserve_module_paths,
            preserve_forward_input_paths=config.preserve_forward_input_paths,
        )
    # A model that declares no preserve paths still declares which modules
    # Diffusers keeps in fp32, and those are the same modules.
    model_paths = model_contract.preserve_module_paths or keep_in_fp32_module_paths(execution_root)
    return FSDPPrecisionContract(
        preserve_module_paths=_deduplicate_paths((*model_paths, *config.preserve_module_paths)),
        preserve_forward_input_paths=_deduplicate_paths(
            (
                *model_contract.preserve_forward_input_paths,
                *config.preserve_forward_input_paths,
            )
        ),
    )


def _resolve_mixed_precision_preserve_modules(
    execution_root: nn.Module,
    *,
    enabled: bool = True,
    paths: tuple[str, ...] = (),
) -> tuple[nn.Module, ...]:
    """Resolve module trees excluded from the FSDP mixed-precision policy.

    Only a module can be excluded, because FSDP expresses the exception by
    giving it a group of its own without an ``mp_policy``. A path that names an
    individual weight is reported and skipped; a path that names nothing at all
    is still an error.
    """
    if not enabled:
        return ()
    module_paths: list[str] = []
    skipped: list[str] = []
    for path in paths:
        weight_paths = _matching_parameter_paths(execution_root, path)
        if weight_paths:
            skipped.extend(weight_paths)
            continue
        module_paths.append(path)
    _report_skipped_preserve_paths(tuple(skipped))
    resolved = _resolve_module_paths(execution_root, tuple(module_paths))
    module_trees: list[set[int]] = []
    for module in resolved:
        if module is execution_root:
            raise ValueError("the execution root cannot be a preserve module")
        descendants = {id(child) for child in module.modules()}
        if any(descendants & other for other in module_trees):
            raise ValueError("mixed-precision preserve module trees must not overlap")
        module_trees.append(descendants)
        floating_dtypes = {
            parameter.dtype for parameter in module.parameters() if parameter.is_floating_point()
        }
        if len(floating_dtypes) > 1:
            raise ValueError(
                "mixed-precision preserve modules must have one floating "
                f"parameter dtype; got {sorted(map(str, floating_dtypes))}"
            )
    return resolved


def _matching_parameter_paths(execution_root: nn.Module, pattern: str) -> tuple[str, ...]:
    """Individual weights a preserve path reaches, when no module answers to it."""
    _validate_path_pattern(pattern)
    if pattern == ".":
        return ()
    module_names = {name for name, _ in execution_root.named_modules() if name}
    if _path_pattern_is_glob(pattern):
        if any(_match_path_pattern(name, pattern) for name in module_names):
            return ()
        return tuple(
            name
            for name, _ in execution_root.named_parameters()
            if name not in module_names and _match_path_pattern(name, pattern)
        )
    if pattern in module_names:
        return ()
    parent_path, separator, name = pattern.rpartition(".")
    owner = execution_root
    if separator:
        try:
            owner = execution_root.get_submodule(parent_path)
        except AttributeError:
            return ()
    if isinstance(owner._parameters.get(name), nn.Parameter):
        return (pattern,)
    return ()


def _report_skipped_preserve_paths(paths: tuple[str, ...]) -> None:
    if not paths:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    print(
        "VFORGE_PRECISION_PRESERVE_SKIPPED mixed-precision preserve applies to "
        "modules only, so these paths keep the policy dtype: " + ", ".join(sorted(paths)),
        flush=True,
    )


def _resolve_mixed_precision_preserve_input_modules(
    execution_root: nn.Module,
    *,
    enabled: bool = True,
    paths: tuple[str, ...] = (),
) -> tuple[nn.Module, ...]:
    """Resolve FSDP groups whose non-parameter inputs must retain model dtypes."""
    if not enabled:
        return ()
    return _resolve_module_paths(execution_root, paths, allow_root=True)


def _resolve_module_paths(
    execution_root: nn.Module,
    paths: tuple[str, ...],
    *,
    allow_root: bool = False,
) -> tuple[nn.Module, ...]:
    resolved: list[nn.Module] = []
    for path in paths:
        for concrete in _expand_path_pattern(execution_root, path):
            if concrete == ".":
                if not allow_root:
                    raise ValueError("the execution root is not valid for this module path")
                module = execution_root
            else:
                try:
                    module = execution_root.get_submodule(concrete)
                except AttributeError as exc:
                    raise ValueError(
                        f"module path {path!r} does not exist under the execution root"
                    ) from exc
            if module not in resolved:
                resolved.append(module)
    return tuple(resolved)


def _path_pattern_is_glob(pattern: str) -> bool:
    return any(part == "*" for part in pattern.split("."))


def _validate_path_pattern(pattern: str) -> None:
    if pattern == ".":
        return
    for part in pattern.split("."):
        if "*" in part and part != "*":
            raise ValueError(
                f"mixed-precision path {pattern!r} may use '*' only as a whole "
                "segment (for example 'blocks.*.norm1')"
            )


def _match_path_pattern(name: str, pattern: str) -> bool:
    name_parts = name.split(".")
    pattern_parts = pattern.split(".")
    if len(name_parts) != len(pattern_parts):
        return False
    return all(
        part == "*" or part == name_part
        for part, name_part in zip(pattern_parts, name_parts, strict=True)
    )


def _expand_path_pattern(execution_root: nn.Module, pattern: str) -> tuple[str, ...]:
    """Expand ``blocks.*.norm1`` into concrete names. ``*`` is one dotted segment."""
    _validate_path_pattern(pattern)
    if pattern == "." or not _path_pattern_is_glob(pattern):
        return (pattern,)
    matched = tuple(
        name
        for name, _ in execution_root.named_modules()
        if name and _match_path_pattern(name, pattern)
    )
    if not matched:
        raise ValueError(
            f"module path {pattern!r} does not match any module under the execution root"
        )
    return matched


def _deduplicate_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(paths))


def _validate_module_paths(
    field_name: str,
    paths: tuple[str, ...],
    *,
    allow_root: bool = False,
) -> None:
    if any(not path for path in paths):
        raise ValueError(f"mixed-precision {field_name} must not contain empty paths")
    if not allow_root and "." in paths:
        raise ValueError(f"mixed-precision {field_name} cannot contain the execution root")
    if len(set(paths)) != len(paths):
        raise ValueError(f"mixed-precision {field_name} paths must be unique")
    for path in paths:
        _validate_path_pattern(path)


def _validate_keep_modules_against_explicit_leaves(
    keep_modules: tuple[nn.Module, ...],
    explicit_leaf_modules: tuple[nn.Module, ...],
) -> None:
    keep_ids = {id(module) for module in keep_modules}
    keep_tree_ids = {id(module) for keep in keep_modules for module in keep.modules()}
    for leaf in explicit_leaf_modules:
        if id(leaf) in keep_tree_ids and id(leaf) not in keep_ids:
            raise ValueError(
                "an explicit FSDP leaf cannot be nested inside a mixed-precision preserve module"
            )


def _resolve_fsdp_execution_roots(model: nn.Module) -> tuple[nn.Module, ...]:
    resolver = getattr(model, "fsdp_execution_roots", None)
    roots = tuple(resolver()) if callable(resolver) else (model,)
    if not roots:
        raise ValueError("FSDP execution roots must not be empty")

    owned = {id(module) for module in model.modules()}
    module_sets: list[set[int]] = []
    for root in roots:
        if not isinstance(root, nn.Module):
            raise TypeError("FSDP execution roots must be nn.Module instances")
        if id(root) not in owned:
            raise ValueError("FSDP execution root is not owned by the lifecycle root")
        descendants = {id(module) for module in root.modules()}
        if any(descendants & other for other in module_sets):
            raise ValueError("FSDP execution roots must have disjoint module trees")
        module_sets.append(descendants)
    return roots


def _cast_floating_buffers(
    model: nn.Module,
    dtype: torch.dtype | None,
    *,
    excluded_modules: tuple[nn.Module, ...] = (),
) -> None:
    if dtype is None:
        return
    excluded_buffer_ids = {id(buffer) for module in excluded_modules for buffer in module.buffers()}
    with torch.no_grad():
        for buffer in model.buffers():
            if (
                id(buffer) not in excluded_buffer_ids
                and buffer.is_floating_point()
                and buffer.dtype != dtype
            ):
                buffer.data = buffer.data.to(dtype=dtype)


def _uses_activation_checkpointing(model: nn.Module) -> bool:
    return any(bool(getattr(module, "gradient_checkpointing", False)) for module in model.modules())


def _target_modules_bottom_up(
    model: nn.Module,
    target_class_names: set[str],
) -> list[tuple[str, nn.Module]]:
    if not target_class_names:
        return []
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ in target_class_names
    ]
    return sorted(matches, key=lambda item: item[0].count("."), reverse=True)


def _resolve_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype alias: {name!r}") from exc


def _validate_dtype_name(field_name: str, name: str | None) -> None:
    if name is None:
        return
    if name not in {"bfloat16", "float32", "float16", "bf16", "fp32", "fp16"}:
        raise ValueError(
            f"invalid mixed precision {field_name}={name!r}; "
            "expected one of 'bfloat16', 'float32', 'float16', "
            "'bf16', 'fp32', 'fp16', or None"
        )
