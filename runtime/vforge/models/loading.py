"""Reusable pretrained-model loading and state-dict key utilities."""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn

ShapeMismatchResolver = Callable[
    [str, torch.Tensor, torch.Tensor],
    torch.Tensor | None,
]
MissingKeyResolver = Callable[
    [str, torch.Tensor, Mapping[str, Any]],
    torch.Tensor | None,
]
MissingAfterLoadInitializer = Callable[[nn.Module, frozenset[str]], None]


@dataclass(frozen=True)
class CompatibleLoadCallbacks:
    """Application-owned policies for reconstructing incompatible weights."""

    resolve_shape_mismatch: ShapeMismatchResolver | None = None
    resolve_missing_key: MissingKeyResolver | None = None


@dataclass(frozen=True)
class PretrainedWeightSource:
    """Framework-owned description of pretrained model weights.

    ``resolve_shape_mismatch`` lets a plugin rebuild a checkpoint tensor whose
    shape no longer matches the model, e.g. a fork that widens a conv input
    axis. It receives ``(model_key, source_tensor, target_tensor)`` and must
    return a plain tensor with the target's global shape, or ``None`` to leave
    the key unloaded. Under FSDP the target may be a ``DTensor``, so build the
    result from ``source_tensor`` and read only ``target_tensor.shape``.

    ``resolve_missing_key`` reconstructs model keys absent from the checkpoint
    after compatible source tensors have loaded. It is never called for a key
    the checkpoint contains, so published/new checkpoint values always win.
    A successfully reconstructed key satisfies strict loading; unresolved keys
    still fail strict mode unless listed in ``allowed_missing_model_keys``.

    ``model_path`` narrows the load to one submodule of the training root. A
    root assembled from several published checkpoints (Wan2.2 A14B's high- and
    low-noise experts) needs one source per tower, and without narrowing each
    load would see the other tower's parameters as missing keys.

    ``state_dict_key`` selects one top-level state dict from a PyTorch ``.pt``
    or ``.pth`` payload. It composes with ``model_path``, allowing one nested
    checkpoint to populate one tower without changing either key space.

    ``source_prefix`` removes a uniform wrapper prefix from an external state
    dict before ``model_prefix`` maps it into the target module namespace.

    ``initialize_missing_after_load`` runs after compatible checkpoint tensors
    have reached the live model. It is reserved for a new parameter whose
    initializer depends on a loaded parameter, and receives the exact missing
    model-key set so an already trained tensor is never overwritten.
    """

    path: str
    subfolder: str = "transformer"
    model_path: str = ""
    model_prefix: str = ""
    loading_mode: str = "auto"
    dtype: torch.dtype | None = None
    allowed_missing_model_keys: frozenset[str] = frozenset()
    initialize_missing: Callable[[nn.Module], None] | None = None
    initialize_missing_after_load: MissingAfterLoadInitializer | None = None
    fallback_model_factory: Callable[[], nn.Module] | None = None
    stage_to_local: bool = False
    resolve_shape_mismatch: ShapeMismatchResolver | None = None
    state_dict_key: str = ""
    source_prefix: str = ""
    resolve_missing_key: MissingKeyResolver | None = None

    def __post_init__(self) -> None:
        if self.loading_mode not in {
            "auto",
            "distributed",
            "rank0_broadcast",
            "replicated",
        }:
            raise ValueError(f"unsupported pretrained loading mode {self.loading_mode!r}")


def stage_pretrained_weights(
    source: PretrainedWeightSource,
) -> PretrainedWeightSource:
    """Stage an explicitly mapped S3 safetensors source once per node."""
    try:
        from vforge.utils.s3_staging import resolve_s3_source
    except ImportError:
        # S3 routing is optional deployment glue; without it no path is
        # S3-mapped and every source is already node-local.
        return source

    if resolve_s3_source(source.path) is None and not source.stage_to_local:
        return source

    from vforge.utils.node_staging import NodeStagingConfig, stage_on_node

    config = NodeStagingConfig()
    if os.path.isdir(source.path):
        result = stage_on_node(source.path, config=config)
        return replace(source, path=result.destination)
    if os.path.isfile(source.path):
        parent = os.path.dirname(source.path) or "."
        basename = os.path.basename(source.path)
        result = stage_on_node(
            parent,
            config=config,
            relative_paths=(basename,),
            require_checkpoint_metadata=False,
        )
        return replace(
            source,
            path=os.path.join(result.destination, basename),
        )
    raise FileNotFoundError(source.path)


def _keep_in_fp32_name_parts(module: nn.Module) -> tuple[str, ...]:
    """Union Diffusers ``_keep_in_fp32_modules`` fragments declared in ``module``."""
    parts: list[str] = []
    seen: set[str] = set()
    for submodule in module.modules():
        names = getattr(submodule, "_keep_in_fp32_modules", None)
        if not names:
            continue
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            parts.append(str(name))
    return tuple(parts)


def _keep_in_fp32_match(path: str, name_parts: tuple[str, ...]) -> bool:
    leaf = path.rpartition(".")[2]
    return any(part in leaf for part in name_parts)


def keep_in_fp32_module_paths(module: nn.Module) -> tuple[str, ...]:
    """Expand ``_keep_in_fp32_modules`` fragments into concrete submodule paths.

    One declaration then decides both Parameter storage dtype and the FSDP
    mixed-precision exemptions. A submodule is a target when the last segment
    of its path carries a declared fragment. Nested hits collapse onto the
    outermost path, since exempting a parent already covers everything below
    it. Fragments naming an individual weight rather than a submodule reach
    storage dtype only, because FSDP has no group to exempt for them. A tree
    that declares nothing is exempt from nothing.
    """
    name_parts = _keep_in_fp32_name_parts(module)
    if not name_parts:
        return ()
    candidates = [
        name for name, _ in module.named_modules() if name and _keep_in_fp32_match(name, name_parts)
    ]
    return tuple(
        path for path in candidates if not any(path.startswith(f"{other}.") for other in candidates)
    )


def cast_module_parameters(
    module: nn.Module,
    dtype: torch.dtype,
) -> nn.Module:
    """Cast floating Parameter storage to ``dtype``.

    Names matching ``_keep_in_fp32_modules`` (Diffusers substring match, unioned
    across the module tree) stay float32. Model-authored buffers are unchanged.
    """
    keep_fp32 = _keep_in_fp32_name_parts(module)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if not parameter.is_floating_point():
                continue
            target_dtype = torch.float32 if any(part in name for part in keep_fp32) else dtype
            if parameter.dtype != target_dtype:
                parameter.data = parameter.data.to(dtype=target_dtype)
    return module


def load_pt_state_dict(
    path: str,
    *,
    state_dict_key: str = "",
) -> dict[str, Any]:
    """Load a flat state dict, optionally selected from a top-level PT key."""
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if state_dict_key:
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"PT checkpoint {path!r} is not a mapping; "
                f"cannot select state_dict_key={state_dict_key!r}"
            )
        if state_dict_key not in payload:
            raise KeyError(f"PT checkpoint {path!r} has no key {state_dict_key!r}")
        payload = payload[state_dict_key]
    if not isinstance(payload, Mapping):
        raise TypeError(f"PT checkpoint state at {path!r} must be a mapping")
    non_string_keys = [key for key in payload if not isinstance(key, str)]
    if non_string_keys:
        raise TypeError(
            f"PT checkpoint state at {path!r} contains non-string keys: {non_string_keys[:3]!r}"
        )
    return dict(payload)


def _huggingface_safetensor_keys(source_path: str) -> set[str]:
    """Return all tensor keys advertised by a Hugging Face safetensors folder."""
    index_paths = [
        os.path.join(source_path, name)
        for name in os.listdir(source_path)
        if name.endswith(".safetensors.index.json")
    ]
    if index_paths:
        keys: set[str] = set()
        for index_path in index_paths:
            with open(index_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            keys.update(dict(payload.get("weight_map") or {}))
        return keys

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("discovering Hugging Face safetensors keys requires safetensors") from exc
    keys = set()
    for name in os.listdir(source_path):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(source_path, name), framework="pt") as handle:
            keys.update(handle.keys())
    return keys


def load_safetensors_tensors(
    source_path: str,
    keys: set[str],
) -> dict[str, torch.Tensor]:
    """Read only requested tensors from a safetensors file or HF directory.

    Sharded Hugging Face directories are resolved through their index when
    available. Without an index, shard metadata is scanned, but tensor payloads
    are materialized only for shards and keys that match the request.
    """
    requested = set(keys)
    if not requested:
        return {}

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("selective loading requires safetensors") from exc

    if os.path.isfile(source_path):
        if not source_path.endswith(".safetensors"):
            raise ValueError(f"selective loading only supports safetensors: {source_path}")
        with safe_open(source_path, framework="pt", device="cpu") as handle:
            available = requested.intersection(handle.keys())
            return {key: handle.get_tensor(key) for key in available}

    if not os.path.isdir(source_path):
        raise FileNotFoundError(source_path)

    shard_names = sorted(name for name in os.listdir(source_path) if name.endswith(".safetensors"))
    if not shard_names:
        raise FileNotFoundError(f"no safetensors files found under {source_path}")

    index_paths = sorted(
        os.path.join(source_path, name)
        for name in os.listdir(source_path)
        if name.endswith(".safetensors.index.json")
    )
    key_to_shard: dict[str, str] = {}
    for index_path in index_paths:
        with open(index_path, encoding="utf-8") as handle:
            weight_map = dict(json.load(handle).get("weight_map") or {})
        for key in requested:
            shard_name = weight_map.get(key)
            if shard_name is not None:
                key_to_shard[key] = shard_name

    result: dict[str, torch.Tensor] = {}
    if key_to_shard:
        shard_to_keys: dict[str, set[str]] = {}
        for key, shard_name in key_to_shard.items():
            shard_to_keys.setdefault(shard_name, set()).add(key)
        for shard_name, shard_keys in shard_to_keys.items():
            shard_path = os.path.join(source_path, shard_name)
            with safe_open(
                shard_path,
                framework="pt",
                device="cpu",
            ) as handle:
                available = (requested | shard_keys).intersection(handle.keys())
                result.update({key: handle.get_tensor(key) for key in available})

    remaining = requested - set(result)
    if remaining:
        indexed_shards = set(key_to_shard.values())
        for shard_name in shard_names:
            if shard_name in indexed_shards:
                continue
            shard_path = os.path.join(source_path, shard_name)
            with safe_open(
                shard_path,
                framework="pt",
                device="cpu",
            ) as handle:
                available = remaining.intersection(handle.keys())
                result.update({key: handle.get_tensor(key) for key in available})
            remaining -= set(result)
            if not remaining:
                break
    return result


def load_huggingface_safetensors_distributed(
    model: nn.Module,
    pretrained_path: str,
    *,
    subfolder: str,
    model_prefix: str = "",
    allowed_missing_model_keys: set[str] | None = None,
    resolve_shape_mismatch: ShapeMismatchResolver | None = None,
    resolve_missing_key: MissingKeyResolver | None = None,
    strict: bool = False,
) -> frozenset[str]:
    """Load Hugging Face safetensors directly into a sharded model."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    try:
        from torch.distributed.checkpoint import HuggingFaceStorageReader
    except ImportError as exc:
        raise ImportError(
            "distributed Hugging Face loading requires a PyTorch build with "
            "HuggingFaceStorageReader"
        ) from exc

    model_state = get_model_state_dict(model)
    source_to_model_key = {
        (key[len(model_prefix) :] if model_prefix and key.startswith(model_prefix) else key): key
        for key in model_state
    }
    source_path = os.path.join(pretrained_path, subfolder)
    if not any(name.endswith(".safetensors") for name in os.listdir(source_path)):
        raise FileNotFoundError(f"no safetensors files found under {source_path}")
    reader = HuggingFaceStorageReader(path=source_path)
    source_metadata = reader.read_metadata().state_dict_metadata
    available_source_keys = set(source_metadata)
    missing_model_keys = {
        model_key
        for source_key, model_key in source_to_model_key.items()
        if source_key not in available_source_keys
    }
    allowed_missing = allowed_missing_model_keys or set()
    unexpected_source = available_source_keys - set(source_to_model_key)
    mismatched: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for source_key, model_key in source_to_model_key.items():
        if source_key not in source_metadata:
            continue
        source_size = getattr(source_metadata[source_key], "size", None)
        target_shape = getattr(model_state[model_key], "shape", None)
        if source_size is not None and target_shape is not None:
            source_shape = tuple(source_size)
            model_shape = tuple(target_shape)
            if source_shape != model_shape:
                mismatched[source_key] = (source_shape, model_shape)
    resolved_full = _resolve_mismatched_source_tensors(
        source_path,
        mismatched,
        source_to_model_key=source_to_model_key,
        model_state=model_state,
        resolve_shape_mismatch=resolve_shape_mismatch,
    )
    unresolved_mismatched = {
        source_key: shapes
        for source_key, shapes in mismatched.items()
        if source_to_model_key[source_key] not in resolved_full
    }
    if strict and (unexpected_source or unresolved_mismatched):
        raise RuntimeError(
            "Hugging Face safetensors do not strictly match model: "
            "missing=pending missing-key resolution, "
            f"unexpected={sorted(unexpected_source)}, "
            f"shape_mismatch={unresolved_mismatched}"
        )
    source_state = {
        source_key: model_state[model_key]
        for source_key, model_key in source_to_model_key.items()
        if source_key in available_source_keys and source_key not in mismatched
    }

    dcp.load(
        state_dict=source_state,
        storage_reader=reader,
    )
    set_model_state_dict(
        model,
        model_state_dict={
            model_key: source_state[source_key]
            for source_key, model_key in source_to_model_key.items()
            if source_key in source_state
        },
        options=StateDictOptions(strict=False),
    )
    if resolved_full:
        # Rebuilt tensors are full, rank-replicated, and never sharded by the
        # reader, so they need a separate full-state-dict assignment.
        set_model_state_dict(
            model,
            model_state_dict=resolved_full,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=False,
                strict=False,
            ),
        )
    refreshed_state = get_model_state_dict(model)
    resolved_missing = _resolve_missing_model_tensors(
        missing_model_keys,
        model_state=refreshed_state,
        resolve_missing_key=resolve_missing_key,
    )
    if resolved_missing:
        set_model_state_dict(
            model,
            model_state_dict=resolved_missing,
            options=StateDictOptions(strict=False),
        )
    unresolved_missing = missing_model_keys - set(resolved_missing) - allowed_missing
    if strict and unresolved_missing:
        raise RuntimeError(
            "Hugging Face safetensors do not strictly match model: "
            f"missing={sorted(unresolved_missing)}, unexpected=[], shape_mismatch={{}}"
        )
    if (
        unresolved_missing
        or unexpected_source
        or unresolved_mismatched
        or resolved_missing
        or resolved_full
    ):
        warnings.warn(
            "compatible safetensors load skipped incompatible state: "
            f"missing={sorted(unresolved_missing)}, "
            f"unexpected={sorted(unexpected_source)}, "
            f"shape_mismatch={unresolved_mismatched}, "
            f"resolved_missing={sorted(resolved_missing)}, "
            f"resolved_shape_mismatch={sorted(resolved_full)}",
            RuntimeWarning,
            stacklevel=2,
        )
    return frozenset(missing_model_keys)


def _resolve_missing_model_tensors(
    missing_keys: set[str],
    *,
    model_state: Mapping[str, Any],
    resolve_missing_key: MissingKeyResolver | None,
) -> dict[str, torch.Tensor]:
    if resolve_missing_key is None:
        return {}
    resolved: dict[str, torch.Tensor] = {}
    available = dict(model_state)
    for key in sorted(missing_keys):
        target = model_state[key]
        if not isinstance(target, torch.Tensor):
            continue
        value = resolve_missing_key(key, target, available)
        if value is None:
            continue
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                "missing-key callback returned the wrong shape: "
                f"key={key!r}, expected={tuple(target.shape)}, "
                f"actual={tuple(value.shape)}"
            )
        resolved[key] = value
        available[key] = value
    return resolved


def _resolve_mismatched_source_tensors(
    source_path: str,
    mismatched: dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    source_to_model_key: dict[str, str],
    model_state: Mapping[str, Any],
    resolve_shape_mismatch: ShapeMismatchResolver | None,
) -> dict[str, torch.Tensor]:
    if resolve_shape_mismatch is None or not mismatched:
        return {}
    raw = load_safetensors_tensors(source_path, set(mismatched))
    resolved_full: dict[str, torch.Tensor] = {}
    for source_key in sorted(mismatched):
        source_value = raw.get(source_key)
        if source_value is None:
            continue
        model_key = source_to_model_key[source_key]
        target_value = model_state[model_key]
        resolved = resolve_shape_mismatch(model_key, source_value, target_value)
        if resolved is None:
            continue
        target_shape = tuple(target_value.shape)
        if tuple(resolved.shape) != target_shape:
            raise ValueError(
                "shape-mismatch callback returned the wrong shape: "
                f"key={model_key!r}, expected={target_shape}, "
                f"actual={tuple(resolved.shape)}"
            )
        resolved_full[model_key] = resolved
    return resolved_full


def load_pretrained_weights(
    model: nn.Module,
    source: PretrainedWeightSource,
    *,
    strict: bool = False,
) -> None:
    """Load pretrained weights after model sharding and materialization."""
    model = model.get_submodule(source.model_path) if source.model_path else model
    if source.initialize_missing is not None:
        source.initialize_missing(model)
    mode = source.loading_mode
    is_pt = os.path.splitext(source.path)[1].lower() in {".pt", ".pth"}
    if not is_pt and mode in {"auto", "distributed"}:
        try:
            missing_model_keys = load_huggingface_safetensors_distributed(
                model,
                source.path,
                subfolder=source.subfolder,
                model_prefix=source.model_prefix,
                allowed_missing_model_keys=set(source.allowed_missing_model_keys),
                resolve_shape_mismatch=source.resolve_shape_mismatch,
                resolve_missing_key=source.resolve_missing_key,
                strict=strict,
            )
            if source.initialize_missing_after_load is not None:
                source.initialize_missing_after_load(model, missing_model_keys)
            return
        except (ImportError, FileNotFoundError):
            if mode == "distributed":
                raise

    factory = source.fallback_model_factory
    if not is_pt and factory is None:
        raise RuntimeError(
            "pretrained distributed loading was unavailable and no fallback "
            "model factory was supplied"
        )
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    broadcast = mode in {"auto", "rank0_broadcast"}
    load_here = not distributed or not broadcast or torch.distributed.get_rank() == 0
    state: dict[str, Any] = {}
    if load_here:
        # A supplied factory may own PT key mapping, such as selecting one expert.
        if is_pt and factory is None:
            loaded_state = load_pt_state_dict(
                source.path,
                state_dict_key=source.state_dict_key,
            )
        else:
            assert factory is not None
            loaded = factory()
            loaded_state = dict(loaded.state_dict())
            del loaded
        if source.source_prefix:
            loaded_state = strip_state_dict_prefix(
                loaded_state,
                source.source_prefix,
            )
        state = {source.model_prefix + key: value for key, value in loaded_state.items()}
        missing_model_keys = frozenset(set(model.state_dict()) - set(state))
        state = filter_compatible_state_dict(
            model,
            state,
            source.path,
            CompatibleLoadCallbacks(
                resolve_shape_mismatch=source.resolve_shape_mismatch,
                resolve_missing_key=source.resolve_missing_key,
            ),
            strict=strict,
        )
    else:
        missing_model_keys = frozenset()
    if distributed and broadcast:
        payload: list[object] = [missing_model_keys if load_here else None]
        torch.distributed.broadcast_object_list(payload, src=0)
        broadcast_missing = payload[0]
        if not isinstance(broadcast_missing, (set, frozenset, list, tuple)):
            raise RuntimeError("rank 0 did not broadcast pretrained missing keys")
        missing_model_keys = frozenset(str(key) for key in broadcast_missing)
    if distributed:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        set_model_state_dict(
            model,
            model_state_dict=state,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=broadcast,
                strict=strict,
            ),
        )
    else:
        incompatible = model.load_state_dict(state, strict=False)
        missing_model_keys = frozenset(incompatible.missing_keys)
        if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            raise RuntimeError(
                "pretrained weights did not match model structure: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
    if source.initialize_missing_after_load is not None:
        source.initialize_missing_after_load(model, missing_model_keys)


def load_pretrained_optimizer_main_param_state(
    model: nn.Module,
    sources: Sequence[PretrainedWeightSource],
) -> dict[str, torch.Tensor]:
    """Build the FP32 model state consumed by MCore ``reload_model_params``.

    Model loading normally writes an FP32 Hugging Face tensor into a BF16
    destination. MCore's distributed optimizer is constructed afterwards, so
    cloning that destination into its FP32 master shard cannot recover the
    discarded bits. This helper reopens only the parameter tensors requested by
    the live model, preserves their checkpoint dtype, applies the same task-owned
    shape resolver as regular loading, and describes TP-local tensors using the
    physical parameter layout expected by MCore.

    Missing published tensors use the already initialized live parameter. This
    covers task-authored additions such as a zero condition tail while ensuring
    that MCore receives one tensor for every parameter it owns.
    """
    from vforge.distributed.parallel_state import get_parallel_state

    parallel = get_parallel_state()
    combined: dict[str, torch.Tensor] = {}
    for source in sources:
        target = model.get_submodule(source.model_path) if source.model_path else model
        named_parameters = dict(target.named_parameters())
        source_to_model_key = {
            (
                key[len(source.model_prefix) :]
                if source.model_prefix and key.startswith(source.model_prefix)
                else key
            ): key
            for key in named_parameters
        }
        source_path = os.path.join(source.path, source.subfolder)
        # Regular loading may legitimately use the task's fallback factory
        # when a source is not a safetensors directory. Preserve that behavior:
        # the live parameter is the best available initializer in that case.
        available = (
            _huggingface_safetensor_keys(source_path) if os.path.isdir(source_path) else set()
        )
        requested = set(source_to_model_key).intersection(available)
        raw = load_safetensors_tensors(source_path, requested)
        resolved_for_source: dict[str, torch.Tensor] = {}

        for source_key, model_key in source_to_model_key.items():
            parameter = named_parameters[model_key]
            value = raw.get(source_key)
            if value is not None:
                value = _tp_local_pretrained_tensor(
                    value,
                    parameter,
                    tp_size=parallel.config.tp_size,
                    tp_rank=parallel.tp_rank,
                )
                if tuple(value.shape) != tuple(parameter.shape):
                    resolver = source.resolve_shape_mismatch
                    value = resolver(model_key, value, parameter) if resolver is not None else None
            if value is None and source.resolve_missing_key is not None:
                value = source.resolve_missing_key(
                    model_key,
                    parameter,
                    resolved_for_source,
                )
            if value is None:
                value = parameter.detach().to(device="cpu", dtype=torch.float32)
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    "pretrained optimizer-main parameter has the wrong shape: "
                    f"key={model_key!r}, expected={tuple(parameter.shape)}, "
                    f"actual={tuple(value.shape)}"
                )
            state_key = f"{source.model_path}.{model_key}" if source.model_path else model_key
            if state_key in combined:
                raise ValueError(f"duplicate pretrained optimizer-main key {state_key!r}")
            combined[state_key] = value
            resolved_for_source[model_key] = value
    return combined


def _tp_local_pretrained_tensor(
    value: torch.Tensor,
    parameter: nn.Parameter,
    *,
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    """Return the contiguous TP slice matching one physical MCore parameter."""
    if tuple(value.shape) == tuple(parameter.shape):
        return value
    if not bool(getattr(parameter, "tensor_model_parallel", False)):
        return value
    partition_dim = int(getattr(parameter, "partition_dim", -1))
    partition_stride = int(getattr(parameter, "partition_stride", 1))
    if partition_stride != 1:
        raise NotImplementedError(
            "pretrained FP32 master loading only supports TP partition_stride=1; "
            f"got {partition_stride}"
        )
    if partition_dim < 0 or partition_dim >= value.ndim:
        raise ValueError(
            f"invalid TP partition_dim={partition_dim} for tensor shape {tuple(value.shape)}"
        )
    expected = list(parameter.shape)
    expected[partition_dim] *= int(tp_size)
    if tuple(value.shape) != tuple(expected):
        return value
    local_size = int(parameter.shape[partition_dim])
    return value.narrow(partition_dim, int(tp_rank) * local_size, local_size).contiguous()


def filter_compatible_state_dict(
    model: nn.Module,
    source_state: dict[str, Any],
    source: str,
    callbacks: CompatibleLoadCallbacks | None = None,
    target_state: dict[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    target_state = model.state_dict() if target_state is None else target_state
    source_state = dict(source_state)
    missing = set(target_state) - set(source_state)
    unexpected = sorted(set(source_state) - set(target_state))
    mismatched: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    compatible: dict[str, Any] = {}
    resolved_shape: list[str] = []
    resolved_missing: list[str] = []
    for key in set(target_state).intersection(source_state):
        source_value = source_state[key]
        target_value = target_state[key]
        if not isinstance(source_value, torch.Tensor) or not isinstance(
            target_value,
            torch.Tensor,
        ):
            if type(source_value) is type(target_value):
                compatible[key] = source_value
            continue
        source_shape = tuple(source_value.shape)
        target_shape = tuple(target_value.shape)
        if source_shape == target_shape:
            compatible[key] = source_value
        else:
            mismatched[key] = (source_shape, target_shape)
            resolver = callbacks.resolve_shape_mismatch if callbacks is not None else None
            resolved = resolver(key, source_value, target_value) if resolver is not None else None
            if resolved is not None:
                if tuple(resolved.shape) != target_shape:
                    raise ValueError(
                        "shape-mismatch callback returned the wrong shape: "
                        f"key={key!r}, expected={target_shape}, "
                        f"actual={tuple(resolved.shape)}"
                    )
                compatible[key] = resolved
                resolved_shape.append(key)
    missing_resolver = callbacks.resolve_missing_key if callbacks is not None else None
    if missing_resolver is not None:
        for key in sorted(missing):
            target_value = target_state[key]
            if not isinstance(target_value, torch.Tensor):
                continue
            resolved = missing_resolver(key, target_value, compatible)
            if resolved is None:
                continue
            if tuple(resolved.shape) != tuple(target_value.shape):
                raise ValueError(
                    "missing-key callback returned the wrong shape: "
                    f"key={key!r}, expected={tuple(target_value.shape)}, "
                    f"actual={tuple(resolved.shape)}"
                )
            compatible[key] = resolved
            resolved_missing.append(key)
            missing.remove(key)
    unresolved_mismatched = {
        key: shapes for key, shapes in mismatched.items() if key not in resolved_shape
    }
    if strict and (missing or unexpected or unresolved_mismatched):
        raise RuntimeError(
            "pretrained weights do not strictly match model: "
            f"missing={sorted(missing)}, unexpected={unexpected}, "
            f"shape_mismatch={unresolved_mismatched}"
        )
    if (missing or unexpected or mismatched or resolved_missing or resolved_shape) and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    ):
        warnings.warn(
            "compatible pretrained load skipped incompatible state: "
            f"source={source!r}, missing={sorted(missing)}, "
            f"unexpected={unexpected}, "
            f"shape_mismatch={unresolved_mismatched}, "
            f"resolved_missing={resolved_missing}, "
            f"resolved_shape_mismatch={resolved_shape}",
            RuntimeWarning,
            stacklevel=2,
        )
    return compatible


def strip_state_dict_prefix(
    state: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    """Strip a wrapper prefix when every state-dict key uses it."""
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix) :]: value for key, value in state.items()}
    return state


def add_state_dict_prefix(
    state: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    """Add a wrapper prefix to an external, unwrapped state dict."""
    if state and not any(key.startswith(prefix) for key in state):
        return {prefix + key: value for key, value in state.items()}
    return state


__all__ = [
    "CompatibleLoadCallbacks",
    "PretrainedWeightSource",
    "add_state_dict_prefix",
    "cast_module_parameters",
    "filter_compatible_state_dict",
    "keep_in_fp32_module_paths",
    "load_huggingface_safetensors_distributed",
    "load_pt_state_dict",
    "load_pretrained_weights",
    "load_safetensors_tensors",
    "stage_pretrained_weights",
    "strip_state_dict_prefix",
]
