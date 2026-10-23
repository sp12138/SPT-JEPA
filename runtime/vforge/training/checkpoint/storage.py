"""Checkpoint storage, state assembly, and path resolution."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import sys
import time
import warnings
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from vforge.distributed.parallelize import unwrap_replicated_data_parallel
from vforge.models.loading import CompatibleLoadCallbacks

from .config import (
    _DCP_STEP_RE,
    _MCORE_STEP_RE,
    _PT_STEP_RE,
    CheckpointConfig,
    ResumeConfig,
)
from .manifest import CheckpointManifest, json_digest, read_manifest, write_manifest
from .mapping import StateDictMapper

_LATEST_NAME = "latest"
_INCOMPLETE_SUFFIX = ".incomplete"
_DCP_SELECTIVE_PLANNING_GROUP: Any = None


def _load_pt_cpu(path: str) -> Any:
    """Load one PT payload on CPU with optional mmap compatibility fallback."""
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except (RuntimeError, TypeError):
        try:
            return torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            return torch.load(path, map_location="cpu")


def cast_state_dict_for_weights_only(state_dict: Any) -> Any:
    """Normalize portable weights-only floating tensors to FP32."""
    if isinstance(state_dict, dict):
        return {key: cast_state_dict_for_weights_only(value) for key, value in state_dict.items()}
    if isinstance(state_dict, torch.Tensor):
        tensor = state_dict.detach()
        if tensor.is_floating_point():
            return tensor.to(dtype=torch.float32).contiguous().cpu()
        return tensor.cpu()
    return state_dict


def gather_full_model_state_dict(model: nn.Module) -> dict[str, Any]:
    """Return a full (unsharded) CPU state dict for ``model``."""
    try:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )
    except ImportError:
        return model.state_dict()

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    return get_model_state_dict(model, options=options)


def save_weights_only_pt(
    model: nn.Module,
    path: str,
    *,
    is_rank_zero: bool,
    state_dict_mapper: StateDictMapper | None = None,
) -> str | None:
    """Gather full weights and atomically write a PT export on rank zero."""
    state_dict = gather_full_model_state_dict(model)
    if not is_rank_zero:
        return None

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if state_dict_mapper is not None:
        state_dict = state_dict_mapper.export(state_dict)
    else:
        logical_model = unwrap_replicated_data_parallel(model)
        export = getattr(logical_model, "export_checkpoint_state_dict", None)
        if callable(export):
            state_dict = export(state_dict)
    payload = cast_state_dict_for_weights_only(state_dict)
    from vforge.utils.s3_staging import resolve_s3_source

    s3 = resolve_s3_source(path)
    target_path = path
    if s3 is not None:
        digest = hashlib.sha256(f"{s3.bucket}/{s3.key_prefix}".encode()).hexdigest()[:16]
        target_path = os.path.join(
            os.environ.get(
                "VFORGE_CKPT_LOCAL_ROOT",
                "/local-ssd/vforge-checkpoints",
            ),
            "pt-exports",
            digest,
            os.path.basename(path),
        )
    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
    tmp_path = target_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, target_path)
    if s3 is not None:
        from vforge.utils.checkpoint_io_worker import (
            CheckpointIORequest,
            start_checkpoint_io_worker,
        )

        start_checkpoint_io_worker(
            CheckpointIORequest(
                operation="upload_file",
                source=target_path,
                destination=path,
                backend="s3-crt",
                transfer_concurrency=int(os.environ.get("VFORGE_S3_TRANSFER_CONCURRENCY", "32")),
                s3=s3,
            )
        ).wait()
    return path


def _stage_pt_for_rank_zero(path: str, *, filesystem_staging: bool) -> str:
    from vforge.utils.s3_staging import download_file_crt, resolve_s3_source

    s3 = resolve_s3_source(path)
    if s3 is None and not filesystem_staging:
        return path
    identity = f"s3://{s3.bucket}/{s3.key_prefix}" if s3 is not None else os.path.realpath(path)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    destination = os.path.join(
        os.environ.get(
            "VFORGE_CKPT_LOCAL_ROOT",
            "/local-ssd/vforge-checkpoints",
        ),
        "pt-inputs",
        digest,
        os.path.basename(path),
    )
    if s3 is not None:
        download_file_crt(
            destination,
            s3=s3,
            transfer_concurrency=int(os.environ.get("VFORGE_S3_TRANSFER_CONCURRENCY", "32")),
        )
    else:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        temporary = destination + ".tmp"
        shutil.copy2(path, temporary)
        os.replace(temporary, destination)
    return destination


def discover_checkpoints(dirpath: str) -> list[tuple[int, str]]:
    """Return complete PT/DCP/MCore checkpoints ordered by completed step."""
    discovered: list[tuple[int, str]] = []
    if not os.path.isdir(dirpath):
        return discovered
    for path in glob.glob(os.path.join(dirpath, "*")):
        name = os.path.basename(path)
        match = _PT_STEP_RE.match(name)
        if match and os.path.isfile(path):
            discovered.append((int(match.group(1)), path))
            continue
        match = _DCP_STEP_RE.match(name) or _MCORE_STEP_RE.match(name)
        if not match or not os.path.isdir(path):
            continue
        try:
            manifest = read_manifest(path)
        except ValueError:
            continue
        step = int(match.group(1))
        if manifest.completed_steps == step:
            discovered.append((step, path))
    return sorted(discovered, key=lambda item: (item[0], os.path.splitext(item[1])[1]))


def read_latest_checkpoint(dirpath: str) -> str | None:
    """Read and validate the advisory ``latest`` pointer."""
    latest_path = os.path.join(dirpath, _LATEST_NAME)
    try:
        with open(latest_path, encoding="utf-8") as handle:
            basename = handle.read().strip()
    except OSError:
        return None
    if not basename or os.path.basename(basename) != basename:
        return None
    candidate = os.path.join(dirpath, basename)
    for _, path in discover_checkpoints(dirpath):
        if os.path.abspath(path) == os.path.abspath(candidate):
            return path
    return None


def resolve_checkpoint_path(resume: str | None, dirpath: str) -> str | None:
    """Resolve ``none``, ``auto``/``latest``, or an explicit checkpoint path."""
    if resume is None:
        return None
    value = str(resume).strip()
    if not value or value.lower() in {"none", "off", "false", "no"}:
        return None
    if value.lower() in {"auto", "latest"}:
        checkpoints = discover_checkpoints(dirpath)
        latest = read_latest_checkpoint(dirpath)
        if latest is not None and all(path != latest for _, path in checkpoints):
            manifest = read_manifest(latest)
            checkpoints.append((manifest.completed_steps, latest))
            checkpoints.sort(key=lambda item: (item[0], os.path.splitext(item[1])[1]))
        return checkpoints[-1][1] if checkpoints else None
    path = os.path.abspath(os.path.expanduser(value))
    if not os.path.exists(path):
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    if path.endswith(_INCOMPLETE_SUFFIX):
        raise ValueError(f"cannot resume incomplete checkpoint: {path}")
    if os.path.isdir(path):
        read_manifest(path)
    return path


def _stage_dcp_files(
    path: str,
    *,
    relative_paths: tuple[str, ...] | None,
    mode: str,
    component_file_sizes: dict[str, dict[str, int]] | None = None,
) -> str:
    from vforge.utils.node_staging import (
        default_node_staging_config,
        stage_on_node,
    )

    from .planner_staging import activate_prestaged_root

    staging_config = default_node_staging_config()
    overlap_callback = None
    if dist.is_initialized() and dist.get_world_size() > 1:

        def warmup_distributed_checkpoint() -> None:
            warmup_started = time.perf_counter()
            dist.barrier()
            print(
                "VFORGE_DCP_STAGE_WARMUP "
                f"rank={dist.get_rank()} backend={dist.get_backend()} "
                f"elapsed_s={time.perf_counter() - warmup_started:.6f}",
                flush=True,
            )

        overlap_callback = warmup_distributed_checkpoint
    started = time.perf_counter()
    staging_kwargs = {
        "config": staging_config,
        "overlap_callback": overlap_callback,
    }
    if relative_paths is not None:
        staging_kwargs["relative_paths"] = relative_paths
    result = stage_on_node(path, **staging_kwargs)
    if component_file_sizes is not None:
        activate_prestaged_root(
            result.destination,
            path,
            component_file_sizes,
        )
    print(
        "VFORGE_DCP_STAGE "
        f"local_rank={os.environ.get('LOCAL_RANK', '0')} "
        f"copied={str(result.copied).lower()} backend={result.backend} "
        f"source={result.source} destination={result.destination} "
        f"files={result.files} bytes={result.bytes} "
        f"mode={mode} "
        f"object_workers={staging_config.copy_workers} "
        f"transfer_concurrency={staging_config.s3_transfer_concurrency} "
        f"elapsed_s={time.perf_counter() - started:.6f}",
        flush=True,
    )
    return result.destination


def prepare_staged_dcp_path(path: str, *, enabled: bool) -> str:
    """Stage a remote DCP directory once per node and return its local path."""
    if not enabled:
        return path
    manifest = read_manifest(path)
    from .planner_staging import (
        activate_deferred_selective_root,
        build_node_preselection,
        planner_selective_enabled,
    )

    preselection = None
    if planner_selective_enabled():
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if manifest.world_size != world_size:
            activate_deferred_selective_root(path)
            print(
                "VFORGE_DCP_STAGE mode=planner-deferred "
                f"saved_world_size={manifest.world_size} "
                f"current_world_size={world_size} "
                f"source={os.path.realpath(path)}",
                flush=True,
            )
            return path
        preselection = build_node_preselection(path)
    return _stage_dcp_files(
        path,
        relative_paths=(preselection.relative_paths if preselection is not None else None),
        mode="planner-preselect" if preselection is not None else "full",
        component_file_sizes=(
            preselection.component_file_sizes if preselection is not None else None
        ),
    )


def _prepare_deferred_selective_components(
    path: str,
    manifest: CheckpointManifest,
    state: dict[str, Any],
) -> str:
    """Plan both components remotely, stage one node union, and cache plans."""
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    from .planner_staging import (
        PlannerStagingError,
        plan_component_load,
        planner_deferred_path_enabled,
        register_replay_plan,
    )

    if not planner_deferred_path_enabled(path):
        return path
    planning_group = _get_dcp_selective_planning_group(path)
    component_results: dict[str, Any] = {}
    planning_started = time.perf_counter()
    for component, component_state, planner in (
        ("weights", {"model": state["model"]}, None),
        (
            "optimizer",
            {"optimizer": state["optimizer"]},
            DefaultLoadPlanner(allow_partial_load=True),
        ),
    ):
        component_started = time.perf_counter()
        component_path = _component_path(path, manifest, component)
        result = plan_component_load(
            component_state,
            component_path,
            component=component,
            planner=planner,
            process_group=planning_group,
        )
        component_results[component] = result
        _trace_dcp_load_phase(
            "planner_only",
            component_started,
            component=component,
        )
    selections = {result.selection.node for result in component_results.values()}
    if len(selections) != 1:
        raise PlannerStagingError(f"component planner node mismatch: {sorted(selections)}")
    relative_paths = {"manifest.json"}
    component_file_sizes: dict[str, dict[str, int]] = {}
    for component, result in component_results.items():
        component_directory = str(manifest.components.get(component, component))
        relative_paths.update(
            f"{component_directory}/{relative}" for relative in result.selection.relative_paths
        )
        component_file_sizes[component] = result.file_sizes
    print(
        "VFORGE_PLANNER_ONLY_TOTAL "
        f"rank={dist.get_rank() if dist.is_initialized() else 0} "
        f"node={next(iter(selections))} "
        f"files={len(relative_paths)} "
        f"elapsed_s={time.perf_counter() - planning_started:.6f}",
        flush=True,
    )

    staged_path = _stage_dcp_files(
        path,
        relative_paths=tuple(sorted(relative_paths)),
        mode="planner-only",
        component_file_sizes=component_file_sizes,
    )
    for component, result in component_results.items():
        component_directory = str(manifest.components.get(component, component))
        register_replay_plan(
            os.path.join(staged_path, component_directory),
            result.plan,
        )
    return staged_path


def save_dcp(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    *,
    manifest: CheckpointManifest,
    rank_state: dict[str, Any] | None = None,
    save_optimizer: bool = True,
    save_trainer_state: bool = True,
    empty_cache_before_save: bool = False,
    state_dict_mapper: StateDictMapper | None = None,
) -> str | None:
    """Synchronously save a crash-consistent weights or full DCP directory."""
    if manifest.state == "full" and optimizer is None:
        raise ValueError("full DCP checkpoint requires an optimizer")
    if empty_cache_before_save and torch.cuda.is_available():
        torch.cuda.empty_cache()
    incomplete = _prepare_dcp_directory(path, manifest)
    # New checkpoints separate independently-restorable components.  The
    # manifest is committed only after every component and sidecar is durable.
    trainer_saved = rank_state is not None and save_trainer_state
    if trainer_saved:
        _write_rank_sidecar(
            incomplete,
            manifest,
            rank_state,
            trainer_directory="trainer",
        )
    _dcp_save(_get_dcp_state(model, None), os.path.join(incomplete, "weights"))
    optimizer_saved = manifest.state == "full" and save_optimizer
    if optimizer_saved:
        optimizer_state = _get_dcp_state(model, optimizer)["optimizer"]
        optimizer_path = os.path.join(incomplete, "optimizer")
        _dcp_save(
            {"optimizer": optimizer_state},
            None,
            storage_writer=_dcp_optimizer_file_system_writer(optimizer_path),
        )
    committed_manifest = _v2_manifest(
        manifest,
        optimizer_saved=optimizer_saved,
        trainer_saved=trainer_saved,
        model=model,
        optimizer=optimizer if optimizer_saved else None,
        state_dict_mapper=state_dict_mapper,
    )
    _barrier()

    if is_global_zero():
        _commit_dcp_directory(path, incomplete, committed_manifest)
    _barrier()
    return path if is_global_zero() else None


def load_dcp(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    *,
    bit_exact: bool = True,
    warn_on_non_bit_exact: bool = True,
    integrity: str = "metadata",
) -> tuple[CheckpointManifest, dict[str, Any] | None]:
    """Restore model/optimizer and return this rank's full-state sidecar."""
    from vforge.training.checkpoint.config import ResumeConfig

    manifest = read_manifest(path)
    resume = (
        ResumeConfig(strict=bit_exact)
        if manifest.state == "full"
        else ResumeConfig(
            weights=True,
            optimizer=False,
            training_state=False,
            strict=bit_exact,
        )
    )
    return load_training_state(
        model,
        optimizer,
        path,
        resume=resume,
        warn_on_non_bit_exact=warn_on_non_bit_exact,
        integrity=integrity,
        weights_strict=bit_exact,
    )


def load_model_weights(
    model: nn.Module,
    path: str,
    *,
    format: str = "auto",
    state_dict_mapper: StateDictMapper | None = None,
    integrity: str = "metadata",
    stage_remote: bool = False,
    strict: bool = False,
    compatible_load_callbacks: CompatibleLoadCallbacks | None = None,
) -> CheckpointManifest | None:
    """Load model weights from a PT export or either DCP layout."""
    if format not in {"auto", "pt", "dcp", "mcore"}:
        raise ValueError(f"unsupported checkpoint format: {format!r}")
    if format == "pt" or (format == "auto" and os.path.isfile(path)):
        distributed = dist.is_initialized()
        load_here = not distributed or dist.get_rank() == 0
        callback_target_state = None
        if distributed and compatible_load_callbacks is not None:
            callback_target_state = gather_full_model_state_dict(model)
        if load_here:
            print(f"VFORGE_PT_LOAD rank=0 path={path}", flush=True)
        load_path = (
            _stage_pt_for_rank_zero(path, filesystem_staging=stage_remote) if load_here else path
        )
        payload = _load_pt_cpu(load_path) if load_here else {}
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise ValueError(f"PT checkpoint does not contain a state dict: {path}")
        if state_dict_mapper is not None:
            state = state_dict_mapper.import_(state)
        else:
            logical_model = unwrap_replicated_data_parallel(model)
            import_state = getattr(logical_model, "import_checkpoint_state_dict", None)
            if callable(import_state):
                state = import_state(state)
        if load_here and (not strict or compatible_load_callbacks is not None):
            from vforge.models.loading import filter_compatible_state_dict

            state = filter_compatible_state_dict(
                model,
                state,
                source=path,
                callbacks=compatible_load_callbacks,
                target_state=callback_target_state,
                strict=strict,
            )
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
                    broadcast_from_rank0=True,
                    strict=strict,
                ),
            )
        else:
            incompatible = model.load_state_dict(state, strict=False)
            if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
                raise ValueError(
                    "model weights are incompatible: "
                    f"missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
        del state
        del payload
        return None

    manifest = read_manifest(path)
    if manifest.format == "mcore":
        from .backend import load_mcore_training_state

        loaded_manifest, _ = load_mcore_training_state(
            model,
            None,
            path,
            manifest=manifest,
            resume=ResumeConfig(
                weights=True,
                optimizer=False,
                training_state=False,
                strict=strict,
            ),
            weights_strict=strict,
            integrity=integrity,
        )
        return loaded_manifest
    if strict:
        _validate_model_fingerprint(model, manifest)
    _validate_component_integrity(
        path,
        manifest,
        "weights",
        distributed=True,
        mode=integrity,
    )
    weights_path = _component_path(path, manifest, "weights")
    state = _get_dcp_state(model, None)
    missing_model_keys = (
        _dcp_missing_model_keys(state["model"], weights_path)
        if not strict and compatible_load_callbacks is not None
        else set()
    )
    _dcp_load(state, weights_path, strict=strict)
    if missing_model_keys and compatible_load_callbacks is not None:
        from vforge.models.loading import _resolve_missing_model_tensors

        state["model"].update(
            _resolve_missing_model_tensors(
                missing_model_keys,
                model_state=state["model"],
                resolve_missing_key=compatible_load_callbacks.resolve_missing_key,
            )
        )
    _set_dcp_state(model, None, state)
    return manifest


def load_model_component_weights(
    model: nn.Module,
    path: str,
    *,
    component_path: str,
    format: str = "auto",
    integrity: str = "metadata",
    stage_remote: bool = False,
    strict: bool = True,
    compatible_load_callbacks: CompatibleLoadCallbacks | None = None,
) -> CheckpointManifest | None:
    """Load one globally named model subtree from a checkpoint.

    The lightweight view keeps the original FQN (for example ``wan.``) while
    exposing only that subtree to PT, Torch DCP, or MCore native loading. This
    lets DCP ignore the other expert saved in a full payload without allowing
    the selected checkpoint to populate any parameter outside its own domain.
    ``strict`` controls key and shape compatibility within that domain.
    """
    if not component_path or any(not part for part in component_path.split(".")):
        raise ValueError(f"invalid model component path: {component_path!r}")
    logical_model = unwrap_replicated_data_parallel(model)
    component: nn.Module = logical_model
    for part in component_path.split("."):
        child = getattr(component, part, None)
        if not isinstance(child, nn.Module):
            raise ValueError(f"model component {component_path!r} does not exist at {part!r}")
        component = child

    view = nn.Module()
    cursor = view
    parts = component_path.split(".")
    for part in parts[:-1]:
        container = nn.Module()
        cursor.add_module(part, container)
        cursor = container
    cursor.add_module(parts[-1], component)

    is_pt = format == "pt" or (format == "auto" and os.path.isfile(path))
    if not is_pt:
        manifest = read_manifest(path)
        if manifest.format == "mcore":
            from .backend import load_mcore_training_state

            loaded_manifest, _ = load_mcore_training_state(
                view,
                None,
                path,
                manifest=manifest,
                resume=ResumeConfig(
                    weights=True,
                    optimizer=False,
                    training_state=False,
                    strict=strict,
                ),
                weights_strict=strict,
                integrity=integrity,
            )
            return loaded_manifest
        _validate_component_integrity(
            path,
            manifest,
            "weights",
            distributed=True,
            mode=integrity,
        )
        state = _get_dcp_state(view, None)
        weights_path = _component_path(path, manifest, "weights")
        missing_model_keys = (
            _dcp_missing_model_keys(state["model"], weights_path)
            if not strict and compatible_load_callbacks is not None
            else set()
        )
        _dcp_load(state, weights_path, strict=strict)
        if missing_model_keys and compatible_load_callbacks is not None:
            from vforge.models.loading import _resolve_missing_model_tensors

            state["model"].update(
                _resolve_missing_model_tensors(
                    missing_model_keys,
                    model_state=state["model"],
                    resolve_missing_key=compatible_load_callbacks.resolve_missing_key,
                )
            )
        _set_dcp_state(view, None, state)
        return manifest

    distributed = dist.is_initialized()
    load_here = not distributed or dist.get_rank() == 0
    load_path = (
        _stage_pt_for_rank_zero(path, filesystem_staging=stage_remote) if load_here else path
    )
    payload = _load_pt_cpu(load_path) if load_here else {}
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f"PT checkpoint does not contain a state dict: {path}")
    prefix = component_path + "."
    if load_here:
        wrong = sorted(str(key) for key in state if not str(key).startswith(prefix))
        if wrong:
            raise ValueError(
                f"checkpoint for component {component_path!r} contains keys outside "
                f"{prefix!r}: {wrong[:8]}"
            )
        if not strict or compatible_load_callbacks is not None:
            from vforge.models.loading import filter_compatible_state_dict

            state = filter_compatible_state_dict(
                view,
                state,
                source=path,
                callbacks=compatible_load_callbacks,
                strict=strict,
            )
    if distributed:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        set_model_state_dict(
            view,
            model_state_dict=state,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                strict=strict,
            ),
        )
    else:
        incompatible = view.load_state_dict(state, strict=False)
        if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            raise ValueError(
                f"checkpoint is incompatible with component {component_path!r}: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
    return None


def load_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    *,
    resume: ResumeConfig,
    warn_on_non_bit_exact: bool = True,
    integrity: str = "metadata",
    weights_strict: bool = False,
) -> tuple[CheckpointManifest, dict[str, Any] | None]:
    """Selectively restore DCP components according to ``ResumeConfig``."""
    manifest = read_manifest(path)
    if manifest.format == "mcore":
        from .backend import load_mcore_training_state

        return load_mcore_training_state(
            model,
            optimizer,
            path,
            manifest=manifest,
            resume=resume,
            weights_strict=weights_strict,
            integrity=integrity,
        )
    non_weights_requested = resume.optimizer or resume.training_state
    if resume.optimizer and optimizer is None:
        raise ValueError("resume.optimizer=true requires an optimizer")
    if non_weights_requested and manifest.state != "full":
        raise ValueError("checkpoint has weights only; requested training-state resume")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    topology_changed = manifest.world_size != world_size
    if resume.effective_strict and non_weights_requested:
        _validate_strict_training_compatibility(manifest, world_size)
    if topology_changed and non_weights_requested:
        optimizer_warmstart = (
            resume.optimizer and not resume.training_state and not resume.effective_strict
        )
        if not optimizer_warmstart:
            raise ValueError(
                "portable resume across a changed world size supports weights "
                "only unless optimizer warm-start is non-strict with "
                "training_state=false. "
                f"saved world_size={manifest.world_size}, "
                f"current world_size={world_size}"
            )
        if warn_on_non_bit_exact and (not dist.is_initialized() or dist.get_rank() == 0):
            warnings.warn(
                "restoring optimizer state across a changed world size; DCP "
                "will reshard model and optimizer state, but the resumed run "
                "is not bit-exact",
                RuntimeWarning,
                stacklevel=2,
            )
    if resume.weights and resume.optimizer:
        assert optimizer is not None
        _load_full_dcp_components(
            model,
            optimizer,
            path,
            manifest=manifest,
            integrity=integrity,
        )
        if (
            topology_changed
            and not resume.training_state
            and (not dist.is_initialized() or dist.get_rank() == 0)
        ):
            print(
                "VFORGE_OPTIMIZER_RESHARD_RESUME "
                f"saved_world_size={manifest.world_size} "
                f"current_world_size={world_size} status=loaded",
                flush=True,
            )
    elif resume.weights:
        load_model_weights(
            model,
            path,
            format="dcp",
            integrity=integrity,
            strict=(weights_strict or (resume.effective_strict and non_weights_requested)),
        )
    elif resume.optimizer:
        # ResumeConfig rejects this combination, but keep the storage boundary
        # defensive for callers that provide a compatible duck-typed plan.
        raise ValueError("resume.optimizer=true requires resume.weights=true")
    if not resume.training_state:
        return manifest, None
    if not _component_present(manifest, "trainer"):
        raise ValueError("checkpoint does not contain trainer state")
    _validate_component_integrity(
        path,
        manifest,
        "trainer",
        distributed=True,
        mode=integrity,
    )
    rank = dist.get_rank() if dist.is_initialized() else 0
    sidecar_path = _trainer_sidecar_path(path, manifest, rank)
    if not os.path.isfile(sidecar_path):
        raise ValueError(f"missing rank-local state: {sidecar_path}")
    rank_state = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    selected = dict(rank_state)
    selected["_vforge_bit_exact"] = bool(resume.effective_strict and not topology_changed)
    selected["_vforge_restore_rng"] = not topology_changed
    return manifest, selected


def _validate_strict_training_compatibility(
    manifest: CheckpointManifest,
    world_size: int,
) -> None:
    if manifest.torch_version != str(torch.__version__):
        raise ValueError(
            "bit-exact full DCP resume requires the saved PyTorch version: "
            f"saved={manifest.torch_version}, current={torch.__version__}"
        )
    current_python = sys.version.split()[0]
    if manifest.python_version != current_python:
        raise ValueError(
            "bit-exact full DCP resume requires the saved Python version: "
            f"saved={manifest.python_version}, current={current_python}"
        )
    if manifest.world_size != world_size:
        raise ValueError(
            "strict training-state resume requires matching world size: "
            f"saved={manifest.world_size}, current={world_size}"
        )


def apply_retention(config: CheckpointConfig) -> None:
    if config.keep_last is None or not is_global_zero():
        return
    matching: list[tuple[int, str]] = []
    for step, path in discover_checkpoints(config.dirpath):
        if (
            (config.format == "pt" and path.endswith(".pt"))
            or (config.format == "dcp" and path.endswith(".dcp"))
            or (config.format == "mcore" and path.endswith(".mcore"))
        ):
            matching.append((step, path))
    for _, path in matching[: -config.keep_last]:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def convert_dcp_weights_to_pt(
    input_dcp_dir: str,
    output_pt_path: str,
    *,
    component_path: str = "",
    weights: str = "raw",
) -> str:
    """Convert all DCP weights or one raw/EMA model role to portable PT."""

    if weights not in {"raw", "ema"}:
        raise ValueError("DCP export weights must be 'raw' or 'ema'")
    if weights == "ema" and not component_path:
        raise ValueError("EMA DCP export requires component_path")
    manifest = read_manifest(input_dcp_dir)
    _validate_component_integrity(
        input_dcp_dir,
        manifest,
        "weights",
        mode="full",
    )
    weights_dir = _component_path(input_dcp_dir, manifest, "weights")
    output = Path(output_pt_path)
    if output.exists() or Path(str(output) + ".json").exists():
        raise FileExistsError(f"refusing to overwrite existing conversion output: {output}")
    if component_path:
        prefix = _dcp_export_prefix(component_path, weights=weights)
        required = _estimate_dcp_prefix_peak_bytes(weights_dir, prefix)
        available = _available_memory_bytes()
        estimate = {
            "estimated_peak_bytes": required,
            "available_bytes": available,
        }
    else:
        estimate = _conversion_estimate(
            input_dcp_dir,
            manifest,
            weights_dir,
        )
    required_bytes = int(estimate["estimated_peak_bytes"])
    available_bytes = estimate["available_bytes"]
    if available_bytes is not None and required_bytes > available_bytes:
        raise MemoryError(
            "insufficient host memory for single-process DCP-to-PT conversion: "
            f"estimated_peak={required_bytes} available={available_bytes}. "
            "Evaluate the DCP directly or run conversion on a larger host."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    converted = str(output) + ".dcp-conversion.tmp"
    try:
        if component_path:
            state = _load_dcp_export_component(
                weights_dir,
                component_path=component_path,
                weights=weights,
            )
        else:
            try:
                from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
            except ImportError as exc:
                raise ImportError(
                    "DCP-to-PT conversion requires "
                    "torch.distributed.checkpoint.format_utils.dcp_to_torch_save"
                ) from exc
            dcp_to_torch_save(weights_dir, converted)
            payload = torch.load(converted, map_location="cpu", weights_only=False)
            state = payload.get("model", payload) if isinstance(payload, dict) else payload
            if not isinstance(state, dict):
                raise ValueError("converted DCP payload does not contain model weights")
            export_prefix = manifest.fingerprints.get("export_prefix")
            if export_prefix and state and all(key.startswith(export_prefix) for key in state):
                state = {key[len(export_prefix) :]: value for key, value in state.items()}
        tmp_output = str(output) + ".tmp"
        torch.save(cast_state_dict_for_weights_only(state), tmp_output)
        os.replace(tmp_output, output)
    finally:
        if os.path.exists(converted):
            os.remove(converted)
    metadata = {
        "source_path": os.path.abspath(input_dcp_dir),
        "step": manifest.completed_steps,
        "schema_version": manifest.schema_version,
        "component_path": component_path or None,
        "weights": weights,
        "sha256": _sha256_file(output),
    }
    sidecar = str(output) + ".json"
    tmp_sidecar = sidecar + ".tmp"
    with open(tmp_sidecar, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(tmp_sidecar, sidecar)
    return str(output)


def estimate_dcp_conversion(
    input_dcp_dir: str,
    *,
    component_path: str = "",
    weights: str = "raw",
) -> dict[str, Any]:
    """Return the host-memory gate used by offline conversion without loading tensors."""
    if weights not in {"raw", "ema"}:
        raise ValueError("DCP export weights must be 'raw' or 'ema'")
    if weights == "ema" and not component_path:
        raise ValueError("EMA DCP export requires component_path")
    manifest = read_manifest(input_dcp_dir)
    _validate_component_integrity(
        input_dcp_dir,
        manifest,
        "weights",
        mode="metadata",
    )
    weights_dir = _component_path(input_dcp_dir, manifest, "weights")
    if component_path:
        prefix = _dcp_export_prefix(component_path, weights=weights)
        required = _estimate_dcp_prefix_peak_bytes(weights_dir, prefix)
        available = _available_memory_bytes()
        return {
            "source_path": os.path.abspath(input_dcp_dir),
            "completed_steps": manifest.completed_steps,
            "component_path": component_path,
            "weights": weights,
            "estimated_peak_bytes": required,
            "available_bytes": available,
            "sufficient": available is None or required <= available,
        }
    return _conversion_estimate(input_dcp_dir, manifest, weights_dir)


def _dcp_export_prefix(component_path: str, *, weights: str) -> str:
    if not component_path or any(not part for part in component_path.split(".")):
        raise ValueError(f"invalid DCP export component_path: {component_path!r}")
    target = component_path if weights == "raw" else f"{component_path}_ema"
    return f"model.{target}."


def _dcp_tensor_metadata_with_prefix(
    weights_dir: str,
    prefix: str,
) -> tuple[Any, dict[str, Any]]:
    from torch.distributed.checkpoint import FileSystemReader

    reader = FileSystemReader(weights_dir)
    metadata = reader.read_metadata()
    selected = {
        key: value for key, value in metadata.state_dict_metadata.items() if key.startswith(prefix)
    }
    if not selected:
        raise KeyError(f"DCP weights contain no tensors under component prefix {prefix!r}")
    for key, value in selected.items():
        properties = getattr(value, "properties", None)
        if getattr(value, "size", None) is None or getattr(properties, "dtype", None) is None:
            raise TypeError(f"DCP export component contains non-tensor state {key!r}")
    return reader, selected


def _estimate_dcp_prefix_peak_bytes(weights_dir: str, prefix: str) -> int:
    _reader, selected = _dcp_tensor_metadata_with_prefix(weights_dir, prefix)
    required = 0
    for value in selected.values():
        dtype = value.properties.dtype
        elements = int(torch.Size(value.size).numel())
        source_bytes = elements * dtype.itemsize
        output_itemsize = torch.float32.itemsize if dtype.is_floating_point else dtype.itemsize
        output_bytes = elements * output_itemsize
        required += source_bytes + output_bytes
    return required


def _load_dcp_export_component(
    weights_dir: str,
    *,
    component_path: str,
    weights: str,
) -> dict[str, Any]:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    prefix = _dcp_export_prefix(component_path, weights=weights)
    reader, metadata = _dcp_tensor_metadata_with_prefix(weights_dir, prefix)
    state = _load_dcp_selected_tensors(
        reader,
        metadata,
        dcp_module=dcp,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )
    component = {key[len(prefix) :]: value for key, value in state.items()}
    if weights == "raw":
        return component
    exported = _ema_export_state(component, component_path=component_path)
    raw_prefix = _dcp_export_prefix(component_path, weights="raw")
    raw_reader, raw_metadata = _dcp_tensor_metadata_with_prefix(weights_dir, raw_prefix)
    buffer_metadata = {
        key: value for key, value in raw_metadata.items() if key[len(raw_prefix) :] not in exported
    }
    if buffer_metadata:
        buffers = _load_dcp_selected_tensors(
            raw_reader,
            buffer_metadata,
            dcp_module=dcp,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        exported.update({key[len(raw_prefix) :]: value for key, value in buffers.items()})
    return exported


def _load_dcp_selected_tensors(
    reader: Any,
    metadata: Mapping[str, Any],
    *,
    dcp_module: Any,
    planner: Any,
) -> dict[str, torch.Tensor]:
    state = {
        key: torch.empty(tuple(value.size), dtype=value.properties.dtype)
        for key, value in metadata.items()
    }
    dcp_module.load(state, storage_reader=reader, planner=planner)
    return state


def _ema_export_state(
    state: Mapping[str, Any],
    *,
    component_path: str,
) -> dict[str, torch.Tensor]:
    initialized = state.get("initialized")
    if not isinstance(initialized, torch.Tensor) or not bool(initialized.item()):
        raise RuntimeError(f"EMA state for {component_path!r} has not been initialized")
    encoded_names = state.get("parameter_names_utf8")
    if not isinstance(encoded_names, torch.Tensor) or encoded_names.dtype != torch.uint8:
        raise ValueError(
            f"EMA state for {component_path!r} does not contain parameter name metadata"
        )
    try:
        names = json.loads(bytes(encoded_names.tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"EMA parameter metadata for {component_path!r} is invalid") from exc
    if not isinstance(names, list) or not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"EMA parameter metadata for {component_path!r} must be a name list")
    exported: dict[str, torch.Tensor] = {}
    for index, name in enumerate(names):
        key = f"shadow_{index:05d}"
        value = state.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"EMA state for {component_path!r} is missing {key!r}")
        exported[name] = value
    expected_shadows = {f"shadow_{index:05d}" for index in range(len(names))}
    extra_shadows = sorted(
        key for key in state if key.startswith("shadow_") and key not in expected_shadows
    )
    if extra_shadows:
        raise ValueError(
            f"EMA state for {component_path!r} has unmapped shadows: {extra_shadows[:8]}"
        )
    return exported


def _conversion_estimate(
    input_dcp_dir: str,
    manifest: CheckpointManifest,
    weights_dir: str,
) -> dict[str, Any]:
    """Estimate conversion memory after the caller validates the component."""
    required = _estimate_conversion_peak_bytes(weights_dir)
    available = _available_memory_bytes()
    return {
        "source_path": os.path.abspath(input_dcp_dir),
        "completed_steps": manifest.completed_steps,
        "estimated_peak_bytes": required,
        "available_bytes": available,
        "sufficient": available is None or required <= available,
    }


def _component_path(path: str, manifest: CheckpointManifest, component: str) -> str:
    if manifest.schema_version >= 2:
        relative = manifest.components.get(component, component)
        return os.path.join(path, relative)
    # v1 stored model and optimizer together under state/.
    return os.path.join(path, "state")


def _v2_manifest(
    manifest: CheckpointManifest,
    *,
    optimizer_saved: bool,
    trainer_saved: bool,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    state_dict_mapper: StateDictMapper | None = None,
) -> CheckpointManifest:
    logical_model = unwrap_replicated_data_parallel(model) if model is not None else None
    fingerprints = dict(manifest.fingerprints)
    if state_dict_mapper is not None and state_dict_mapper.export_prefix is None:
        raise ValueError(
            "DCP conversion requires a serializable StateDictMapper; "
            "use StateDictMapper.strip_prefix() or save PT directly"
        )
    export_prefix = (
        state_dict_mapper.export_prefix
        if state_dict_mapper is not None
        else (
            getattr(logical_model, "checkpoint_export_prefix", None)
            if logical_model is not None
            else None
        )
    )
    if export_prefix:
        fingerprints["export_prefix"] = str(export_prefix)
    return replace(
        manifest,
        schema_version=2,
        components={"weights": "weights", "optimizer": "optimizer", "trainer": "trainer"},
        components_present={
            "weights": True,
            "optimizer": optimizer_saved,
            "trainer": trainer_saved,
        },
        model_fingerprint=(
            _model_fingerprint(logical_model)
            if logical_model is not None
            else manifest.model_fingerprint
        ),
        optimizer_fingerprint=(
            _optimizer_fingerprint(logical_model, optimizer)
            if logical_model is not None and optimizer is not None
            else manifest.optimizer_fingerprint
        ),
        fingerprints=fingerprints,
    )


def _model_fingerprint(model: nn.Module) -> dict[str, Any]:
    parameters = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in model.named_parameters()
    ]
    payload = {
        "type": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameters": parameters,
    }
    return {
        "type": payload["type"],
        "parameter_count": len(parameters),
        "digest": json_digest(payload),
    }


def _optimizer_entries(optimizer: Any) -> tuple[tuple[str, torch.optim.Optimizer], ...]:
    """Normalize one optimizer or a named optimizer bundle.

    Specialized trainers expose a mapping through an ``optimizers`` attribute.
    Keeping that protocol structural prevents the checkpoint layer from
    importing a task-specific trainer class.
    """

    declared = getattr(optimizer, "optimizers", optimizer)
    if isinstance(declared, Mapping):
        entries = tuple((str(name), value) for name, value in declared.items())
    else:
        entries = (("optimizer", declared),)
    if not entries:
        raise ValueError("optimizer collection must be non-empty")
    for name, value in entries:
        if not isinstance(value, torch.optim.Optimizer):
            raise TypeError(f"optimizer {name!r} is not a torch.optim.Optimizer")
    return entries


def _optimizer_argument(
    optimizer: Any,
) -> torch.optim.Optimizer | tuple[torch.optim.Optimizer, ...]:
    entries = _optimizer_entries(optimizer)
    if len(entries) == 1 and entries[0][0] == "optimizer":
        return entries[0][1]
    return tuple(value for _name, value in entries)


def _optimizer_fingerprint(
    model: nn.Module,
    optimizer: Any,
) -> dict[str, Any]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    entries = []
    for optimizer_name, live_optimizer in _optimizer_entries(optimizer):
        groups = []
        for group in live_optimizer.param_groups:
            groups.append(
                {
                    "parameters": [
                        names.get(id(parameter), "<unknown>") for parameter in group["params"]
                    ],
                    "option_keys": sorted(key for key in group if key != "params"),
                }
            )
        entries.append(
            {
                "name": optimizer_name,
                "type": (f"{type(live_optimizer).__module__}.{type(live_optimizer).__qualname__}"),
                "groups": groups,
            }
        )
    if len(entries) == 1 and entries[0]["name"] == "optimizer":
        payload = {
            "type": entries[0]["type"],
            "groups": entries[0]["groups"],
        }
        result_type = payload["type"]
        group_count = len(payload["groups"])
    else:
        payload = {"type": "named_optimizer_collection", "optimizers": entries}
        result_type = "named_optimizer_collection"
        group_count = sum(len(entry["groups"]) for entry in entries)
    return {
        "type": result_type,
        "group_count": group_count,
        "digest": json_digest(payload),
    }


def _validate_model_fingerprint(
    model: nn.Module,
    manifest: CheckpointManifest,
) -> None:
    if manifest.model_fingerprint is None:
        return
    current = _model_fingerprint(unwrap_replicated_data_parallel(model))
    if current != manifest.model_fingerprint:
        raise ValueError(
            "model fingerprint does not match checkpoint; "
            f"saved={manifest.model_fingerprint!r}, current={current!r}"
        )


def _validate_optimizer_fingerprint(
    model: nn.Module,
    optimizer: Any,
    manifest: CheckpointManifest,
) -> None:
    if manifest.optimizer_fingerprint is None:
        return
    current = _optimizer_fingerprint(unwrap_replicated_data_parallel(model), optimizer)
    if current != manifest.optimizer_fingerprint:
        raise ValueError(
            "optimizer fingerprint does not match checkpoint; set "
            "resume.optimizer=false for a weights-only warm start. "
            f"saved={manifest.optimizer_fingerprint!r}, current={current!r}"
        )


def _component_present(manifest: CheckpointManifest, component: str) -> bool:
    if manifest.schema_version < 2:
        return component in {"weights", "optimizer", "trainer"} and (
            component != "trainer" or manifest.state == "full"
        )
    return bool(manifest.components_present.get(component, False))


def _trainer_sidecar_path(
    path: str,
    manifest: CheckpointManifest,
    rank: int,
) -> str:
    directory = "trainer" if manifest.schema_version >= 2 else ""
    return os.path.join(path, directory, f"rank-{rank:05d}.pt")


def _prepare_dcp_directory(
    path: str,
    manifest: CheckpointManifest,
    *,
    process_group: Any = None,
) -> str:
    incomplete = path + _INCOMPLETE_SUFFIX
    error: str | None = None
    try:
        _validate_dcp_target(path, manifest)
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite complete checkpoint: {path}")
        if is_global_zero():
            shutil.rmtree(incomplete, ignore_errors=True)
            os.makedirs(incomplete, exist_ok=True)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    _raise_if_checkpoint_phase_failed(
        error,
        phase="prepare",
        process_group=process_group,
    )
    return incomplete


def _write_rank_sidecar(
    incomplete: str,
    manifest: CheckpointManifest,
    rank_state: dict[str, Any] | None,
    *,
    process_group: Any = None,
    trainer_directory: str = "",
) -> None:
    error: str | None = None
    try:
        if rank_state is not None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            directory = os.path.join(incomplete, trainer_directory)
            os.makedirs(directory, exist_ok=True)
            sidecar = os.path.join(directory, f"rank-{rank:05d}.pt")
            tmp_sidecar = sidecar + ".tmp"
            torch.save(rank_state, tmp_sidecar)
            os.replace(tmp_sidecar, sidecar)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    _raise_if_checkpoint_phase_failed(
        error,
        phase="sidecar",
        process_group=process_group,
    )


def _run_checkpoint_phase[T](
    phase: str,
    operation: Callable[[], T],
    process_group: Any,
) -> T:
    result: T | None = None
    error: str | None = None
    try:
        result = operation()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    _raise_if_checkpoint_phase_failed(
        error,
        phase=phase,
        process_group=process_group,
    )
    if result is None:
        raise RuntimeError(f"DCP {phase} returned no result")
    return result


def _raise_if_checkpoint_phase_failed(
    local_error: str | None,
    *,
    phase: str,
    process_group: Any,
) -> None:
    errors = [local_error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size(group=process_group)
        dist.all_gather_object(
            errors,
            local_error,
            group=process_group,
        )
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None]
    if failures:
        raise RuntimeError(f"DCP {phase} failed before upload: " + "; ".join(failures))


def _commit_dcp_directory(
    path: str,
    incomplete: str,
    manifest: CheckpointManifest,
) -> None:
    manifest = _with_component_integrity(incomplete, manifest)
    write_manifest(incomplete, manifest)
    os.replace(incomplete, path)
    _write_text_atomic(
        os.path.join(os.path.dirname(path), _LATEST_NAME),
        os.path.basename(path),
    )


def _with_component_integrity(
    root: str,
    manifest: CheckpointManifest,
) -> CheckpointManifest:
    # Multiple logical components may share one physical payload. MCore native
    # checkpoints intentionally store model and optimizer in the same directory
    # so that one fully-parallel plan can reason about both. Hash a shared
    # payload once; a 20+ GiB optimizer must not be read again merely because it
    # is addressable through two manifest component names.
    integrity_by_path: dict[str, dict[str, Any]] = {}
    integrity: dict[str, dict[str, Any]] = {}
    for component, present in manifest.components_present.items():
        if not present:
            continue
        relative_path = str(manifest.components.get(component, component))
        component_integrity = integrity_by_path.get(relative_path)
        if component_integrity is None:
            component_integrity = _component_integrity(os.path.join(root, relative_path))
            integrity_by_path[relative_path] = component_integrity
        integrity[component] = component_integrity
    fingerprints = dict(manifest.fingerprints)
    fingerprints["component_integrity"] = integrity
    return replace(manifest, fingerprints=fingerprints)


def _component_integrity(component_path: str) -> dict[str, Any]:
    if not os.path.isdir(component_path):
        raise ValueError(f"checkpoint component is missing: {component_path}")
    files = []
    metadata_files = []
    for current_root, _, names in os.walk(component_path):
        for name in sorted(names):
            path = os.path.join(current_root, name)
            relative = os.path.relpath(path, component_path)
            size = os.path.getsize(path)
            metadata_entry = {
                "path": relative,
                "size": size,
            }
            entry = {
                **metadata_entry,
                "sha256": _sha256_file(path),
            }
            metadata_files.append(metadata_entry)
            files.append(entry)
    files.sort(key=lambda entry: entry["path"])
    metadata_files.sort(key=lambda entry: entry["path"])
    return {
        "file_count": len(files),
        "total_bytes": sum(int(entry["size"]) for entry in files),
        "digest": json_digest(files),
        "metadata_digest": json_digest(metadata_files),
    }


def _component_metadata(component_path: str) -> dict[str, Any]:
    if not os.path.isdir(component_path):
        raise ValueError(f"checkpoint component is missing: {component_path}")
    files = []
    for current_root, _, names in os.walk(component_path):
        for name in sorted(names):
            path = os.path.join(current_root, name)
            files.append(
                {
                    "path": os.path.relpath(path, component_path),
                    "size": os.path.getsize(path),
                }
            )
    files.sort(key=lambda entry: entry["path"])
    return {
        "file_count": len(files),
        "total_bytes": sum(int(entry["size"]) for entry in files),
        "metadata_digest": json_digest(files),
    }


def _integrity_matches(
    saved: dict[str, Any],
    current: dict[str, Any],
    *,
    mode: str,
) -> bool:
    required = ("file_count", "total_bytes")
    if mode == "full":
        required += ("digest",)
    if any(saved.get(key) != current.get(key) for key in required):
        return False
    saved_metadata_digest = saved.get("metadata_digest")
    return saved_metadata_digest is None or saved_metadata_digest == current.get("metadata_digest")


def _validate_component_integrity(
    root: str,
    manifest: CheckpointManifest,
    component: str,
    *,
    distributed: bool = False,
    mode: str = "metadata",
) -> None:
    if mode not in {"none", "metadata", "full"}:
        raise ValueError(f"unsupported DCP integrity mode: {mode!r}")
    if mode == "none":
        return
    saved = manifest.fingerprints.get("component_integrity", {}).get(component)
    if saved is None:
        return
    from .planner_staging import resolve_integrity_source

    integrity_root = resolve_integrity_source(root)
    component_path = _component_path(integrity_root, manifest, component)
    compute = _component_integrity if mode == "full" else _component_metadata
    if distributed and dist.is_initialized() and mode == "full":
        # Hashing the full payload is expensive, so rank 0 performs it once and
        # broadcasts the result. Metadata validation is intentionally local on
        # every rank: its stat-only scan is cheap and avoids a Python-object
        # collective on the critical resume path.
        result: list[Any] = [None]
        if dist.get_rank() == 0:
            compute_started = time.perf_counter()
            try:
                result[0] = {
                    "integrity": compute(component_path),
                    "error": None,
                }
            except Exception as exc:
                result[0] = {
                    "integrity": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            _trace_dcp_load_phase(
                "integrity_compute",
                compute_started,
                component=component,
            )
        broadcast_started = time.perf_counter()
        dist.broadcast_object_list(result, src=0)
        _trace_dcp_load_phase(
            "integrity_broadcast",
            broadcast_started,
            component=component,
        )
        payload = result[0]
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid distributed integrity result for {component}: {payload!r}")
        if payload.get("error") is not None:
            raise ValueError(
                f"checkpoint {component} integrity validation failed on rank 0: {payload['error']}"
            )
        current = payload.get("integrity")
    else:
        compute_started = time.perf_counter()
        current = compute(component_path)
        _trace_dcp_load_phase(
            "integrity_compute",
            compute_started,
            component=component,
        )
    if not isinstance(current, dict) or not _integrity_matches(
        saved,
        current,
        mode=mode,
    ):
        raise ValueError(
            f"checkpoint {component} component failed integrity validation: "
            f"saved={saved!r}, current={current!r}"
        )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _estimate_conversion_peak_bytes(weights_dir: str) -> int:
    stored_bytes = sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, names in os.walk(weights_dir)
        for name in names
    )
    # DCP materialization + converted payload + atomic torch.save staging.
    return int(stored_bytes * 2.5)


def _available_memory_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None
    return page_size * available_pages


def _trace_dcp_load_phase(
    name: str,
    started: float,
    *,
    component: str | None = None,
) -> None:
    if os.environ.get("VFORGE_DCP_LOAD_TRACE") != "1":
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    component_text = f" component={component}" if component is not None else ""
    print(
        f"VFORGE_DCP_LOAD_PHASE name={name} rank={rank}{component_text} "
        f"elapsed_s={time.perf_counter() - started:.6f} "
        f"at_s={time.perf_counter():.6f}",
        flush=True,
    )


def _load_full_dcp_components(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    *,
    manifest: CheckpointManifest,
    integrity: str,
) -> None:
    """Restore split weights/optimizer components through one FSDP state."""
    total_started = time.perf_counter()
    validation_started = time.perf_counter()
    fingerprint_started = time.perf_counter()
    _validate_model_fingerprint(model, manifest)
    _trace_dcp_load_phase("model_fingerprint", fingerprint_started)
    integrity_started = time.perf_counter()
    _validate_component_integrity(
        path,
        manifest,
        "weights",
        distributed=True,
        mode=integrity,
    )
    _trace_dcp_load_phase(
        "integrity_total",
        integrity_started,
        component="weights",
    )
    if not _component_present(manifest, "optimizer"):
        raise ValueError("checkpoint does not contain optimizer state")
    fingerprint_started = time.perf_counter()
    _validate_optimizer_fingerprint(model, optimizer, manifest)
    _trace_dcp_load_phase("optimizer_fingerprint", fingerprint_started)
    integrity_started = time.perf_counter()
    _validate_component_integrity(
        path,
        manifest,
        "optimizer",
        distributed=True,
        mode=integrity,
    )
    _trace_dcp_load_phase(
        "integrity_total",
        integrity_started,
        component="optimizer",
    )
    _trace_dcp_load_phase("validate", validation_started)

    get_state_started = time.perf_counter()
    state = _get_dcp_state(model, optimizer)
    _trace_dcp_load_phase("get_state_dict", get_state_started)
    path = _prepare_deferred_selective_components(path, manifest, state)

    weights_started = time.perf_counter()
    _dcp_load(
        {"model": state["model"]},
        _component_path(path, manifest, "weights"),
    )
    _trace_dcp_load_phase("load_weights", weights_started)

    optimizer_started = time.perf_counter()
    missing_optimizer_states = _dcp_load(
        {"optimizer": state["optimizer"]},
        _component_path(path, manifest, "optimizer"),
    )
    _trace_dcp_load_phase("load_optimizer", optimizer_started)

    set_state_started = time.perf_counter()
    _set_dcp_state(
        model,
        optimizer,
        state,
        missing_optimizer_states=missing_optimizer_states,
    )
    _trace_dcp_load_phase("set_state_dict", set_state_started)
    _trace_dcp_load_phase("full_components_total", total_started)


def _get_dcp_state(
    model: nn.Module,
    optimizer: Any,
) -> dict[str, Any]:
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_state_dict,
    )

    if optimizer is None:
        return {"model": get_model_state_dict(model)}
    model_state, optimizer_state = get_state_dict(model, _optimizer_argument(optimizer))
    return {"model": model_state, "optimizer": optimizer_state}


def _set_dcp_state(
    model: nn.Module,
    optimizer: Any,
    state: dict[str, Any],
    *,
    missing_optimizer_states: set[str] | None = None,
) -> None:
    from torch.distributed.checkpoint.state_dict import (
        set_model_state_dict,
        set_state_dict,
    )

    if optimizer is None:
        set_model_state_dict(model, model_state_dict=state["model"])
    else:
        optimizer_argument = _optimizer_argument(optimizer)
        set_state_dict(
            model,
            optimizer_argument,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )
        _clear_missing_optimizer_parameter_states(
            optimizer,
            state["optimizer"],
            missing_optimizer_states or set(),
        )


def _dcp_save(
    state: dict[str, Any],
    checkpoint_id: str | None,
    *,
    process_group: Any = None,
    storage_writer: Any = None,
) -> None:
    import torch.distributed.checkpoint as dcp

    dcp.save(
        state_dict=state,
        checkpoint_id=checkpoint_id,
        storage_writer=storage_writer,
        process_group=process_group,
        no_dist=not dist.is_initialized(),
    )


def _dcp_file_system_writer(checkpoint_id: str, *, cpu_staged: bool = False) -> Any:
    from torch.distributed.checkpoint import FileSystemWriter

    # Match FileSystemWriter.stage(): CPU snapshots need no CUDA copy stream.
    options = {"per_thread_copy_ahead": 0} if cpu_staged else {}
    return FileSystemWriter(checkpoint_id, **options)


def _dcp_optimizer_file_system_writer(checkpoint_id: str, *, cpu_staged: bool = False) -> Any:
    from vforge.training.checkpoint.optimizer_layout import (
        OptimizerFileSystemWriter,
    )

    options = {"per_thread_copy_ahead": 0} if cpu_staged else {}
    return OptimizerFileSystemWriter(checkpoint_id, **options)


def _stage_dcp_state(state: dict[str, Any]) -> dict[str, Any]:
    from torch.distributed.checkpoint.staging import (
        DefaultStager,
        StagingOptions,
    )
    from torch.distributed.tensor import DTensor
    from torch.utils._pytree import tree_flatten, tree_unflatten

    # DefaultStager may copy DTensor wrapper storage, which is invalid for
    # FSDP views. Stage their local tensors through the same stager and
    # restore the distribution metadata after the snapshot is independent.
    leaves, tree = tree_flatten(state)
    dtensors = {index: value for index, value in enumerate(leaves) if isinstance(value, DTensor)}
    staged_input = {
        "leaves": [
            value.to_local() if index in dtensors else value for index, value in enumerate(leaves)
        ]
    }

    stager = DefaultStager(
        StagingOptions(
            use_pinned_memory=False,
            use_shared_memory=False,
            use_async_staging=False,
            use_non_blocking_copy=False,
        )
    )
    try:
        staged = stager.stage(staged_input)
        if isinstance(staged, Future):
            staged = staged.result()
        stager.synchronize_staging()
        staged_leaves = staged["leaves"]
        for index, original in dtensors.items():
            # from_local() would move a CPU snapshot back to the CUDA mesh.
            # Reuse the immutable distribution spec while keeping staged data
            # on CPU, as DTensor.to("cpu") does internally.
            staged_leaves[index] = DTensor(
                staged_leaves[index], original._spec, requires_grad=False
            )
        return tree_unflatten(staged_leaves, tree)
    finally:
        stager.close()


def _get_dcp_selective_planning_group(checkpoint_id: str) -> Any:
    from .planner_staging import (
        planner_deferred_path_enabled,
        planner_selective_path_enabled,
    )

    if (
        not dist.is_initialized()
        or dist.get_world_size() <= 1
        or not (
            planner_selective_path_enabled(checkpoint_id)
            or planner_deferred_path_enabled(checkpoint_id)
        )
    ):
        return None
    global _DCP_SELECTIVE_PLANNING_GROUP
    if _DCP_SELECTIVE_PLANNING_GROUP is None:
        _DCP_SELECTIVE_PLANNING_GROUP = dist.new_group(backend="gloo")
    return _DCP_SELECTIVE_PLANNING_GROUP


def cleanup_checkpoint_runtime() -> None:
    """Release process-local checkpoint planning resources."""
    global _DCP_SELECTIVE_PLANNING_GROUP
    if _DCP_SELECTIVE_PLANNING_GROUP is not None and dist.is_initialized():
        dist.destroy_process_group(_DCP_SELECTIVE_PLANNING_GROUP)
    _DCP_SELECTIVE_PLANNING_GROUP = None
    from .planner_staging import reset_planner_staging_state

    reset_planner_staging_state()


def _dcp_missing_model_keys(
    model_state: dict[str, Any],
    checkpoint_id: str,
) -> set[str]:
    """Return model-state keys absent from a DCP weights payload."""
    from .planner_staging import create_storage_reader

    metadata = create_storage_reader(checkpoint_id, component="weights").read_metadata()
    checkpoint_keys = set(metadata.state_dict_metadata)
    return {
        key
        for key in model_state
        if key not in checkpoint_keys and f"model.{key}" not in checkpoint_keys
    }


def _dcp_load(
    state: dict[str, Any],
    checkpoint_id: str,
    *,
    strict: bool = True,
) -> set[str]:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    from .planner_staging import (
        CompatibleLoadPlanner,
        ReplayLoadPlanner,
        create_storage_reader,
        pop_replay_plan,
    )

    no_dist = not dist.is_initialized()
    planning_group = _get_dcp_selective_planning_group(checkpoint_id)
    replay_plan = pop_replay_plan(checkpoint_id)
    compatible_planner = (
        CompatibleLoadPlanner() if not strict and "optimizer" not in state else None
    )
    default_planner = (
        compatible_planner or DefaultLoadPlanner(allow_partial_load="optimizer" in state)
        if replay_plan is not None
        else None
    )
    replay_planner = (
        ReplayLoadPlanner(default_planner, replay_plan) if replay_plan is not None else None
    )
    if replay_plan is not None:
        print(
            "VFORGE_PLANNER_REPLAY "
            f"component={'optimizer' if 'optimizer' in state else 'weights'} "
            f"rank={dist.get_rank() if dist.is_initialized() else 0} "
            f"items={len(replay_plan.items)}",
            flush=True,
        )
    storage_reader = (
        FileSystemReader(checkpoint_id)
        if replay_plan is not None
        else create_storage_reader(
            checkpoint_id,
            component="optimizer" if "optimizer" in state else "weights",
        )
    )
    if "optimizer" not in state:
        dcp.load(
            state_dict=state,
            storage_reader=storage_reader,
            planner=replay_planner or compatible_planner,
            process_group=planning_group,
            no_dist=no_dist,
        )
        return set()

    missing_optimizer_states = _optimizer_parameters_without_checkpoint_state(
        state["optimizer"],
        checkpoint_id,
    )
    if "model" in state:
        dcp.load(
            state_dict={"model": state["model"]},
            storage_reader=storage_reader,
            planner=replay_planner,
            process_group=planning_group,
            no_dist=no_dist,
        )
    dcp.load(
        state_dict={"optimizer": state["optimizer"]},
        storage_reader=storage_reader,
        planner=(
            replay_planner
            if replay_planner is not None
            else DefaultLoadPlanner(allow_partial_load=True)
        ),
        process_group=planning_group,
        no_dist=no_dist,
    )
    _fill_missing_optimizer_parameter_states(
        state["optimizer"],
        missing_optimizer_states,
    )
    return missing_optimizer_states


def _fill_missing_optimizer_parameter_states(
    optimizer_state: dict[str, Any],
    missing_parameter_states: set[str],
) -> None:
    """Make canonical optimizer state acceptable to set_state_dict."""
    state = optimizer_state.get("state")
    if not isinstance(state, dict):
        return
    for parameter_name in missing_parameter_states:
        state.setdefault(parameter_name, {})


def _optimizer_parameters_without_checkpoint_state(
    optimizer_state: dict[str, Any],
    checkpoint_id: str,
) -> set[str]:
    from torch.distributed.checkpoint import FileSystemReader

    metadata_keys = FileSystemReader(checkpoint_id).read_metadata().state_dict_metadata
    param_groups = optimizer_state.get("param_groups")
    if not isinstance(param_groups, list):
        return set()

    missing: set[str] = set()
    for group in param_groups:
        if not isinstance(group, dict):
            continue
        for parameter_name in group.get("params", ()):
            if not isinstance(parameter_name, str):
                continue
            prefix = f"optimizer.state.{parameter_name}"
            if not any(key == prefix or key.startswith(prefix + ".") for key in metadata_keys):
                missing.add(parameter_name)
    return missing


def _clear_missing_optimizer_parameter_states(
    optimizer: Any,
    optimizer_state: dict[str, Any],
    missing_parameter_states: set[str],
) -> None:
    """Restore the checkpoint's lazy-state membership after loading."""
    if not missing_parameter_states:
        return
    serialized_groups = optimizer_state.get("param_groups")
    if not isinstance(serialized_groups, list):
        raise RuntimeError("loaded optimizer state is missing param_groups")
    live_optimizers = tuple(value for _name, value in _optimizer_entries(optimizer))
    live_groups = [
        (live_optimizer, live_group)
        for live_optimizer in live_optimizers
        for live_group in live_optimizer.param_groups
    ]
    if len(serialized_groups) != len(live_groups):
        raise RuntimeError("loaded optimizer param group count does not match optimizer")

    for serialized_group, (live_optimizer, live_group) in zip(
        serialized_groups,
        live_groups,
        strict=True,
    ):
        parameter_names = serialized_group.get("params", ())
        live_parameters = live_group.get("params", ())
        if len(parameter_names) != len(live_parameters):
            raise RuntimeError("loaded optimizer parameter count does not match optimizer")
        for parameter_name, parameter in zip(
            parameter_names,
            live_parameters,
            strict=True,
        ):
            if parameter_name in missing_parameter_states:
                live_optimizer.state.pop(parameter, None)


def _validate_dcp_target(path: str, manifest: CheckpointManifest) -> None:
    expected_suffix = ".dcp" if manifest.format == "dcp" else ".mcore"
    pattern = _DCP_STEP_RE if manifest.format == "dcp" else _MCORE_STEP_RE
    if not path.endswith(expected_suffix):
        raise ValueError(
            f"{manifest.format} checkpoint path must end with {expected_suffix}: {path}"
        )
    basename = os.path.basename(path)
    match = pattern.match(basename)
    if match is None or int(match.group(1)) != manifest.completed_steps:
        raise ValueError(
            "checkpoint filename must match completed_steps: "
            f"path={path}, completed_steps={manifest.completed_steps}"
        )


def _write_text_atomic(path: str, value: str) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(value)
        handle.write("\n")
    os.replace(tmp_path, path)


def _barrier(process_group: Any = None) -> None:
    if dist.is_initialized():
        dist.barrier(group=process_group)


def is_global_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


__all__ = [
    "apply_retention",
    "cast_state_dict_for_weights_only",
    "discover_checkpoints",
    "estimate_dcp_conversion",
    "gather_full_model_state_dict",
    "is_global_zero",
    "convert_dcp_weights_to_pt",
    "load_dcp",
    "load_model_weights",
    "load_training_state",
    "prepare_staged_dcp_path",
    "read_latest_checkpoint",
    "resolve_checkpoint_path",
    "save_dcp",
    "save_weights_only_pt",
]
