"""Framework-owned lifecycle for standalone model evaluation."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Protocol, TypeAlias

import torch
import torch.distributed as dist
import torch.nn as nn

from vforge.distributed.parallel_state import get_parallel_state
from vforge.distributed.parallelize import module_has_meta_tensors
from vforge.models.factory import ModelSpec, resolve_model
from vforge.models.loading import (
    PretrainedWeightSource,
    load_pretrained_weights,
    stage_pretrained_weights,
)
from vforge.training.checkpoint import (
    TrainingCheckpointWeightSource,
    load_model_component_weights,
    load_model_weights,
)
from vforge.training.engine import EngineRunContext, TrainingEngine

ModelWeightSource: TypeAlias = PretrainedWeightSource | TrainingCheckpointWeightSource
EngineForward: TypeAlias = Callable[..., Any]


class EvalInference(Protocol):
    def infer_sample(self, sample: Any, *, seed: int) -> dict[str, Any]: ...


@dataclass(frozen=True)
class EvalOutputWriter:
    """Own expected paths, provenance validation, and atomic sidecars."""

    artifact_name: Callable[[str], str]
    write_output: Callable[[dict[str, Any], str], str]
    minimum_size: int = 1

    def expected_path(self, output_dir: str, sample_id: str) -> str:
        return os.path.join(output_dir, self.artifact_name(sample_id))

    def is_complete(
        self,
        output_dir: str,
        *,
        sample_id: str,
        seed: int,
        provenance: Mapping[str, Any],
    ) -> bool:
        path = self.expected_path(output_dir, sample_id)
        try:
            if not os.path.isfile(path) or os.path.getsize(path) < self.minimum_size:
                return False
            with open(path + ".json", encoding="utf-8") as handle:
                sidecar = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError):
            return False
        return sidecar.get("provenance") == _sample_provenance(
            sample_id=sample_id,
            seed=seed,
            provenance=provenance,
        )

    def write(
        self,
        result: dict[str, Any],
        output_dir: str,
        *,
        sample_id: str,
        seed: int,
        provenance: Mapping[str, Any],
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = self.write_output(result, output_dir)
        expected = self.expected_path(output_dir, sample_id)
        if os.path.abspath(path) != os.path.abspath(expected):
            raise RuntimeError(
                "evaluation writer returned a path different from its declared artifact: "
                f"expected={expected!r}, actual={path!r}"
            )
        if not os.path.isfile(path) or os.path.getsize(path) < self.minimum_size:
            raise RuntimeError(f"evaluation writer produced an incomplete artifact: {path}")
        payload = {
            "sample_id": sample_id,
            "seed": int(seed),
            "metadata": result.get("metadata") or {},
            "provenance": _sample_provenance(
                sample_id=sample_id,
                seed=seed,
                provenance=provenance,
            ),
        }
        temporary = path + ".json.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary, path + ".json")
        return path


@dataclass(frozen=True)
class EvalComponents:
    """Model-owned evaluation recipe; execution remains framework-owned."""

    model: nn.Module | ModelSpec
    dataloader: Iterable[Any]
    initial_weights: ModelWeightSource | Sequence[ModelWeightSource] | None
    build_inference: Callable[[nn.Module, EngineForward], EvalInference]
    output_writer: EvalOutputWriter


def run_evaluation(
    components: EvalComponents,
    *,
    engine: TrainingEngine,
    engine_run: EngineRunContext,
    device: torch.device | str,
    output_dir: str,
    seed: int = 42,
    num_samples: int | None = None,
    skip_existing: bool = True,
    weights_strict: bool = False,
    manifest_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare once, then evaluate each DP-sharded loader sample."""
    state = get_parallel_state()
    os.makedirs(output_dir, exist_ok=True)

    model = resolve_model(components.model)
    device = torch.device(device)
    if not module_has_meta_tensors(model):
        model = model.to(device)
    model.eval()
    model = engine.prepare_model(model)
    model = engine.materialize_model(model, device)
    loaded_weights = _load_initial_weights(
        model,
        components.initial_weights,
        weights_strict=weights_strict,
    )
    # DCP fingerprints include requires_grad, so validate/load against the
    # training-time parameter contract before freezing the model for inference.
    model.requires_grad_(False)
    model.eval()

    def engine_forward(*args: Any, **kwargs: Any) -> Any:
        return engine.run_forward(
            lambda: model(*args, **kwargs),
            run=engine_run,
        )

    inference = components.build_inference(model, engine_forward)
    infer_sample = getattr(inference, "infer_sample", None)
    if not callable(infer_sample):
        raise TypeError("build_inference must return an object with infer_sample()")
    resolve_seed = getattr(inference, "resolve_seed", None)
    resolve_sample_seed = getattr(inference, "resolve_sample_seed", None)

    loader = components.dataloader
    global_length = getattr(loader, "global_length", None)
    if global_length is not None:
        total = int(global_length)
        if num_samples is not None and int(num_samples) != total:
            raise ValueError(
                "num_samples must be applied when the distributed eval loader is built; "
                f"loader={total}, runner={num_samples}"
            )
        samples = iter(loader)
    else:
        local_length = len(loader)  # type: ignore[arg-type]
        total = local_length if num_samples is None else min(local_length, int(num_samples))
        samples = itertools.islice(loader, total)

    context = dict(manifest_context or {})
    checkpoint = _weight_metadata(
        components.initial_weights,
        loaded_weights,
        weights_strict=weights_strict,
    )
    provenance = {
        "resolved_config_hash": context.get("resolved_config_hash"),
        "checkpoint": checkpoint,
    }
    written: list[str] = []
    is_leader = bool(state.is_batch_leader)
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise TypeError(f"eval dataloader must yield mappings, got {type(sample).__name__}")
        sample_id = str(sample.get("sample_id") or f"sample-{index:06d}")
        sample_seed = int(seed) + _stable_sample_salt(sample_id)
        if callable(resolve_sample_seed):
            sample_seed = int(resolve_sample_seed(sample, int(seed)))
        elif callable(resolve_seed):
            sample_seed = int(resolve_seed(sample_id, sample_seed))
        is_padding = bool(sample.get("__vforge_eval_padding__", False))
        local_complete = is_padding or (
            skip_existing
            and is_leader
            and components.output_writer.is_complete(
                output_dir,
                sample_id=sample_id,
                seed=sample_seed,
                provenance=provenance,
            )
        )
        if dist.is_initialized() and state.batch_group is not None:
            decision = [bool(local_complete) if is_leader else None]
            dist.broadcast_object_list(
                decision,
                src=int(dist.get_global_rank(state.batch_group, 0)),
                group=state.batch_group,
            )
            local_complete = bool(decision[0])
        skip_round = local_complete
        if dist.is_initialized():
            collective_decision = torch.tensor(
                int(skip_round),
                dtype=torch.int32,
                device=device,
            )
            dist.all_reduce(collective_decision, op=dist.ReduceOp.MIN)
            skip_round = bool(collective_decision.item())
        if skip_round:
            if is_leader and not is_padding:
                written.append(components.output_writer.expected_path(output_dir, sample_id))
            continue
        result = infer_sample(sample, seed=sample_seed)
        if is_leader and not is_padding:
            if local_complete:
                written.append(components.output_writer.expected_path(output_dir, sample_id))
                continue
            result.setdefault("sample_id", sample_id)
            result.setdefault("seed", sample_seed)
            written.append(
                components.output_writer.write(
                    result,
                    output_dir,
                    sample_id=sample_id,
                    seed=sample_seed,
                    provenance=provenance,
                )
            )

    if dist.is_initialized():
        dist.barrier()
        gathered: list[list[str] | None] = [None] * state.world_size
        dist.all_gather_object(gathered, written)
        all_written = sorted({path for rank_paths in gathered for path in (rank_paths or [])})
    else:
        all_written = written
    manifest = {
        "output_dir": os.path.abspath(output_dir),
        "seed": int(seed),
        "num_samples": total,
        "written": all_written if state.global_rank == 0 else [],
        "resolved_config_hash": context.get("resolved_config_hash"),
        "dp_size": int(state.config.dp_size),
        "usp_size": state.config.usp_size,
        "runtime": context,
        "checkpoint": checkpoint,
    }
    if state.global_rank == 0:
        manifest_path = os.path.join(output_dir, "eval_manifest.json")
        temporary = manifest_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary, manifest_path)
    return manifest


def _weight_sources(
    declared: ModelWeightSource | Sequence[ModelWeightSource] | None,
) -> tuple[ModelWeightSource, ...]:
    if declared is None:
        return ()
    if isinstance(declared, (PretrainedWeightSource, TrainingCheckpointWeightSource)):
        return (declared,)
    return tuple(declared)


def _load_initial_weights(
    model: nn.Module,
    declared: ModelWeightSource | Sequence[ModelWeightSource] | None,
    *,
    weights_strict: bool,
) -> tuple[Any, ...]:
    loaded: list[Any] = []
    for source in _weight_sources(declared):
        if isinstance(source, PretrainedWeightSource):
            staged = stage_pretrained_weights(source)
            load_pretrained_weights(model, staged, strict=weights_strict)
            loaded.append(None)
            continue
        if source.model_path:
            loaded.append(
                load_model_component_weights(
                    model,
                    source.path,
                    component_path=source.model_path,
                    format=source.format,
                    integrity=source.integrity,
                    strict=weights_strict,
                    compatible_load_callbacks=source.compatible_load_callbacks,
                )
            )
        else:
            loaded.append(
                load_model_weights(
                    model,
                    source.path,
                    format=source.format,
                    state_dict_mapper=source.mapper,
                    integrity=source.integrity,
                    strict=weights_strict,
                    compatible_load_callbacks=source.compatible_load_callbacks,
                )
            )
    return tuple(loaded)


def _weight_metadata(
    declared: ModelWeightSource | Sequence[ModelWeightSource] | None,
    loaded: Sequence[Any],
    *,
    weights_strict: bool,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for source, result in zip(_weight_sources(declared), loaded, strict=True):
        if isinstance(source, PretrainedWeightSource):
            content_path = os.path.join(source.path, source.subfolder)
            entry = {
                "kind": "pretrained",
                "path": os.path.abspath(os.path.expanduser(source.path)),
                "model_path": source.model_path,
                "subfolder": source.subfolder,
                "loading_mode": source.loading_mode,
                "content": _path_fingerprint(content_path),
            }
        else:
            entry = {
                "kind": "training_checkpoint",
                "path": os.path.abspath(os.path.expanduser(source.path)),
                "model_path": source.model_path,
                "format": source.format,
                "integrity": source.integrity,
            }
        entry["weights_strict"] = bool(weights_strict)
        if is_dataclass(result):
            payload = asdict(result)
            entry["manifest"] = payload
            entry["manifest_sha256"] = _json_hash(payload)
        elif result is not None:
            entry["result"] = str(result)
        elif isinstance(source, TrainingCheckpointWeightSource):
            entry["content"] = _path_fingerprint(source.path)
        entries.append(entry)
    return {
        "sources": entries,
        "identity_sha256": _json_hash(entries),
    }


def _sample_provenance(
    *,
    sample_id: str,
    seed: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "seed": int(seed),
        "resolved_config_hash": provenance.get("resolved_config_hash"),
        "checkpoint": provenance.get("checkpoint"),
    }


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _path_fingerprint(path: str) -> dict[str, Any]:
    expanded = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(expanded):
        digest = hashlib.sha256()
        with open(expanded, "rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        return {"kind": "file", "sha256": digest.hexdigest()}
    if os.path.isdir(expanded):
        files: list[dict[str, Any]] = []
        for root, _directories, names in os.walk(expanded):
            for name in sorted(names):
                candidate = os.path.join(root, name)
                stat = os.stat(candidate)
                files.append(
                    {
                        "path": os.path.relpath(candidate, expanded),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
        return {
            "kind": "directory",
            "metadata_sha256": _json_hash(files),
            "files": len(files),
        }
    return {"kind": "unresolved", "path": path}


def _stable_sample_salt(sample_id: str) -> int:
    value = 0
    for char in sample_id:
        value = (value * 131 + ord(char)) % 1_000_000_003
    return value


def sample_artifact_name(sample_id: str, *, suffix: str = ".pt") -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("._")
    return f"{safe or 'sample'}{suffix}"


def save_video_artifact(
    result: dict[str, Any],
    output_dir: str,
    *,
    fps: float = 16.0,
) -> str:
    """Atomically write one video; EvalOutputWriter owns its sidecar."""
    from vforge.data.video_io import write_video_mp4

    os.makedirs(output_dir, exist_ok=True)
    sample_id = str(result.get("sample_id") or "sample")
    path = os.path.join(output_dir, sample_artifact_name(sample_id, suffix=".mp4"))
    video = result.get("video")
    if not isinstance(video, torch.Tensor):
        raise TypeError("a video eval result must contain a video tensor")
    temporary = path + ".tmp.mp4"
    write_video_mp4(video, temporary, fps=fps)
    os.replace(temporary, path)
    return path


__all__ = [
    "EngineForward",
    "EvalComponents",
    "EvalInference",
    "EvalOutputWriter",
    "ModelWeightSource",
    "run_evaluation",
    "sample_artifact_name",
    "save_video_artifact",
]
