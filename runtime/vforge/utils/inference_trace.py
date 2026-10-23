"""Opt-in bitwise tensor tracing for cross-entry inference parity."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_TRACE_DIR_ENV = "WAN_TAICHU_INFERENCE_TRACE_DIR"
_TRACE_SIDE_ENV = "WAN_TAICHU_INFERENCE_TRACE_SIDE"
_TRACE_SAVE_ENV = "WAN_TAICHU_INFERENCE_TRACE_SAVE"
_TRACE_MODULES_ENV = "WAN_TAICHU_INFERENCE_TRACE_MODULES"
_TRACE_INSTANCE: InferenceTrace | None = None
_TRACE_KEY: tuple[str, str, int, str] | None = None


class InferenceTrace:
    """Write rank-zero tensor hashes, with optional full tensor snapshots."""

    def __init__(self, root: Path | None, side: str, save: frozenset[str]) -> None:
        self.root = root
        self.side = side
        self.save = save
        self._sequence = 0
        self._trace_file: Path | None = None
        if root is not None:
            side_root = root / side
            side_root.mkdir(parents=True, exist_ok=True)
            self._trace_file = side_root / "trace.jsonl"
            self._trace_file.write_text("", encoding="utf-8")

    @classmethod
    def from_env(cls) -> InferenceTrace:
        raw_root = os.environ.get(_TRACE_DIR_ENV, "").strip()
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        root = Path(raw_root).expanduser() if raw_root and rank == 0 else None
        side = os.environ.get(_TRACE_SIDE_ENV, "unknown").strip() or "unknown"
        raw_save = os.environ.get(_TRACE_SAVE_ENV, "").strip()
        save = frozenset(item.strip() for item in raw_save.split(",") if item.strip())
        return cls(root, side, save)

    @property
    def enabled(self) -> bool:
        return self._trace_file is not None

    def event(self, stage: str, values: Mapping[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        record = {
            "sequence": self._sequence,
            "stage": str(stage),
            **dict(values or {}),
        }
        self._append(record)

    def tensor(
        self,
        stage: str,
        value: torch.Tensor | None,
        values: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled or value is None:
            return
        stride = list(value.stride())
        is_contiguous = value.is_contiguous()
        tensor = value.detach().contiguous().cpu()
        payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        record = {
            "sequence": self._sequence,
            "stage": str(stage),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "stride": stride,
            "is_contiguous": is_contiguous,
            "numel": tensor.numel(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            **dict(values or {}),
        }
        if "all" in self.save or stage in self.save:
            assert self.root is not None
            snapshot = self.root / self.side / f"{self._sequence:04d}-{_safe_name(stage)}.pt"
            torch.save(tensor, snapshot)
            record["snapshot"] = str(snapshot)
        self._append(record)

    def _append(self, record: Mapping[str, Any]) -> None:
        assert self._trace_file is not None
        with self._trace_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
        self._sequence += 1


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_" for character in value
    )


def get_inference_trace() -> InferenceTrace:
    """Return one process-local trace so encoder and sampler share record order."""
    global _TRACE_INSTANCE, _TRACE_KEY
    raw_root = os.environ.get(_TRACE_DIR_ENV, "").strip()
    side = os.environ.get(_TRACE_SIDE_ENV, "unknown").strip() or "unknown"
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    raw_save = os.environ.get(_TRACE_SAVE_ENV, "").strip()
    key = (raw_root, side, rank, raw_save)
    if _TRACE_INSTANCE is None or _TRACE_KEY != key:
        _TRACE_INSTANCE = InferenceTrace.from_env()
        _TRACE_KEY = key
    return _TRACE_INSTANCE


def _enable_deterministic_runtime() -> None:
    if os.environ.get("WAN_TAICHU_INFERENCE_TRACE_DETERMINISTIC") != "1":
        return
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def install_inference_module_trace(model: torch.nn.Module) -> None:
    """Hash matching Wan module outputs without changing their forward path."""
    trace = get_inference_trace()
    if os.environ.get(_TRACE_MODULES_ENV) != "1" or not trace.enabled:
        return
    _enable_deterministic_runtime()
    transformer = (
        model.primary_transformer()
        if callable(getattr(model, "primary_transformer", None))
        else model
    )
    if getattr(transformer, "_wan_inference_module_trace_installed", False):
        return

    modules: list[tuple[str, torch.nn.Module]] = []
    for name in ("patch_embedding", "patch_embedding_ref", "condition_embedder"):
        module = getattr(transformer, name, None)
        if isinstance(module, torch.nn.Module):
            modules.append((name, module))
    condition_embedder = getattr(transformer, "condition_embedder", None)
    if isinstance(condition_embedder, torch.nn.Module):
        for name in ("timesteps_proj", "time_embedder", "text_embedder"):
            module = getattr(condition_embedder, name, None)
            if isinstance(module, torch.nn.Module):
                modules.append((f"condition_embedder.{name}", module))
    blocks = getattr(transformer, "blocks", ())
    modules.extend(
        (f"blocks.{index}", block)
        for index, block in enumerate(blocks)
        if isinstance(block, torch.nn.Module)
    )
    for name in ("norm_out", "proj_out"):
        module = getattr(transformer, name, None)
        if isinstance(module, torch.nn.Module):
            modules.append((name, module))

    call_counts = {name: 0 for name, _ in modules}

    def record(stage: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            trace.tensor(stage, value)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                record(f"{stage}.{index}", item)

    for name, module in modules:

        def pre_hook(
            hooked_module: torch.nn.Module,
            inputs: tuple[Any, ...],
            *,
            module_name: str = name,
        ) -> None:
            call_index = call_counts[module_name]
            if call_index == 0 and module_name in {
                "patch_embedding",
                "patch_embedding_ref",
            }:
                compute_dtype = next(
                    (
                        value.dtype
                        for value in inputs
                        if isinstance(value, torch.Tensor) and value.is_floating_point()
                    ),
                    None,
                )
                for parameter_name in ("weight", "bias"):
                    parameter = getattr(hooked_module, parameter_name, None)
                    if isinstance(parameter, torch.Tensor):
                        full_tensor = getattr(parameter, "full_tensor", None)
                        value = full_tensor() if callable(full_tensor) else parameter
                        trace.tensor(
                            f"module.weight.{module_name}.{parameter_name}",
                            value,
                        )
                        if compute_dtype is not None and value.is_floating_point():
                            trace.tensor(
                                f"module.compute_weight.{module_name}.{parameter_name}",
                                value.to(dtype=compute_dtype),
                            )
            record(f"module.{call_index:03d}.{module_name}.input", inputs)

        def hook(
            hooked_module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
            *,
            module_name: str = name,
        ) -> None:
            call_index = call_counts[module_name]
            call_counts[module_name] += 1
            record(f"module.{call_index:03d}.{module_name}.output", output)
            if (
                call_index == 0
                and module_name == "patch_embedding_ref"
                and isinstance(hooked_module, torch.nn.Conv3d)
                and inputs
                and isinstance(inputs[0], torch.Tensor)
            ):
                value = inputs[0]
                weight = hooked_module.weight
                full_weight = getattr(weight, "full_tensor", None)
                weight = full_weight() if callable(full_weight) else weight
                bias = hooked_module.bias
                if bias is not None:
                    full_bias = getattr(bias, "full_tensor", None)
                    bias = full_bias() if callable(full_bias) else bias
                    bias = bias.to(dtype=value.dtype)
                replay = F.conv3d(
                    value,
                    weight.to(dtype=value.dtype),
                    bias,
                    hooked_module.stride,
                    hooked_module.padding,
                    hooked_module.dilation,
                    hooked_module.groups,
                )
                trace.tensor(
                    f"module.{call_index:03d}.{module_name}.functional_replay",
                    replay,
                )

        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(hook)
    transformer._wan_inference_module_trace_installed = True


__all__ = [
    "InferenceTrace",
    "get_inference_trace",
    "install_inference_module_trace",
]
