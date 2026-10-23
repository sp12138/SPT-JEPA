"""Framework-independent step reporting helpers."""

from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.distributed as dist

_GLOBAL_MEAN_TOP_LEVEL_KEYS = (
    "loss",
    "loss_backward",
    "learning_rate",
    "learning_rate_before_step",
    "learning_rate_after_step",
)
_GLOBAL_MEAN_SECTIONS = (
    "metrics",
    "timing",
    "attention_offload",
    "perf",
    "memory",
)


def scalar_step_metrics(metrics: Any) -> dict[str, float]:
    """Normalize scalar tensor and numeric metrics for a step payload."""
    values: dict[str, float] = {}
    for key, value in (metrics or {}).items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                values[key] = float(value.detach().float().cpu())
        elif isinstance(value, (int, float)):
            values[key] = float(value)
    return values


def global_mean_step_payload(
    payload: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Globally average reportable scalars while retaining rank-zero losses."""
    averaged = dict(payload)
    for section in _GLOBAL_MEAN_SECTIONS:
        value = payload.get(section)
        if isinstance(value, dict):
            averaged[section] = dict(value)

    loss_rank0 = float(payload["loss"])
    loss_backward_rank0 = float(payload.get("loss_backward", loss_rank0))
    locations: list[tuple[dict[str, Any], str]] = []
    values: list[float] = []

    def collect(container: dict[str, Any], key: str) -> None:
        value = container.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            locations.append((container, key))
            values.append(float(value))

    for key in _GLOBAL_MEAN_TOP_LEVEL_KEYS:
        collect(averaged, key)
    for section in _GLOBAL_MEAN_SECTIONS:
        section_values = averaged.get(section)
        if not isinstance(section_values, dict):
            continue
        for key in section_values:
            collect(section_values, key)

    if values and dist.is_available() and dist.is_initialized():
        scalars = torch.tensor(values, device=device, dtype=torch.float64)
        dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
        scalars.div_(dist.get_world_size())
        values = scalars.cpu().tolist()

    for (container, key), value in zip(locations, values, strict=True):
        container[key] = value

    averaged["loss_rank0"] = loss_rank0
    averaged["loss_global"] = float(averaged["loss"])
    averaged["loss_backward_rank0"] = loss_backward_rank0
    averaged["loss_backward_global"] = float(averaged.get("loss_backward", averaged["loss"]))
    return averaged


def print_json_step(payload: dict[str, Any]) -> None:
    """Default rank-zero JSON reporter for adapters and smoke runs."""
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "VFORGE_TRAIN " + json.dumps(payload, sort_keys=True, separators=(",", ":")),
            flush=True,
        )


class ConsoleReporter:
    """Emit one rank-zero JSON record per optimizer step."""

    def __init__(self, *, rank: int = 0) -> None:
        self._enabled = rank == 0

    def report_step(self, payload: dict[str, Any]) -> None:
        if self._enabled:
            print(
                "VFORGE_TRAIN "
                + json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )

    def close(self) -> None:
        pass


def wandb_step_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Map a Trainer payload to stable cross-framework W&B metric names."""
    loss_scalar = float(payload["loss"])
    loss_backward = float(payload.get("loss_backward", loss_scalar))
    metrics = {
        "trainer/global_step": int(payload["step"]),
        "train/loss": loss_backward,
        "train/loss_backward": loss_backward,
        "train/loss_scalar": loss_scalar,
    }
    for key, value in payload.get("metrics", {}).items():
        if isinstance(value, (int, float)):
            metric_name = key if "/" in key else f"train/{key}"
            if key == "loss_scalar":
                metrics[metric_name] = float(value)
            else:
                metrics.setdefault(metric_name, float(value))
    for key, value in payload.get("timing", {}).items():
        if isinstance(value, (int, float)):
            metrics[f"time/{key.removesuffix('_s')}"] = float(value)
    for key in (
        "learning_rate",
        "learning_rate_before_step",
        "learning_rate_after_step",
    ):
        value = payload.get(key)
        if value is not None:
            metrics[f"train/{key}"] = float(value)
    for key, value in payload.get("attention_offload", {}).items():
        if isinstance(value, (int, float)):
            metrics[f"offload/{key}"] = float(value)
    for key, value in payload.get("perf", {}).items():
        if isinstance(value, (int, float)):
            metrics[f"perf/{key}"] = float(value)
    return metrics


def compute_throughput_metrics(
    *,
    local_batch: float | int,
    local_tokens: float | int,
    dp_size: int,
    world_size: int,
    timing: dict[str, Any] | None,
) -> dict[str, float]:
    """Build global and per-GPU throughput metrics."""
    dp = max(1, int(dp_size))
    world = max(1, int(world_size))
    global_batch = float(local_batch) * dp
    global_tokens = float(local_tokens) * dp
    metrics = {
        "local_batch": float(local_batch),
        "local_tokens": float(local_tokens),
        "global_batch": global_batch,
        "global_tokens": global_tokens,
        "dp_size": float(dp),
        "world_size": float(world),
    }
    if not timing:
        return metrics

    step_compute_s = float(timing.get("step_compute_s") or 0.0)
    step_s = float(timing.get("step_s") or 0.0)
    if step_compute_s > 0.0:
        tokens_per_sec = global_tokens / step_compute_s
        metrics["tokens_per_sec"] = tokens_per_sec
        metrics["samples_per_sec"] = global_batch / step_compute_s
        metrics["tokens_per_sec_per_gpu"] = tokens_per_sec / world
        metrics["step_time_s"] = step_compute_s
    if step_s > 0.0:
        tokens_per_sec_e2e = global_tokens / step_s
        metrics["tokens_per_sec_e2e"] = tokens_per_sec_e2e
        metrics["tokens_per_sec_per_gpu_e2e"] = tokens_per_sec_e2e / world
        metrics["step_time_e2e_s"] = step_s
    return metrics


def resolve_wandb_run_name(
    configured_name: str,
    *,
    output_dir: str,
    model_name: str,
) -> str:
    """Resolve a useful run name without requiring every config to set one."""
    if configured_name.strip():
        return configured_name.strip()
    for env_name in ("WANDB_NAME", "CI_TASK_NAME"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    output_name = os.path.basename(os.path.normpath(output_dir))
    if output_name and output_name not in {".", os.path.sep}:
        return output_name
    return model_name.strip() or "vforge-train"


class WandbReporter:
    """Small optional W&B sink used by application integrations."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        name: str,
        save_dir: str,
        entity: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        mode: str | None = None,
        config: dict[str, Any] | None = None,
        rank: int = 0,
        include_optimizer_in_backward: bool = False,
        exclude_dataloader_from_step: bool = False,
    ) -> None:
        self._run = None
        self._include_optimizer_in_backward = include_optimizer_in_backward
        self._exclude_dataloader_from_step = exclude_dataloader_from_step
        if (
            not enabled
            or rank != 0
            or os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}
            or os.environ.get("WANDB_MODE", "").lower() == "disabled"
        ):
            return
        if not os.environ.get("WANDB_API_KEY"):
            print(
                "[vforge] WANDB_API_KEY is unset; W&B reporting disabled",
                flush=True,
            )
            return

        try:
            import wandb
        except Exception as exc:
            print(
                f"[vforge] W&B unavailable ({exc}); continuing without W&B",
                flush=True,
            )
            return

        kwargs: dict[str, Any] = {
            "project": project,
            "name": name,
            "dir": save_dir,
            "config": config or {},
        }
        if entity:
            kwargs["entity"] = entity
        if group:
            kwargs["group"] = group
        if tags:
            kwargs["tags"] = tags
        if mode:
            kwargs["mode"] = mode
        try:
            self._run = wandb.init(**kwargs)
        except Exception as exc:
            print(
                f"[vforge] W&B initialization failed ({exc}); continuing without W&B",
                flush=True,
            )
            self._run = None

    def report_step(self, payload: dict[str, Any]) -> None:
        if self._run is not None:
            timing = dict(payload.get("timing", {}) or {})
            if (
                getattr(self, "_include_optimizer_in_backward", False)
                and "backward_s" in timing
                and "optimizer_s" in timing
            ):
                timing["backward_s"] += timing["optimizer_s"]
            if (
                getattr(self, "_exclude_dataloader_from_step", False)
                and "step_s" in timing
                and "dataloader_s" in timing
            ):
                timing["step_s"] = max(
                    0.0,
                    timing["step_s"] - timing["dataloader_s"],
                )
            try:
                self._run.log(
                    wandb_step_metrics({**payload, "timing": timing}),
                    step=int(payload["step"]),
                    commit=True,
                )
            except Exception as exc:
                print(
                    f"[vforge] W&B reporting failed ({exc}); continuing without W&B",
                    flush=True,
                )
                self._run = None

    def close(self) -> None:
        if self._run is not None:
            run = self._run
            self._run = None
            try:
                run.finish()
            except Exception as exc:
                print(
                    f"[vforge] W&B shutdown failed ({exc}); ignoring",
                    flush=True,
                )


class JsonlReporter:
    """Rank-zero durable step trace used by cloud regression and acceptance."""

    def __init__(self, path: str, *, rank: int = 0) -> None:
        self._handle = None
        if rank == 0:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._handle = open(path, "a", encoding="utf-8")

    def report_step(self, payload: dict[str, Any]) -> None:
        if self._handle is None:
            return
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


__all__ = [
    "ConsoleReporter",
    "WandbReporter",
    "JsonlReporter",
    "compute_throughput_metrics",
    "global_mean_step_payload",
    "print_json_step",
    "resolve_wandb_run_name",
    "scalar_step_metrics",
    "wandb_step_metrics",
]
