"""Wan Taichu mixed-length training semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

_SCHEDULE_FIELDS = frozenset({"start_step", "end_step", "start_value", "end_value"})
_NON_TEMPORAL_LATENTS = frozenset({"first_frame_latents", "ref_frame_latents"})


@dataclass(frozen=True)
class MaxLatentsSchedule:
    """Integer-linear cap applied on top of one data source's max bound."""

    start_step: int
    end_step: int
    start_value: int
    end_value: int

    @classmethod
    def from_config(cls, raw: Any) -> MaxLatentsSchedule | None:
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise TypeError("model.extras.max_latents_schedule must be a mapping")
        keys = frozenset(str(key) for key in raw)
        missing = sorted(_SCHEDULE_FIELDS - keys)
        unknown = sorted(keys - _SCHEDULE_FIELDS)
        if missing or unknown:
            raise ValueError(
                "model.extras.max_latents_schedule requires exactly "
                f"{sorted(_SCHEDULE_FIELDS)}; missing={missing}, unknown={unknown}"
            )
        values = {field: _config_int(raw[field], field=field) for field in _SCHEDULE_FIELDS}
        schedule = cls(**values)
        if schedule.start_step < 0 or schedule.end_step < 0:
            raise ValueError("max_latents_schedule steps must be non-negative")
        if schedule.end_step < schedule.start_step:
            raise ValueError("max_latents_schedule end_step must be >= start_step")
        if schedule.start_value < 1 or schedule.end_value < 1:
            raise ValueError("max_latents_schedule values must be positive")
        return schedule

    def value_at(self, step: int) -> int:
        if step <= self.start_step:
            return self.start_value
        if step >= self.end_step:
            return self.end_value
        return self.start_value + (
            (self.end_value - self.start_value)
            * (step - self.start_step)
            // (self.end_step - self.start_step)
        )


def truncate_encoded_latents(
    encoded: Mapping[str, Any],
    *,
    step: int,
    device: torch.device,
    generator: torch.Generator,
    schedule: MaxLatentsSchedule | None = None,
) -> dict[str, Any]:
    """Uniformly choose a source-valid length and slice time-aligned latents.

    A dedicated generator keeps length sampling independent of the processor's
    noise and reference-frame draws. Only the length curriculum uses ``step``.
    """
    out = dict(encoded)
    minimum = _optional_int(out.get("min_latents"), field="min_latents")
    maximum = _optional_int(out.get("max_latents"), field="max_latents")
    if minimum is None and maximum is None:
        return out

    frames = out.get("frame_latents")
    if not isinstance(frames, torch.Tensor) or frames.ndim != 5:
        raise ValueError("Wan Taichu temporal truncation requires rank-5 frame_latents")
    original_length = int(frames.shape[2])
    if original_length < 1:
        raise ValueError("frame_latents must contain at least one latent frame")

    lower = 1 if minimum is None else minimum
    upper = original_length if maximum is None else maximum
    if lower < 1 or upper < 1:
        raise ValueError(
            f"latent bounds must be positive; got min_latents={minimum}, max_latents={maximum}"
        )
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"min_latents={minimum} must be <= max_latents={maximum}")
    if schedule is not None:
        upper = min(upper, schedule.value_at(step))
    upper = min(upper, original_length)
    lower = min(lower, upper)

    selected = int(
        torch.randint(
            lower,
            upper + 1,
            (1,),
            device=device,
            generator=generator,
        ).item()
    )
    if selected >= original_length:
        return out

    truncated: dict[str, Any] = {}
    for field, value in out.items():
        if field in _NON_TEMPORAL_LATENTS:
            truncated[field] = value
        elif (
            isinstance(value, torch.Tensor)
            and value.ndim == 5
            and int(value.shape[2]) == original_length
        ):
            truncated[field] = value[:, :, :selected].contiguous()
        else:
            truncated[field] = value
    return truncated


def _config_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"max_latents_schedule.{field} must be an integer")
    return int(value)


def _optional_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer, not bool")
    if isinstance(value, torch.Tensor):
        values = [int(item) for item in value.reshape(-1).tolist()]
    elif isinstance(value, (tuple, list)):
        values = [int(item) for item in value]
    else:
        return int(value)
    if not values:
        raise ValueError(f"{field} must not be empty")
    if any(item != values[0] for item in values[1:]):
        raise ValueError(f"{field} must be homogeneous within one batch; got {values}")
    return values[0]


__all__ = ["MaxLatentsSchedule", "truncate_encoded_latents"]
