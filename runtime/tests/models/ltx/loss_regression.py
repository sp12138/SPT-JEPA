import json
from pathlib import Path
from typing import Any

LOSS_FIELDS = ("loss", "video_loss", "audio_loss")


def encode_loss_trace(path: str | Path, *, steps: int) -> list[dict[str, int | str]]:
    """Encode consecutive step losses with Python's exact float64 hexadecimal form."""
    records = []
    for line in Path(path).read_text().splitlines():
        payload = json.loads(line)
        if payload.get("event") == "step":
            records.append(payload)

    actual_steps = [int(payload["step"]) for payload in records]
    expected_steps = list(range(steps))
    if actual_steps != expected_steps:
        raise AssertionError(f"expected steps 0..{steps - 1}, got {actual_steps}")

    encoded = []
    for payload in records:
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            raise AssertionError(f"step {payload['step']} has no metrics mapping")
        values: dict[str, Any] = {
            "loss": payload.get("loss_global"),
            "video_loss": metrics.get("video_loss"),
            "audio_loss": metrics.get("audio_loss"),
        }
        for field, value in values.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise AssertionError(f"step {payload['step']} has no numeric {field}")
        encoded.append(
            {
                "step": int(payload["step"]),
                **{field: float(values[field]).hex() for field in LOSS_FIELDS},
            }
        )
    return encoded


def verify_loss_trace(
    actual: list[dict[str, int | str]],
    expected: list[dict[str, int | str]],
    *,
    fields: tuple[str, ...] = LOSS_FIELDS,
) -> None:
    """Fail at the first step/field whose encoded IEEE-754 value differs."""
    if len(actual) != len(expected):
        raise AssertionError(
            f"loss trace length mismatch: actual={len(actual)} expected={len(expected)}"
        )
    for actual_step, expected_step in zip(actual, expected, strict=True):
        step = int(expected_step["step"])
        if actual_step.get("step") != step:
            raise AssertionError(
                f"loss trace step mismatch: actual={actual_step.get('step')} expected={step}"
            )
        for field in fields:
            if actual_step.get(field) != expected_step.get(field):
                raise AssertionError(
                    f"step {step} {field} bitwise mismatch: "
                    f"actual={actual_step.get(field)} expected={expected_step.get(field)}"
                )


def load_loss_golden(path: str | Path, *, steps: int) -> list[dict[str, int | str]]:
    """Load a B300 DP2xUSP2 loss contract from a JSON file."""
    payload = json.loads(Path(path).read_text())
    expected_metadata = {
        "schema_version": 1,
        "machine": "b300",
        "topology": "DP2xUSP2",
        "steps": steps,
    }
    actual_metadata = {key: payload.get(key) for key in expected_metadata}
    if actual_metadata != expected_metadata:
        raise AssertionError(f"unexpected LTX-2.3 golden metadata: {actual_metadata!r}")
    losses = payload.get("losses")
    if not isinstance(losses, list) or len(losses) != steps:
        raise AssertionError(f"golden must contain exactly {steps} loss records")
    return losses
