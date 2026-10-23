"""DCP checkpoint manifest schema, validation, and I/O."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import torch
import torch.distributed as dist

CHECKPOINT_SCHEMA_VERSION = 2
MANIFEST_NAME = "manifest.json"


def json_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CheckpointManifest:
    """Durable metadata written only after every rank finishes checkpointing."""

    schema_version: int
    state: str
    format: str
    completed_steps: int
    world_size: int
    torch_version: str
    python_version: str
    created_at: str
    execution_policy: dict[str, Any]
    topology: dict[str, Any]
    components: dict[str, Any] = field(default_factory=dict)
    components_present: dict[str, bool] = field(default_factory=dict)
    fingerprints: dict[str, Any] = field(default_factory=dict)
    model_fingerprint: dict[str, Any] | None = None
    optimizer_fingerprint: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        *,
        state: str,
        completed_steps: int,
        format: str = "dcp",
        execution_policy: dict[str, Any] | None = None,
        topology: dict[str, Any] | None = None,
        fingerprints: dict[str, Any] | None = None,
    ) -> CheckpointManifest:
        return cls(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            state=state,
            format=format,
            completed_steps=int(completed_steps),
            world_size=dist.get_world_size() if dist.is_initialized() else 1,
            torch_version=str(torch.__version__),
            python_version=sys.version.split()[0],
            created_at=datetime.now(timezone.utc).isoformat(),
            execution_policy=dict(execution_policy or {}),
            topology=dict(topology or {}),
            components={},
            components_present={},
            fingerprints=dict(fingerprints or {}),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CheckpointManifest:
        payload = dict(payload)
        payload.setdefault("topology", {})
        payload.setdefault("components", {})
        payload.setdefault("components_present", {})
        payload.setdefault("fingerprints", {})
        payload.setdefault("model_fingerprint", None)
        payload.setdefault("optimizer_fingerprint", None)
        manifest = cls(**payload)
        if manifest.schema_version not in {1, CHECKPOINT_SCHEMA_VERSION}:
            raise ValueError(
                "unsupported checkpoint schema_version="
                f"{manifest.schema_version}; expected one of 1, "
                f"{CHECKPOINT_SCHEMA_VERSION}"
            )
        if manifest.format not in {"dcp", "mcore"} or manifest.state not in {
            "weights",
            "full",
        }:
            raise ValueError(f"invalid checkpoint manifest: {payload!r}")
        return manifest

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": self.state,
            "format": self.format,
            "completed_steps": self.completed_steps,
            "world_size": self.world_size,
            "torch_version": self.torch_version,
            "python_version": self.python_version,
            "created_at": self.created_at,
            "execution_policy": self.execution_policy,
            "topology": self.topology,
            "components": self.components,
            "components_present": self.components_present,
            "fingerprints": self.fingerprints,
            "model_fingerprint": self.model_fingerprint,
            "optimizer_fingerprint": self.optimizer_fingerprint,
        }


def read_manifest(path: str) -> CheckpointManifest:
    manifest_path = os.path.join(path, MANIFEST_NAME)
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid or missing checkpoint manifest at {path}") from exc
    return CheckpointManifest.from_dict(payload)


def write_manifest(path: str, manifest: CheckpointManifest) -> None:
    """Atomically write a manifest into a checkpoint directory."""
    manifest_path = os.path.join(path, MANIFEST_NAME)
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(tmp_path, manifest_path)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointManifest",
    "read_manifest",
    "write_manifest",
]
