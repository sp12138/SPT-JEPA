"""Declarative model-weight sources produced by vforge training."""

from __future__ import annotations

from dataclasses import dataclass

from vforge.models.loading import CompatibleLoadCallbacks

from .mapping import StateDictMapper


@dataclass(frozen=True)
class TrainingCheckpointWeightSource:
    """Model weights from a PT, DCP, or MCore training checkpoint."""

    path: str
    model_path: str = ""
    format: str = "dcp"
    mapper: StateDictMapper | None = None
    integrity: str = "metadata"
    compatible_load_callbacks: CompatibleLoadCallbacks | None = None

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("training checkpoint weight source requires a path")
        if self.format not in {"auto", "pt", "dcp", "mcore"}:
            raise ValueError(
                "training checkpoint weight source format must be auto, pt, dcp, or mcore; "
                f"got {self.format!r}"
            )
        if self.integrity not in {"none", "metadata", "full"}:
            raise ValueError(
                "training checkpoint integrity must be none, metadata, or full; "
                f"got {self.integrity!r}"
            )


__all__ = ["TrainingCheckpointWeightSource"]
