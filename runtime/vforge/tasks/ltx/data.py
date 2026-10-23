"""Dataset adapter for the upstream LTX ``.precomputed`` layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

PRECOMPUTED_DIR_NAME = ".precomputed"
SOURCE_TO_OUTPUT = {
    "latents": "video_latents",
    "conditions": "conditions",
    "audio_latents": "audio_latents",
}


def normalize_legacy_video_latents(data: dict[str, Any]) -> dict[str, Any]:
    """Convert legacy ``[F*H*W, C]`` video latents to ``[C, F, H, W]``."""
    latents = data["latents"]
    if not isinstance(latents, torch.Tensor) or latents.ndim != 2:
        return data

    frames = int(data["num_frames"])
    height = int(data["height"])
    width = int(data["width"])
    expected_tokens = frames * height * width
    if latents.shape[0] != expected_tokens:
        raise ValueError(
            "legacy video latent token count does not match metadata: "
            f"tokens={latents.shape[0]} metadata={frames}x{height}x{width}"
        )
    normalized = data.copy()
    normalized["latents"] = latents.reshape(
        frames,
        height,
        width,
        latents.shape[1],
    ).permute(3, 0, 1, 2)
    return normalized


class LtxPrecomputedDataset(Dataset):
    """Load paired condition, video-latent, and audio-latent files."""

    def __init__(
        self, data_root: str | Path, *, sources: tuple[str, ...] = tuple(SOURCE_TO_OUTPUT)
    ) -> None:
        if (
            not sources
            or not {"latents", "conditions"} <= set(sources)
            or set(sources) - set(SOURCE_TO_OUTPUT)
        ):
            raise ValueError(
                "LTX sources must include latents/conditions and use known source names"
            )
        self.sources = {name: SOURCE_TO_OUTPUT[name] for name in sources}
        root = Path(data_root).expanduser().resolve()
        if (root / PRECOMPUTED_DIR_NAME).is_dir():
            root = root / PRECOMPUTED_DIR_NAME
        if not root.is_dir():
            raise FileNotFoundError(f"LTX precomputed root does not exist: {root}")

        self.root = root
        roots = _discover_roots(root, sources)
        samples: list[tuple[str, dict[str, Path]]] = []
        for shard_root in roots:
            source_paths = {name: shard_root / name for name in self.sources}
            source_sets = {
                name: {path.relative_to(directory) for path in directory.rglob("*.pt")}
                for name, directory in source_paths.items()
            }
            for relative in sorted(source_sets["latents"], key=lambda path: path.as_posix()):
                condition_relative = _condition_relative_path(relative)
                expected = {
                    "latents": relative,
                    "conditions": condition_relative,
                    "audio_latents": relative,
                }
                if all(expected[name] in source_sets[name] for name in self.sources):
                    prefix = "" if len(roots) == 1 else f"{shard_root.name}/"
                    sample_id = prefix + relative.with_suffix("").as_posix()
                    absolute = {name: source_paths[name] / expected[name] for name in self.sources}
                    samples.append((sample_id, absolute))

        if not samples:
            raise ValueError(f"no paired LTX samples found under {root}")
        self._samples = tuple(samples)
        self.sample_ids = tuple(sample_id for sample_id, _ in samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id, paths = self._samples[index]
        result: dict[str, Any] = {"sample_id": sample_id}
        for source_name, output_name in self.sources.items():
            path = paths[source_name]
            try:
                value = torch.load(path, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise RuntimeError(f"failed to load {output_name} from {path}: {exc}") from exc
            if source_name == "latents":
                value = normalize_legacy_video_latents(value)
            result[output_name] = value
        return result


def _discover_roots(
    root: Path, sources: tuple[str, ...] = tuple(SOURCE_TO_OUTPUT)
) -> tuple[Path, ...]:
    if all((root / name).is_dir() for name in sources):
        return (root,)
    shards = tuple(
        path
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_dir() and all((path / name).is_dir() for name in sources)
    )
    if not shards:
        missing = [name for name in sources if not (root / name).is_dir()]
        raise FileNotFoundError(
            f"required LTX source directories are missing under {root}: {missing}"
        )
    return shards


def _condition_relative_path(latent_relative: Path) -> Path:
    if latent_relative.name.startswith("latent_"):
        return latent_relative.with_name(f"condition_{latent_relative.name[7:]}")
    return latent_relative


__all__ = [
    "LtxPrecomputedDataset",
    "PRECOMPUTED_DIR_NAME",
    "normalize_legacy_video_latents",
]
