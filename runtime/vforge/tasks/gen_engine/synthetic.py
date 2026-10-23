"""Generate synthetic precomputed samples for Gen Engine pipeline verification.

These independent Gaussian tensors have no caption/media semantics. Codec
normalization and posterior-moment storage follow the real dataset contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from vforge.data.cache import save_latent_bundle
from vforge.tasks.gen_engine.shared import _validate_shared


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def generate_synthetic_manifest(
    output_dir: str | Path,
    *,
    transformer_config_path: str | Path,
    video_latent_codec_config_path: str | Path,
    audio_latent_codec_config_path: str | Path,
    seed: int = 42,
    samples_per_kind: int = 1,
    profile: str = "smoke",
) -> Path:
    """Write five smoke cases, or only long video cases, into a new directory.

    ``smoke`` includes video, audio, video_audio, reference-conditioned AV and
    long_video. Each has 256 text tokens; long_video has 12,480 visual tokens
    with the catalog's [1,2,2] patch. Repeatable tensor values use a private CPU
    generator. Bundle digests describe the actual files, not a cross-version
    serialization guarantee. Runtime posterior sampling uses the loader's RNG.
    """
    if type(samples_per_kind) is not int or samples_per_kind < 1:
        raise ValueError("samples_per_kind must be a positive integer")
    if profile not in ("smoke", "long_video"):
        raise ValueError("profile must be 'smoke' or 'long_video'")
    paths = {
        "transformer_config_path": str(transformer_config_path),
        "video_latent_codec_config_path": str(video_latent_codec_config_path),
        "audio_latent_codec_config_path": str(audio_latent_codec_config_path),
    }
    config = _validate_shared(
        extras=paths,
        pretrained_path="",
        init_device="meta",
        latent_cache_mode="offline",
        batch_size=1,
    )
    if config.patch_size != (1, 2, 2):
        raise ValueError("synthetic presets require the catalog patch_size=[1,2,2]")
    codec = config.codec
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def save(name: str, field: str, value: torch.Tensor) -> dict:
        path = output / f"{name}.pt"
        save_latent_bundle({field: value}, str(path))
        if field == "embedding":
            return {
                "embedding_path": path.name,
                "shape": list(value.shape),
                "dtype": "float32",
                "embedding_digest": _digest(path),
                "fingerprint": f"synthetic-gaussian-v1-seed-{seed}",
            }
        return {
            "latent_path": path.name,
            "latent_shape": list(value.shape),
            "latent_dtype": "float32",
            "latent_digest": _digest(path),
        }

    def visual(name: str, shape: tuple[int, int, int], *, image: bool = False) -> dict:
        mean = codec.visual.normalization.denormalize(
            torch.randn(
                (codec.visual.latent_channels, *shape), generator=generator, dtype=torch.float32
            ),
            channel_dim=0,
        )
        if codec.visual.posterior_sampling is None:
            latent = mean
        else:
            # A posterior std of 0.1 in normalized model space; bundles store
            # raw VAE-space [mean, logvar], and packing performs the sampling.
            std = (
                torch.tensor(codec.visual.normalization.std, dtype=torch.float32).view(-1, 1, 1, 1)
                * 0.1
            )
            logvar = (2 * std.log()).expand_as(mean)
            latent = torch.cat((mean, logvar), dim=0)
        descriptor = save(name, "latent", latent)
        if not image:
            descriptor["effective_fps"] = 24.0
        return descriptor

    def audio(name: str, frames: int) -> dict:
        latent = codec.audio.normalization.denormalize(
            torch.randn(
                (codec.audio.planes, codec.audio.latent_channels, frames),
                generator=generator,
                dtype=torch.float32,
            ),
            channel_dim=1,
        )
        return {**save(name, "latent", latent), "effective_fps": 32000.0}

    kinds = ("video", "audio", "video_audio", "r2v", "long_video")
    if profile == "long_video":
        kinds = ("long_video",)
    manifest = output / "samples.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for kind in kinds:
            for index in range(samples_per_kind):
                name = f"synthetic-{kind}-{index:04d}"
                record = {
                    "schema": "mmgen.sample",
                    "schema_version": 1,
                    "sample_id": name,
                    "task": "r2v" if kind == "r2v" else "t2v",
                    "caption": "Synthetic independent Gaussian tensors for pipeline verification.",
                    "mme": save(
                        f"{name}-mme",
                        "embedding",
                        torch.randn(
                            (256, config.text_width), generator=generator, dtype=torch.float32
                        ),
                    ),
                    "references": [],
                    "target": {},
                    "metadata": {"synthetic": True, "generator_version": 1, "seed": seed},
                }
                if kind != "audio":
                    shape = (32, 30, 52) if kind == "long_video" else (12, 16, 16)
                    record["target"]["video"] = visual(f"{name}-video", shape)
                if kind in ("audio", "video_audio", "r2v"):
                    record["target"]["audio"] = audio(f"{name}-audio", 65)
                if kind == "r2v":
                    record["references"] = [
                        {
                            "ref_id": "image-1",
                            "type": "image",
                            "image": visual(f"{name}-ref-image", (1, 16, 16), image=True),
                        },
                        {
                            "ref_id": "audio-1",
                            "type": "audio",
                            "audio": audio(f"{name}-ref-audio", 40),
                        },
                    ]
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    (output / "generation.json").write_text(
        json.dumps(
            {
                "generator_version": 1,
                "seed": seed,
                "torch_version": torch.__version__,
                "profile": profile,
                "samples_per_kind": samples_per_kind,
                "text_tokens": 256,
                "config_digests": {name: _digest(Path(path)) for name, path in paths.items()},
                "transformer": config.transformer,
                "video_codec": codec.video_config_dict(),
                "audio_codec": codec.audio_config_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transformer-config", type=Path, required=True)
    parser.add_argument("--video-latent-codec-config", type=Path, required=True)
    parser.add_argument("--audio-latent-codec-config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-kind", type=int, default=1)
    parser.add_argument("--profile", choices=("smoke", "long_video"), default="smoke")
    args = parser.parse_args()
    print(
        generate_synthetic_manifest(
            args.output_dir,
            transformer_config_path=args.transformer_config,
            video_latent_codec_config_path=args.video_latent_codec_config,
            audio_latent_codec_config_path=args.audio_latent_codec_config,
            seed=args.seed,
            samples_per_kind=args.samples_per_kind,
            profile=args.profile,
        )
    )


if __name__ == "__main__":
    main()
