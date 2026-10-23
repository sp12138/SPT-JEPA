"""Frozen-asset assembly for Wan Taichu condition encoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from vforge.models.wan_taichu.encoder import WanTaichuConditionEncoder
from vforge.utils.training import resolve_dtype


def build_condition_encoder(
    *,
    architecture: str,
    streams: Sequence[str],
    mask_channels: int,
    raw_stream_keys: Mapping[str, str],
    extras: Mapping[str, Any],
    pretrained_path: str,
    precision: str,
) -> WanTaichuConditionEncoder:
    """Load frozen assets and bind one architecture's encoded-batch contract."""
    if not pretrained_path:
        raise ValueError("model.pretrained_path is required for online encode")

    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        AutoTokenizer,
        CLIPImageProcessor,
        CLIPVisionModel,
        UMT5EncoderModel,
    )

    condition_root = str(extras.get("condition_pretrained_path") or pretrained_path)
    vae_root = str(extras.get("vae_pretrained_path") or condition_root)
    tokenizer = AutoTokenizer.from_pretrained(condition_root, subfolder="tokenizer")
    vae = AutoencoderKLWan.from_pretrained(
        vae_root,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).eval()
    text_encoder = UMT5EncoderModel.from_pretrained(
        condition_root,
        subfolder="text_encoder",
        # Pin fp32 rather than inherit the checkpoint's bf16: transformers
        # changed this default between 5.8 and 5.16, so leaving it unset makes
        # the prompt embeddings a function of the installed version. The
        # corpora this task targets were built against the fp32 behaviour.
        dtype=torch.float32,
    ).eval()

    encoding = "ti2v" if architecture == "wan22_ti2v_5b" else "wan21"
    use_clip = encoding == "wan21" and bool(extras.get("use_clip_condition", False))
    image_encoder = None
    image_processor = None
    if use_clip:
        clip_root = str(extras.get("clip_pretrained_path") or pretrained_path)
        image_encoder = CLIPVisionModel.from_pretrained(
            clip_root,
            subfolder="image_encoder",
            torch_dtype=torch.float32,
        ).eval()
        image_processor = CLIPImageProcessor.from_pretrained(
            clip_root,
            subfolder="image_processor",
        )

    use_dino = encoding == "wan21" and bool(extras.get("use_dino_condition", False))
    dino_encoder = None
    dino_processor = None
    if use_dino:
        dino_root = extras.get("dino_pretrained_path")
        if not dino_root:
            raise ValueError("use_dino_condition=true requires dino_pretrained_path")
        dino_encoder = AutoModel.from_pretrained(
            str(dino_root),
            torch_dtype=torch.float32,
        ).eval()
        dino_processor = AutoImageProcessor.from_pretrained(str(dino_root))

    use_audio = encoding == "wan21" and bool(extras.get("use_audio_condition", False))
    audio_wav2vec = None
    audio_processor = None
    audio_type = str(extras.get("audio_type", "wav2vec2"))
    if use_audio:
        audio_root = extras.get("audio_pretrained_path")
        if not audio_root:
            raise ValueError("Wan Taichu audio online encoding requires audio_pretrained_path")
        from transformers import AutoFeatureExtractor

        if audio_type == "wav2vec2":
            from transformers import Wav2Vec2ForCTC

            audio_wav2vec = Wav2Vec2ForCTC.from_pretrained(
                str(audio_root),
                torch_dtype=torch.float32,
            ).eval()
        elif audio_type == "wav2bert":
            from transformers import Wav2Vec2BertModel

            audio_wav2vec = Wav2Vec2BertModel.from_pretrained(
                str(audio_root),
                torch_dtype=torch.float32,
            ).eval()
        else:
            raise ValueError(f"audio_type must be wav2vec2 or wav2bert; got {audio_type!r}")
        audio_processor = AutoFeatureExtractor.from_pretrained(str(audio_root))

    from vforge.distributed.runtime import get_runtime

    runtime = get_runtime()
    if runtime is not None:
        for module in (vae, text_encoder, image_encoder, dino_encoder, audio_wav2vec):
            if module is not None:
                module.to(runtime.device)

    return WanTaichuConditionEncoder(
        encoding=encoding,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        video_processor=VideoProcessor(vae_scale_factor=2 ** len(vae.temperal_downsample)),
        training_dtype=resolve_dtype(precision),
        streams=streams,
        raw_stream_keys=raw_stream_keys,
        mask_channels=mask_channels,
        use_vae_channels_last_3d=bool(extras.get("use_vae_channels_last_3d", True)),
        use_vae_condition=(encoding == "wan21" and bool(extras.get("use_vae_condition", True))),
        use_clip_condition=use_clip,
        image_encoder=image_encoder,
        image_processor=image_processor,
        use_dino_condition=use_dino,
        dino_encoder=dino_encoder,
        dino_processor=dino_processor,
        use_audio_condition=use_audio,
        audio_wav2vec=audio_wav2vec,
        audio_processor=audio_processor,
        audio_type=audio_type,
        audio_video_rate=int(extras.get("audio_video_rate", 25)),
        audio_fps=int(extras.get("audio_fps", 25)),
        audio_batch_frames=int(extras.get("audio_batch_frames", 121)),
        audio_bucket_m=int(extras.get("audio_bucket_m", 0)),
        audio_sample_rate=int(extras.get("audio_sample_rate", 16000)),
        grid_random_rot_deg=float(extras.get("grid_random_rot_deg", 0.0)),
    )


def build_inference_collate_fn(
    *,
    tokenizer: Any,
    extras: Mapping[str, Any],
    streams: Sequence[str],
    allow_target_video: bool,
):
    """Build Wan Taichu inference collation, including inline camera grids."""
    from vforge.data.collate import collate_raw_wan_inference_batch
    from vforge.tasks.wan_taichu.batch import collate_raw_samples

    max_frames = extras.get("max_frames")
    configured_frames = extras.get("num_frames")
    temporal_scale_factor = int(extras.get("temporal_scale_factor", 4))
    required_files = tuple(name for name in streams if name not in {"rgb", "grid"})

    def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        decoded = bool(samples) and isinstance(samples[0].get("video_frames"), torch.Tensor)
        if decoded:
            if not all(isinstance(sample.get("video_frames"), torch.Tensor) for sample in samples):
                raise ValueError("Wan Taichu eval batch mixes decoded and path-based samples")
            batch = collate_raw_samples(
                samples,
                tokenizer=tokenizer,
                max_frames=None if max_frames is None else int(max_frames),
                max_sequence_length=int(extras.get("max_sequence_length", 512)),
            )
            video = batch["video_frames"]
            batch["num_frames"] = int(video.shape[1])
            batch["height"] = int(video.shape[-2])
            batch["width"] = int(video.shape[-1])
        else:
            batch = collate_raw_wan_inference_batch(
                samples,
                tokenizer=tokenizer,
                max_frames=None if max_frames is None else int(max_frames),
                max_sequence_length=int(extras.get("max_sequence_length", 512)),
                default_num_frames=(None if configured_frames is None else int(configured_frames)),
                required_streams=required_files,
                temporal_scale_factor=temporal_scale_factor,
                allow_target_video=allow_target_video,
            )
        if "grid" in streams and batch.get("grid_condition") is None:
            has_camera = any(
                sample.get("c2ws") is not None or sample.get("Ks") is not None for sample in samples
            )
            if has_camera:
                frame_count = int(batch["num_frames"])
                batch["c2ws"] = _stack_camera_field(
                    samples,
                    key="c2ws",
                    frame_count=frame_count,
                    matrix_shape=(4, 4),
                )
                batch["Ks"] = _stack_camera_field(
                    samples,
                    key="Ks",
                    frame_count=frame_count,
                    matrix_shape=(3, 3),
                )
        return batch

    return collate


def _stack_camera_field(
    samples: list[dict[str, Any]],
    *,
    key: str,
    frame_count: int,
    matrix_shape: tuple[int, int],
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for sample in samples:
        source = sample.get(key)
        if source is None:
            raise ValueError(
                "Wan Taichu grid conditioning requires either a grid video for every "
                f"sample or paired inline c2ws/Ks; sample {sample.get('sample_id')!r} "
                f"is missing {key!r}"
            )
        value = torch.as_tensor(source, dtype=torch.float32)
        if value.ndim != 3 or value.shape[1:] != matrix_shape:
            raise ValueError(
                f"Wan Taichu camera field {key!r} must be [T,{matrix_shape[0]},"
                f"{matrix_shape[1]}], got {tuple(value.shape)} for "
                f"sample {sample.get('sample_id')!r}"
            )
        if value.shape[0] == 0:
            raise ValueError(
                f"Wan Taichu camera field {key!r} is empty for sample {sample.get('sample_id')!r}"
            )
        if value.shape[0] < frame_count:
            value = torch.cat(
                [
                    value,
                    value[-1:].expand(
                        frame_count - value.shape[0],
                        *matrix_shape,
                    ),
                ],
                dim=0,
            )
        values.append(value[:frame_count].contiguous())
    return torch.stack(values, dim=0)


__all__ = [
    "build_condition_encoder",
    "build_inference_collate_fn",
]
