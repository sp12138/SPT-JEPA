"""Wan-Normal 冻结资产 encoder。

把三个冻结资产（VAE / tokenizer / UMT5 text encoder）加上 ``VideoProcessor``
统一放到一个 ``nn.Module`` 中，使训练路径（:class:`WanNormalLitModule`）与
推理路径（:class:`WanNormalInferencer`）能共享同一套 encode_batch 实现，
无需重复代码或通过 mixin 继承。

职责
----

* 持有冻结子模块（VAE / text encoder / tokenizer / ``video_processor``），
  并把它们钉在 bf16，与可训练 transformer 对齐，避免 ZeRO-3 等 precision
  plugin 造成的 weight/input dtype 不一致。
* 提供 ``encode_image`` / ``encode_prompt`` —— 对 VAE / UMT5 forward 的
  薄封装，输出时 cast 到 training dtype。
* 提供 ``encode_batch`` —— 把原始 dataset batch 变成训练 loss 构造与推理
  pipeline 调用共同消费的 bundle。

设计要点
~~~~~~~~

* 继承 :class:`torch.nn.Module`，PL strategy（DDP / DeepSpeed / FSDP）才
  能通过 ``self.modules()`` 递归进入并搬到正确设备；但参数是冻结的
  （``requires_grad_(False)``）且钉 bf16，optimizer / grad bucket 不会触碰。
* training dtype 以普通属性（而非 PL property）持有，因为 encoder 可能
  在 ``LightningModule`` 之外使用（例如 ``evaluate.py``），此时
  ``self.trainer`` 为 None。
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from diffusers.video_processor import VideoProcessor
from transformers import CLIPImageProcessor

from vforge.models.wan.encoder import WanFrozenAssetEncoder


def apply_encoded_training_dropouts(
    bundle: dict[str, Any],
    *,
    rgb_first_frame_drop_prob: float,
    rgb_cond_drop_prob: float,
    normal_drop_prob: float,
    text_drop_prob: float,
    streams: Sequence[str],
) -> dict[str, Any]:
    """Apply the canonical Wan conditioning dropout policy to encoded tensors."""
    drop_rgb_first = rgb_first_frame_drop_prob > 0.0 and random.random() < rgb_first_frame_drop_prob
    drop_rgb_cond = rgb_cond_drop_prob > 0.0 and random.random() < rgb_cond_drop_prob
    drop_normal = normal_drop_prob > 0.0 and random.random() < normal_drop_prob
    drop_text = text_drop_prob > 0.0 and random.random() < text_drop_prob

    out = dict(bundle)
    if drop_text:
        out["prompt"] = torch.zeros_like(out["prompt"])
    if drop_rgb_cond:
        for key in (
            "image_condition_latent_vae",
            "image_condition_latent_clip",
            "image_condition_latent_dino",
        ):
            if out.get(key) is not None:
                out[key] = torch.zeros_like(out[key])
    cln = out.get("condition_latents_normal")
    if cln is None or not streams:
        return out
    if cln.shape[1] % len(streams) != 0:
        raise ValueError(
            "packed Wan condition channels cannot be divided across resolved "
            f"streams {streams!r}: shape={tuple(cln.shape)}"
        )
    block_width = cln.shape[1] // len(streams)
    blocks = list(cln.split(block_width, dim=1))
    for index, name in enumerate(streams):
        should_drop = (name == "rgb" and drop_rgb_first) or (name == "normal" and drop_normal)
        if should_drop:
            blocks[index] = torch.zeros_like(blocks[index])
    out["condition_latents_normal"] = torch.cat(blocks, dim=1)
    return out


class Wan21ConditionEncoder(WanFrozenAssetEncoder):
    """训练 / 推理路径共享的冻结资产 encoder。

    支持 VAE-latent 图像条件与 / 或 CLIP-visual 图像条件：

    * ``use_vae_condition=True`` 时，通过 VAE 把参考帧编码为 ``image_condition_latent_vae``。
    * ``use_clip_condition=True`` 时，通过 CLIP Vision 把参考帧编码为
      ``image_condition_latent_clip``，编码路径与 ``diffusers`` 上游
      ``WanImageToVideoPipeline`` 保持一致。

    两者可以同时为 ``True``；至少要有一个为 ``True``。
    """

    def __init__(
        self,
        vae: nn.Module | None,
        text_encoder: nn.Module | None,
        tokenizer,
        video_processor: VideoProcessor | None,
        training_dtype: torch.dtype,
        streams: Sequence[str],
        mask_channels: int,
        raw_stream_keys: Mapping[str, str],
        use_vae_condition: bool = True,
        use_clip_condition: bool = False,
        image_encoder: nn.Module | None = None,
        image_processor: CLIPImageProcessor | None = None,
        use_dino_condition: bool = False,
        dino_encoder: nn.Module | None = None,
        dino_processor: Any | None = None,
        rgb_first_frame_drop_prob: float = 0.0,
        rgb_cond_drop_prob: float = 0.0,
        normal_drop_prob: float = 0.0,
        text_drop_prob: float = 0.0,
    ):
        # 当 host LitModule 设了 ``_skip_frozen_assets=True`` 时，vae /
        # text_encoder / video_processor / image_encoder 都可能为 None（仅训练
        # 路径，带完备 latent cache，此时 encode_batch 不会被调用，因为所有
        # dataset 样本都从磁盘走）。唯一*不能*跳过的冻结资产是 tokenizer：
        # BaseDataModule.link_train_module 会在任何 cache 查找之前读
        # ``lit.tokenizer``，缺失会在 dataset 构造阶段就挂。
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            video_processor=video_processor,
            training_dtype=training_dtype,
        )
        # if not (use_vae_condition or use_clip_condition):
        #    raise ValueError(
        #        "At least one of use_vae_condition / use_clip_condition must be True."
        #    )
        if mask_channels != 4:
            raise ValueError(
                f"Wan21ConditionEncoder requires four mask channels per stream; got {mask_channels}"
            )
        streams = tuple(streams)
        unknown = sorted(set(streams) - ({"rgb"} | set(raw_stream_keys)))
        if unknown:
            raise ValueError(f"Wan21ConditionEncoder does not support streams {unknown}")
        missing_raw = sorted(
            name for name in streams if name != "rgb" and name not in raw_stream_keys
        )
        if missing_raw:
            raise ValueError(f"Wan condition streams have no raw pixel mapping: {missing_raw}")
        self.use_vae_condition = use_vae_condition
        self.use_clip_condition = use_clip_condition
        self.use_dino_condition = use_dino_condition
        self.dino_encoder = dino_encoder
        self.dino_processor = dino_processor
        self._dino_drop_prefix = 5
        self.streams = streams
        self.mask_channels = int(mask_channels)
        self.raw_stream_keys = dict(raw_stream_keys)

        if not (0.0 <= rgb_first_frame_drop_prob <= 1.0):
            raise ValueError(
                f"rgb_first_frame_drop_prob must be in [0, 1], got {rgb_first_frame_drop_prob!r}."
            )
        if not (0.0 <= rgb_cond_drop_prob <= 1.0):
            raise ValueError(f"rgb_cond_drop_prob must be in [0, 1], got {rgb_cond_drop_prob!r}.")
        if not (0.0 <= normal_drop_prob <= 1.0):
            raise ValueError(f"normal_drop_prob must be in [0, 1], got {normal_drop_prob!r}.")
        if not (0.0 <= text_drop_prob <= 1.0):
            raise ValueError(f"text_drop_prob must be in [0, 1], got {text_drop_prob!r}.")
        self.rgb_first_frame_drop_prob = float(rgb_first_frame_drop_prob)
        self.rgb_cond_drop_prob = float(rgb_cond_drop_prob)
        self.normal_drop_prob = float(normal_drop_prob)
        self.text_drop_prob = float(text_drop_prob)

        self.image_encoder = image_encoder
        self.image_processor = image_processor

        if self.image_encoder is not None:
            self.freeze_bf16(self.image_encoder)

    @torch.no_grad()
    def encode_image_clip(self, image: torch.Tensor) -> torch.Tensor:
        """把参考帧 ``[B, 3, H, W]``（range ``[0, 1]``）过 CLIP Vision，
        返回倒数第二层 hidden state。

        与 ``diffusers.WanImageToVideoPipeline.encode_image`` 等价：用
        ``CLIPImageProcessor`` 做归一化，再取 ``hidden_states[-2]``。
        """
        assert self.image_encoder is not None and self.image_processor is not None, (
            "CLIP vision encoder / processor not configured; "
            "set use_clip_condition=True and pass image_encoder + image_processor."
        )
        device = next(self.image_encoder.parameters()).device
        dtype = next(self.image_encoder.parameters()).dtype
        # image_processor 期望 PIL.Image 列表或 [0, 1] 范围的 batched tensor。
        processed = self.image_processor(
            images=image.float().clamp(0.0, 1.0), return_tensors="pt", do_rescale=False
        )
        pixel_values = processed["pixel_values"].to(device=device, dtype=dtype)
        out = self.image_encoder(pixel_values=pixel_values, output_hidden_states=True)
        return out.hidden_states[-2].to(self.training_dtype)

    @torch.no_grad()
    def encode_image_dino(self, image: torch.Tensor) -> torch.Tensor:
        """把参考帧 [B,3,H,W]（range [0,1]）过 DINOv3，取 patch token。

        DINOv3 输出 last_hidden_state = [B, 1+4+N, D]（cls+register+patch）。
        扔掉前 _dino_drop_prefix=5 个（cls+register），留 N 个 patch token 做
        密集几何条件。H+/16 @ 224 -> N=196, D=1280。
        """
        assert self.dino_encoder is not None and self.dino_processor is not None, (
            "DINOv3 encoder / processor not configured; "
            "set use_dino_condition=True and pass dino_encoder + dino_processor."
        )
        device = next(self.dino_encoder.parameters()).device
        dtype = next(self.dino_encoder.parameters()).dtype
        processed = self.dino_processor(
            images=image.float().clamp(0.0, 1.0), return_tensors="pt", do_rescale=False
        )
        pixel_values = processed["pixel_values"].to(device=device, dtype=dtype)
        out = self.dino_encoder(pixel_values=pixel_values)
        feat = out.last_hidden_state  # [B, 201, 1280]
        feat = feat[:, self._dino_drop_prefix :, :]  # [B, 196, 1280]
        return feat.to(self.training_dtype)

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    @staticmethod
    def _build_vae_condition_video(image: torch.Tensor, num_frames_pixel: int) -> torch.Tensor:
        """构造 VAE 条件视频，与上游保持兼容。

        Parameters
        ----------
        image:
            ``[B, 1, 3, H, W]`` 参考帧，已缩放到 ``[-1, 1]``。
        num_frames_pixel:
            目标的像素帧数 ``T_pixel``（例如 Wan-I2V-14B-480P 是 81）。
        """
        assert image.dim() == 5 and image.shape[1] == 1, (
            f"expected [B, 1, 3, H, W] reference frame, got {tuple(image.shape)}"
        )
        zeros_shape = list(image.shape)
        zeros_shape[1] = num_frames_pixel - 1
        return torch.cat([image, image.new_zeros(zeros_shape)], dim=1)

    @staticmethod
    def _expand_images_for_wan_vae(image: torch.Tensor) -> torch.Tensor:
        """为 WAN VAE 把单帧条件帧重复 4 次。"""
        return image.repeat(1, 4, 1, 1, 1)

    # ------------------------------------------------------------------
    # batch 级入口（训练 + 推理共享）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_batch(
        self,
        batch_data: dict[str, Any],
    ) -> dict[str, Any]:
        """把原始 dataset batch 确定性地 encode 为 canonical latent bundle。

        是 ``batch_data`` 的纯函数：不引 RNG、不做条件 drop、不读
        training-mode flag。训练期的 CFG drop 放在 :meth:`apply_training_dropouts`
        里，由调用方触发。
        """
        if self.vae is None:
            raise RuntimeError(
                "Wan21ConditionEncoder.encode_batch was called but vae was not "
                "loaded (host LitModule had _skip_frozen_assets=True). "
                "This path should be unreachable when the latent cache is "
                "fully populated; if you see this, the dataset returned a "
                "raw batch instead of a cached bundle. Rebuild the cache "
                "(prepare_latents.py) or remove latent_cache_dir from the "
                "dataset yaml."
            )
        device = next(self.vae.parameters()).device
        dtype = self.training_dtype

        frames = self.scale_pixels(batch_data["video_frames"]).to(device, dtype)
        stream_conditions: dict[str, torch.Tensor] = {}
        for name in self.streams:
            if name == "rgb":
                continue
            raw_key = self.raw_stream_keys[name]
            pixels = batch_data.get(raw_key)
            if pixels is None:
                raise ValueError(
                    f"resolved Wan streams {list(self.streams)} need a {name!r} "
                    f"stream, but the batch carries no {raw_key!r} video"
                )
            stream_conditions[name] = self.scale_pixels(pixels).to(device, dtype)
        image_condition_raw = batch_data["image_condition"].to(device, dtype)

        text_inputs = batch_data["text_inputs"].to(device).long()
        text_mask = batch_data["text_mask"].to(device).long()
        prompt = self.encode_prompt(text_inputs, text_mask, frames.shape[0])

        image_condition_clip_input = image_condition_raw[:, 0]

        image_condition_vae_input = self.scale_pixels(image_condition_raw)
        if self.streams == ("normal",):
            image_condition_vae_input = self._expand_images_for_wan_vae(image_condition_vae_input)
        else:
            num_frames_pixel = int(batch_data["video_frames"].shape[1])
            image_condition_vae_input = self._build_vae_condition_video(
                image_condition_vae_input, num_frames_pixel
            )

        frames = frames.permute(0, 2, 1, 3, 4)
        stream_conditions = {
            name: value.permute(0, 2, 1, 3, 4) for name, value in stream_conditions.items()
        }
        image_condition_vae_input = image_condition_vae_input.permute(0, 2, 1, 3, 4)

        image_condition_latent_vae = None
        if self.use_vae_condition:
            image_condition_latent_vae = self.encode_image(image_condition_vae_input)

        image_condition_latent_clip = None
        if self.use_clip_condition:
            image_condition_latent_clip = self.encode_image_clip(image_condition_clip_input)

        image_condition_latent_dino = None
        if self.use_dino_condition:
            image_condition_latent_dino = self.encode_image_dino(
                image_condition_clip_input  # 复用首帧 [0,1] 像素，processor 各自归一化
            )

        frame_latents = self.encode_image(frames)

        condition_blocks = []
        for name in self.streams:
            if name == "rgb":
                first_frame_raw = batch_data["video_frames"][:, 0:1].to(device, dtype)
                body = self._encode_rgb_first_frame_condition(
                    first_frame_raw,
                    int(batch_data["video_frames"].shape[1]),
                )
                mask = self._build_temporal_first_frame_mask(
                    int(first_frame_raw.shape[0]),
                    int(batch_data["video_frames"].shape[1]),
                    body,
                )
            else:
                body = self.encode_image(stream_conditions[name])
                mask = torch.ones_like(body)[:, : self.mask_channels]
            condition_blocks.append(torch.cat([mask, body], dim=1))
        condition_latents_normal = torch.cat(condition_blocks, dim=1) if condition_blocks else None

        return {
            "prompt": prompt,
            "frame_latents": frame_latents,
            "image_condition_latent_vae": image_condition_latent_vae,
            "image_condition_latent_clip": image_condition_latent_clip,
            "image_condition_latent_dino": image_condition_latent_dino,
            "condition_latents_normal": condition_latents_normal,
        }

    @torch.no_grad()
    def encode_inference_batch(self, batch_data: dict[str, Any]) -> dict[str, Any]:
        """Encode Wan conditions without encoding a target RGB video."""
        if self.vae is None:
            raise RuntimeError("Wan21 inference condition encoding requires a VAE")
        device = next(self.vae.parameters()).device
        dtype = self.training_dtype
        image_condition_raw = batch_data["image_condition"].to(device, dtype)
        num_frames = int(batch_data["num_frames"])
        text_inputs = batch_data["text_inputs"].to(device).long()
        text_mask = batch_data["text_mask"].to(device).long()
        prompt = self.encode_prompt(text_inputs, text_mask, int(image_condition_raw.shape[0]))
        image_condition_clip_input = image_condition_raw[:, 0]
        image_condition_vae_input = self.scale_pixels(image_condition_raw)
        if self.streams == ("normal",):
            image_condition_vae_input = self._expand_images_for_wan_vae(image_condition_vae_input)
        else:
            image_condition_vae_input = self._build_vae_condition_video(
                image_condition_vae_input,
                num_frames,
            )
        image_condition_vae_input = image_condition_vae_input.permute(0, 2, 1, 3, 4)
        image_condition_latent_vae = (
            self.encode_image(image_condition_vae_input) if self.use_vae_condition else None
        )
        image_condition_latent_clip = (
            self.encode_image_clip(image_condition_clip_input) if self.use_clip_condition else None
        )
        image_condition_latent_dino = (
            self.encode_image_dino(image_condition_clip_input) if self.use_dino_condition else None
        )
        condition_blocks = []
        for name in self.streams:
            if name == "rgb":
                body = self._encode_rgb_first_frame_condition(
                    image_condition_raw,
                    num_frames,
                )
                mask = self._build_temporal_first_frame_mask(
                    int(image_condition_raw.shape[0]),
                    num_frames,
                    body,
                )
            else:
                raw_key = self.raw_stream_keys[name]
                raw_pixels = batch_data.get(raw_key)
                if raw_pixels is None:
                    raise ValueError(
                        f"Wan inference requires a {name!r} condition video in {raw_key!r}"
                    )
                pixels = self.scale_pixels(raw_pixels).to(device, dtype)
                body = self.encode_image(pixels.permute(0, 2, 1, 3, 4))
                mask = torch.ones_like(body)[:, : self.mask_channels]
            condition_blocks.append(torch.cat([mask, body], dim=1))
        return {
            "prompt": prompt,
            "image_condition_latent_vae": image_condition_latent_vae,
            "image_condition_latent_clip": image_condition_latent_clip,
            "image_condition_latent_dino": image_condition_latent_dino,
            "condition_latents_normal": (
                torch.cat(condition_blocks, dim=1) if condition_blocks else None
            ),
        }

    def apply_training_dropouts(self, bundle: dict[str, Any]) -> dict[str, Any]:
        """在已 encode 的 bundle 上叠加 per-input 独立的 CFG drop。

        何时调用由上层决定（仅训练）。encode 所得 bundle 本身保持确定性、
        可缓存。
        """
        return apply_encoded_training_dropouts(
            bundle,
            rgb_first_frame_drop_prob=self.rgb_first_frame_drop_prob,
            rgb_cond_drop_prob=self.rgb_cond_drop_prob,
            normal_drop_prob=self.normal_drop_prob,
            text_drop_prob=self.text_drop_prob,
            streams=self.streams,
        )

    # ------------------------------------------------------------------
    # rgb-cat-rgb-cond helpers
    # ------------------------------------------------------------------
    def _vae_scale_factor_temporal(self) -> int:
        return int(getattr(self.vae.config, "scale_factor_temporal", 4))

    def _encode_rgb_first_frame_condition(
        self,
        first_frame_raw: torch.Tensor,
        num_frames_pixel: int,
    ) -> torch.Tensor:
        # first_frame_raw：[B, 1, 3, H, W]，在 [0, 1]（视频首个像素帧）。
        ref = self.scale_pixels(first_frame_raw).permute(0, 2, 1, 3, 4)  # [B,3,1,H,W]
        zeros_shape = list(ref.shape)
        zeros_shape[2] = num_frames_pixel - 1
        video_condition = torch.cat(
            [ref, ref.new_zeros(zeros_shape)], dim=2
        )  # [B, 3, T_pixel, H, W]
        return self.encode_image(video_condition)  # [B, 16, T_lat, h, w]

    def _build_temporal_first_frame_mask(
        self,
        batch_size: int,
        num_frames_pixel: int,
        latent_condition: torch.Tensor,
    ) -> torch.Tensor:
        _, _, lat_t, lat_h, lat_w = latent_condition.shape
        scale_t = self._vae_scale_factor_temporal()
        mask = torch.ones(
            batch_size,
            1,
            num_frames_pixel,
            lat_h,
            lat_w,
            dtype=latent_condition.dtype,
            device=latent_condition.device,
        )
        mask[:, :, 1:] = 0
        first = mask[:, :, 0:1].repeat_interleave(scale_t, dim=2)
        mask = torch.cat([first, mask[:, :, 1:]], dim=2)
        mask = mask.view(batch_size, -1, scale_t, lat_h, lat_w).transpose(1, 2)
        assert mask.shape[2] == lat_t, (
            f"temporal mask T={mask.shape[2]} disagrees with latent T={lat_t} "
            f"(num_frames_pixel={num_frames_pixel}, scale_t={scale_t})."
        )
        return mask


__all__ = [
    "Wan21ConditionEncoder",
    "apply_encoded_training_dropouts",
]
