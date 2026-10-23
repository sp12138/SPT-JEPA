"""LTX-2.3 convolutional video/audio decoding using the pinned model library."""

import torch


class LtxDecoder:
    """Keep decoder weights on CPU and run one decoding stage at a time."""

    def __init__(self, base_path, device):
        self.base_path = str(base_path)
        self.device = torch.device(device)
        self._models = None

    def _load(self):
        from ltx_core.loader import SingleGPUModelBuilder
        from ltx_core.model.audio_vae import (
            AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            VOCODER_COMFY_KEYS_FILTER,
            AudioDecoderConfigurator,
            VocoderConfigurator,
        )
        from ltx_core.model.video_vae import VAE_DECODER_COMFY_KEYS_FILTER, VideoDecoderConfigurator

        self._models = tuple(
            SingleGPUModelBuilder(
                model_path=self.base_path,
                model_class_configurator=configurator,
                model_sd_ops=ops,
            )
            .build(device=torch.device("cpu"), dtype=torch.bfloat16)
            .eval()
            .requires_grad_(False)
            for configurator, ops in (
                (VideoDecoderConfigurator, VAE_DECODER_COMFY_KEYS_FILTER),
                (AudioDecoderConfigurator, AUDIO_VAE_DECODER_COMFY_KEYS_FILTER),
                (VocoderConfigurator, VOCODER_COMFY_KEYS_FILTER),
            )
        )

    @torch.no_grad()
    def decode(self, values, layout):
        from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
        from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig
        from ltx_core.types import AudioLatentShape, VideoLatentShape

        if self._models is None:
            self._load()
        video_decoder, audio_decoder, vocoder = self._models
        video = VideoLatentPatchifier(patch_size=1).unpatchify(
            values["video"],
            output_shape=VideoLatentShape(
                batch=1,
                channels=128,
                frames=layout.frames,
                height=layout.height,
                width=layout.width,
            ),
        )
        tiling = TileSizeConfig(
            height=DimensionSizeConfig(tile_size=384, overlap=192),
            width=DimensionSizeConfig(tile_size=384, overlap=192),
            frames=DimensionSizeConfig(tile_size=48, overlap=24),
        )
        try:
            video_decoder.to(self.device)
            chunks = [
                chunk[0].float().cpu()
                for chunk in video_decoder.tiled_decode(
                    video.to(device=self.device, dtype=torch.bfloat16),
                    tiling_config=tiling,
                )
            ]
            video_decoder.to("cpu")
            audio = AudioPatchifier(patch_size=1).unpatchify(
                values["audio"],
                output_shape=AudioLatentShape(
                    batch=1,
                    channels=8,
                    frames=layout.audio_ranges[-1][1],
                    mel_bins=16,
                ),
            )
            audio_decoder.to(self.device)
            mel = audio_decoder(
                audio.to(device=self.device, dtype=next(audio_decoder.parameters()).dtype)
            )
            audio_decoder.to("cpu")
            vocoder.to(self.device)
            waveform = vocoder(mel).squeeze(0).float().cpu()
            return torch.cat(chunks, dim=1), waveform, vocoder.output_sampling_rate
        finally:
            for model in self._models:
                model.to("cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
