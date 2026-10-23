"""Frozen convolution VAE used for exact video handover between DMD segments."""

import hashlib
import json

import torch
from safetensors import safe_open

from .enums import LogVarianceType, NormLayerType, PaddingModeType
from .model import ConvVideoDecoder, VideoEncoder


def _prepare_video_encoder_kwargs(config: dict) -> dict:
    if "encoder" in config:
        encoder_config = config["encoder"]
        out_channels = encoder_config.get("out_channels", config.get("latent_channels", 128))
        encoder_blocks = encoder_config.get("blocks", encoder_config.get("encoder_blocks", []))
    else:
        encoder_config = config
        out_channels = config.get("latent_channels", 128)
        encoder_blocks = config.get("encoder_blocks", [])
    return {
        "convolution_dimensions": encoder_config.get("dims", config.get("dims", 3)),
        "in_channels": encoder_config.get("in_channels", 3),
        "out_channels": out_channels,
        "encoder_blocks": encoder_blocks,
        "patch_size": encoder_config.get("patch_size", 4),
        "norm_layer": NormLayerType(encoder_config.get("norm_layer", "pixel_norm")),
        "latent_log_var": LogVarianceType(encoder_config.get("latent_log_var", "uniform")),
        "encoder_spatial_padding_mode": PaddingModeType(
            encoder_config.get(
                "spatial_padding_mode", config.get("encoder_spatial_padding_mode", "zeros")
            )
        ),
    }


def _build_conv_video_decoder(config: dict) -> ConvVideoDecoder:
    """Build a ``ConvVideoDecoder`` from a ``vae`` configuration dictionary."""
    return ConvVideoDecoder(
        convolution_dimensions=config.get("dims", 3),
        in_channels=config.get("latent_channels", 128),
        out_channels=config.get("out_channels", 3),
        decoder_blocks=config.get("decoder_blocks", []),
        patch_size=config.get("patch_size", 4),
        norm_layer=NormLayerType(config.get("norm_layer", "pixel_norm")),
        causal=config.get("causal_decoder", False),
        timestep_conditioning=config.get("timestep_conditioning", True),
        decoder_spatial_padding_mode=PaddingModeType(config.get("spatial_padding_mode", "reflect")),
        base_channels=config.get("decoder_base_channels", 128),
    )


def video_vae_identity(path):
    """Content identity of VAE tensors/config, independent of staging path and mtime."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(header_size))
        config = json.loads(header["__metadata__"]["config"])["vae"]
        digest.update(json.dumps(config, sort_keys=True).encode())
        keys = sorted(key for key in header if key.startswith("vae."))
        if not keys:
            raise ValueError("base checkpoint has no video VAE tensors")
        for key in keys:
            entry = header[key]
            digest.update(json.dumps([key, entry["dtype"], entry["shape"]]).encode())
            start, end = entry["data_offsets"]
            stream.seek(8 + header_size + start)
            remaining = end - start
            while remaining:
                chunk = stream.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("truncated video VAE checkpoint")
                digest.update(chunk)
                remaining -= len(chunk)
    return digest.hexdigest()


def load_video_vae(path):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        config = json.loads(handle.metadata()["config"])["vae"]
        if config.get("_class_name", "CausalVideoAutoencoder") != "CausalVideoAutoencoder":
            raise ValueError("long tuning requires the reference convolution video VAE")
        with torch.device("meta"):
            encoder = VideoEncoder(**_prepare_video_encoder_kwargs(config))
            decoder = _build_conv_video_decoder(config)
        for name, model in (("encoder", encoder), ("decoder", decoder)):
            state = {}
            for key in model.state_dict():
                source = (
                    f"vae.{key}"
                    if key.startswith("per_channel_statistics.")
                    else f"vae.{name}.{key}"
                )
                state[key] = handle.get_tensor(source).to(dtype=torch.bfloat16)
            model.load_state_dict(state, strict=True, assign=True)
            model.requires_grad_(False).eval()
    return (encoder, decoder)


class VideoHandover:
    def __init__(self, path):
        (self.encoder, self.decoder) = load_video_vae(path)

    @torch.no_grad()
    def __call__(self, tail, *, height, width):
        device = tail.device
        latent = tail.reshape(1, -1, height, width, 128).permute(0, 4, 1, 2, 3)
        # Only the USP source runs the stochastic decoder. Preserve the caller's
        # RNG so subsequent rollout/score noise stays synchronized with its peers.
        # Forking retains the incoming random stream and hence the decoded result.
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.autocast(device.type, enabled=False):
            try:
                self.decoder.to(device)
                pixels = self.decoder(latent.to(torch.bfloat16))[:, :, -1:].contiguous()
            finally:
                self.decoder.cpu()
            try:
                self.encoder.to(device)
                encoded = self.encoder(pixels)
                if isinstance(encoded, (tuple, list)):
                    encoded = encoded[0]
                encoded = encoded[:, :128].to(torch.bfloat16)
            finally:
                self.encoder.cpu()
        if encoded.shape[2:] != (1, height, width):
            raise ValueError("video handover must encode to one latent frame")
        return encoded.permute(0, 2, 3, 4, 1).reshape(1, height * width, 128)
