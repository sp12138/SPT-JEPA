"""Shared model and weight assembly for LTX DMD train and eval."""

import torch
from safetensors import safe_open

from vforge.models.loading import PretrainedWeightSource
from vforge.models.ltx.causal import CausalLTXModelConfigurator
from vforge.models.ltx.loading import map_transformer_checkpoint_keys
from vforge.models.ltx.model_configurator import LTXModelConfigurator
from vforge.models.ltx.ops import dmd_transformer_ops
from vforge.models.ltx.text_encoder import checkpoint_config


def build_model(*, pretrained_path, init_device, causal, param_dtype, ops=None):
    configurator = CausalLTXModelConfigurator if causal else LTXModelConfigurator
    model = configurator.from_config(
        checkpoint_config(pretrained_path),
        device=init_device,
        # USP/cache transforms own communication; the injected kernel stays local.
        ops=dmd_transformer_ops() if ops is None else ops,
    )
    model.to(dtype=param_dtype)
    model.video_embeddings_connector.to(dtype=torch.bfloat16)
    model.audio_embeddings_connector.to(dtype=torch.bfloat16)
    return model


def _connector_resolver(base_path):
    def resolve(key, target, _loaded):
        if not key.startswith(("video_embeddings_connector.", "audio_embeddings_connector.")):
            return None
        with safe_open(base_path, framework="pt", device="cpu") as handle:
            value = handle.get_tensor(f"model.diffusion_model.{key}")
        return value.to(dtype=target.dtype)

    return resolve


def build_weight_source(
    *,
    pretrained_path,
    weights_path,
    model_path,
    causal,
    param_dtype,
    source_prefix="",
    weight_loading="auto",
):
    resolver = _connector_resolver(pretrained_path)

    def load():
        model = build_model(
            pretrained_path=pretrained_path,
            init_device="meta",
            causal=causal,
            param_dtype=param_dtype,
        )
        state = {}
        with safe_open(weights_path or pretrained_path, framework="pt", device="cpu") as handle:
            keys = handle.keys()
            prefix = source_prefix
            if not weights_path:
                mapping, _ = map_transformer_checkpoint_keys(keys, set(model.state_dict()))
                keys = mapping.values()
                prefix = "model.diffusion_model."
            for name in keys:
                if prefix and not name.startswith(prefix):
                    raise ValueError(f"checkpoint key lacks configured prefix: {name}")
                key = name.removeprefix(prefix) if prefix else name
                state[key] = handle.get_tensor(name).to(dtype=torch.bfloat16)
        for key, target in model.state_dict().items():
            if key not in state:
                value = resolver(key, target, state)
                if value is not None:
                    state[key] = value.to(dtype=torch.bfloat16)
        model.load_state_dict(state, strict=True, assign=True)
        return model

    return PretrainedWeightSource(
        path=weights_path or pretrained_path,
        model_path=model_path,
        subfolder="",
        loading_mode="rank0_broadcast" if weight_loading == "auto" else weight_loading,
        fallback_model_factory=load,
        resolve_missing_key=resolver,
    )
