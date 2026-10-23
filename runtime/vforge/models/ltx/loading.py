"""Standalone LTX-2.3 safetensors loading."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import torch

from vforge.models.ltx.model import LTXModel
from vforge.models.ltx.model_configurator import LTXModelConfigurator
from vforge.models.ltx.transformer import DEFAULT_TRANSFORMER_OPS, TransformerOpsConfig

_TRANSFORMER_PREFIX = "model.diffusion_model."
_IGNORED_NON_TRANSFORMER_PREFIXES = (
    "audio_vae.",
    "language_model.",
    "multi_modal_projector.",
    "model_ema.",
    "text_encoder.",
    "text_embedding_projection.",
    "vae.",
    "video_vae.",
    "vision_tower.",
    "vocoder.",
)


def map_transformer_checkpoint_keys(
    source_keys: Iterable[str],
    expected_keys: set[str],
) -> tuple[dict[str, str], list[str]]:
    """Map checkpoint keys to model keys and reject ambiguous transformer state."""
    source_by_target: dict[str, str] = {}
    ignored: list[str] = []
    for source_name in source_keys:
        if source_name.startswith(_TRANSFORMER_PREFIX):
            target_name = source_name.removeprefix(_TRANSFORMER_PREFIX)
            if target_name not in expected_keys:
                raise RuntimeError(
                    f"unexpected prefixed transformer key: {source_name!r} maps to {target_name!r}"
                )
        elif source_name in expected_keys:
            target_name = source_name
        elif source_name.startswith(_IGNORED_NON_TRANSFORMER_PREFIXES):
            ignored.append(source_name)
            continue
        else:
            raise RuntimeError(f"unclassified checkpoint key: {source_name!r}")

        previous = source_by_target.get(target_name)
        if previous is not None:
            raise RuntimeError(
                "duplicate transformer key mapping: "
                f"{previous!r} and {source_name!r} both map to {target_name!r}"
            )
        source_by_target[target_name] = source_name
    return source_by_target, sorted(ignored)


def load_pretrained_model(
    pretrained_path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    connector_dtype: torch.dtype | None = None,
    device: torch.device | str = "cpu",
    ops: TransformerOpsConfig = DEFAULT_TRANSFORMER_OPS,
) -> LTXModel:
    """Load the transformer from an LTX model safetensors file."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("loading LTX weights requires safetensors") from exc

    path = str(pretrained_path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        raw_config = metadata.get("config")
        if raw_config is None:
            raise ValueError(f"LTX checkpoint metadata has no 'config' entry: {path}")
        config = json.loads(raw_config)
        model = LTXModelConfigurator.from_config(config, ops=ops, device="meta")
        expected = set(model.state_dict())
        source_by_target, _ = map_transformer_checkpoint_keys(handle.keys(), expected)
        state = {
            target_name: handle.get_tensor(source_name).to(
                device=device,
                dtype=_parameter_dtype(
                    target_name,
                    dtype=dtype,
                    connector_dtype=connector_dtype,
                ),
            )
            for target_name, source_name in source_by_target.items()
        }

    missing = expected - set(state)
    if missing:
        raise RuntimeError(f"LTX checkpoint is missing transformer keys: {sorted(missing)}")
    model.load_state_dict(state, strict=True, assign=True)
    return model


def build_ltx23_model(
    *,
    pretrained_path: str | Path,
    init_device: torch.device | str = "meta",
    extras: object | None = None,
    dtype: torch.dtype = torch.float32,
    connector_dtype: torch.dtype = torch.bfloat16,
    ops: TransformerOpsConfig = DEFAULT_TRANSFORMER_OPS,
) -> LTXModel:
    """Build the LTX-2.3 structure from checkpoint metadata without loading weights."""
    del extras
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("building LTX from checkpoint metadata requires safetensors") from exc

    path = str(pretrained_path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    raw_config = metadata.get("config")
    if raw_config is None:
        raise ValueError(f"LTX checkpoint metadata has no 'config' entry: {path}")
    model = LTXModelConfigurator.from_config(
        json.loads(raw_config),
        ops=ops,
        device=init_device,
    )
    model.to(dtype=dtype)
    model.video_embeddings_connector.to(dtype=connector_dtype)
    model.audio_embeddings_connector.to(dtype=connector_dtype)
    return model


def _parameter_dtype(
    name: str,
    *,
    dtype: torch.dtype,
    connector_dtype: torch.dtype | None,
) -> torch.dtype:
    if connector_dtype is not None and name.startswith(
        ("video_embeddings_connector.", "audio_embeddings_connector.")
    ):
        return connector_dtype
    return dtype


build_model = build_ltx23_model
