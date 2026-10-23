"""HF / PT / DCP load and PT / DCP write contracts for keep-in-fp32 weights.

An fp32 table next to a bf16 projection is the case where naming and dtype can
each drift on their own. Load paths have to land both the published name and the
storage dtype; portable PT export keeps the name but normalizes every float to
fp32, so these tests pin where each guarantee stops.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from vforge.models.loading import (
    cast_module_parameters,
    load_huggingface_safetensors_distributed,
)
from vforge.training.checkpoint import (
    CheckpointManifest,
    convert_dcp_weights_to_pt,
    load_model_weights,
    save_dcp,
    save_weights_only_pt,
)

PUBLISHED_KEYS = {"scale_shift_table", "proj.weight"}


class _ContractModule(nn.Module):
    _keep_in_fp32_modules = ["scale_shift_table"]

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self.scale_shift_table = nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(1, 2, 4))
        self.register_buffer(
            "rope",
            torch.ones(2, dtype=torch.float64),
            persistent=False,
        )


def _make_model(*, seed: int = 0) -> _ContractModule:
    torch.manual_seed(seed)
    model = _ContractModule()
    with torch.no_grad():
        model.proj.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4))
        model.scale_shift_table.copy_(torch.arange(8, dtype=torch.float32).reshape(1, 2, 4) + seed)
    cast_module_parameters(model, torch.bfloat16)
    return model


def _scramble(model: nn.Module) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(123)


def _assert_storage_dtypes(model: nn.Module) -> None:
    assert model.scale_shift_table.dtype is torch.float32
    assert model.proj.weight.dtype is torch.bfloat16
    assert model.rope.dtype is torch.float64
    assert set(model.state_dict()) == PUBLISHED_KEYS


def _assert_live_matches(
    model: nn.Module,
    *,
    table: torch.Tensor,
    proj: torch.Tensor,
) -> None:
    _assert_storage_dtypes(model)
    live_table = model.scale_shift_table.detach().cpu()
    live_proj = model.proj.weight.detach().cpu()
    assert torch.equal(live_table, table.cpu().to(live_table.dtype))
    assert torch.equal(live_proj, proj.cpu().to(live_proj.dtype))


def _snapshot(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        model.scale_shift_table.detach().cpu().clone(),
        model.proj.weight.detach().cpu().clone(),
    )


def _load_pt_file(path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    assert isinstance(state, dict)
    return state


def _dcp_logical_keys(path: str) -> set[str]:
    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(os.path.join(path, "weights")).read_metadata()
    keys = set(metadata.state_dict_metadata)
    prefix = "model."
    if keys and all(key.startswith(prefix) for key in keys):
        return {key[len(prefix) :] for key in keys}
    return keys


def _write_hf_published(tmp_path, *, table: torch.Tensor, proj: torch.Tensor) -> str:
    from safetensors.torch import save_file

    folder = tmp_path / "pretrained" / "transformer"
    folder.mkdir(parents=True)
    save_file(
        {
            "scale_shift_table": table.contiguous(),
            "proj.weight": proj.contiguous(),
        },
        folder / "model.safetensors",
    )
    return str(tmp_path / "pretrained")


def _save_pt(model: nn.Module, tmp_path, name: str = "weights.pt") -> str:
    path = str(tmp_path / name)
    saved = save_weights_only_pt(model, path, is_rank_zero=True)
    assert saved == path
    return path


def _save_weights_dcp(
    model: nn.Module,
    tmp_path,
    *,
    step: int = 1,
    subdir: str | None = None,
) -> str:
    root = tmp_path / subdir if subdir is not None else tmp_path
    root.mkdir(parents=True, exist_ok=True)
    path = str(root / f"{step:05d}.dcp")
    saved = save_dcp(
        model,
        None,
        path,
        manifest=CheckpointManifest.create(state="weights", completed_steps=step),
    )
    assert saved == path
    return path


def test_pt_write_publishes_keys_and_normalizes_floats_to_fp32(tmp_path) -> None:
    model = _make_model()
    _assert_storage_dtypes(model)

    state = _load_pt_file(_save_pt(model, tmp_path))

    # Portable export normalizes floats to fp32 whatever the storage dtype was.
    # Widening bf16 is lossless, so values still compare exactly rather than
    # approximately.
    assert set(state) == PUBLISHED_KEYS
    assert state["scale_shift_table"].dtype is torch.float32
    assert state["proj.weight"].dtype is torch.float32
    assert torch.equal(state["scale_shift_table"], model.scale_shift_table.cpu())
    assert torch.equal(state["proj.weight"], model.proj.weight.cpu().to(torch.float32))


def test_dcp_write_uses_the_same_published_keys(tmp_path) -> None:
    model = _make_model()

    assert _dcp_logical_keys(_save_weights_dcp(model, tmp_path)) == PUBLISHED_KEYS


def test_dcp_to_pt_matches_pt_write_keys_and_dtypes(tmp_path) -> None:
    model = _make_model()
    table, proj = _snapshot(model)
    dcp_path = _save_weights_dcp(model, tmp_path, subdir="source")
    direct = _load_pt_file(_save_pt(model, tmp_path, "direct.pt"))
    converted_path = str(tmp_path / "converted.pt")
    convert_dcp_weights_to_pt(dcp_path, converted_path)

    converted = _load_pt_file(converted_path)

    assert set(converted) == PUBLISHED_KEYS
    assert converted["scale_shift_table"].dtype is torch.float32
    assert converted["proj.weight"].dtype is torch.float32
    assert torch.equal(converted["scale_shift_table"], table)
    assert torch.equal(converted["proj.weight"], proj.to(torch.float32))
    assert torch.equal(converted["scale_shift_table"], direct["scale_shift_table"])
    assert torch.equal(converted["proj.weight"], direct["proj.weight"])


def test_hf_load_keeps_published_keys_and_storage_dtypes(tmp_path) -> None:
    try:
        from torch.distributed.checkpoint import HuggingFaceStorageReader
    except ImportError:
        pytest.skip("HuggingFaceStorageReader is unavailable")
    del HuggingFaceStorageReader

    table, proj = _snapshot(_make_model())
    pretrained = _write_hf_published(tmp_path, table=table, proj=proj)

    target = _make_model(seed=9)
    _scramble(target)
    load_huggingface_safetensors_distributed(
        target,
        pretrained,
        subfolder="transformer",
        strict=True,
    )

    _assert_live_matches(target, table=table, proj=proj)


def test_pt_load_round_trips_strictly(tmp_path) -> None:
    table, proj = _snapshot(source := _make_model())
    path = _save_pt(source, tmp_path)

    target = _make_model(seed=3)
    _scramble(target)
    load_model_weights(target, path, format="pt", strict=True)

    _assert_live_matches(target, table=table, proj=proj)


def test_dcp_load_round_trips_strictly(tmp_path) -> None:
    table, proj = _snapshot(source := _make_model())
    path = _save_weights_dcp(source, tmp_path)

    target = _make_model(seed=6)
    _scramble(target)
    load_model_weights(target, path, format="dcp", strict=True)

    _assert_live_matches(target, table=table, proj=proj)


def test_convert_then_pt_load_round_trips(tmp_path) -> None:
    table, proj = _snapshot(source := _make_model())
    converted = str(tmp_path / "from-dcp.pt")
    convert_dcp_weights_to_pt(_save_weights_dcp(source, tmp_path), converted)

    target = _make_model(seed=1)
    _scramble(target)
    load_model_weights(target, converted, format="pt", strict=False)

    _assert_live_matches(target, table=table, proj=proj)


def test_a_stem_only_key_does_not_fill_a_linear_weight(tmp_path) -> None:
    target = _make_model()
    original_proj = target.proj.weight.detach().clone()
    path = str(tmp_path / "stem-only.pt")
    torch.save(
        {
            "proj": torch.full((4, 4), 7.0),
            "scale_shift_table": torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
        },
        path,
    )

    with pytest.warns(RuntimeWarning):
        load_model_weights(target, path, format="pt", strict=False)

    assert torch.equal(target.proj.weight, original_proj)
    assert torch.equal(
        target.scale_shift_table.cpu(),
        torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
    )
