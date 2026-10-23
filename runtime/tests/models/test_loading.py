from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


def test_pretrained_safetensors_stages_explicit_s3_mount(
    monkeypatch,
    tmp_path,
) -> None:
    import vforge.utils.node_staging as node_staging
    from vforge.models.loading import (
        PretrainedWeightSource,
        stage_pretrained_weights,
    )
    from vforge.utils.s3_staging import (
        S3MountConfig,
        StorageConfig,
        activate_storage_config,
        restore_storage_config,
    )

    mount = tmp_path / "mount"
    source = mount / "model"
    source.mkdir(parents=True)
    (source / "model.safetensors").touch()
    destination = tmp_path / "local"
    calls = []

    def fake_stage(path, *, config):
        calls.append((path, config))
        return node_staging.StagingResult(
            source=path,
            destination=str(destination),
            copied=True,
            files=1,
            bytes=1,
            backend="s3-crt",
        )

    monkeypatch.setattr(node_staging, "stage_on_node", fake_stage)
    token = activate_storage_config(StorageConfig(s3_mounts=(S3MountConfig(str(mount), "bucket"),)))
    try:
        staged = stage_pretrained_weights(PretrainedWeightSource(path=str(source)))
    finally:
        restore_storage_config(token)

    assert staged.path == str(destination)
    assert calls[0][0] == str(source)


def test_pretrained_can_force_local_staging_for_unmapped_mount(
    monkeypatch,
    tmp_path,
) -> None:
    import vforge.utils.node_staging as node_staging
    from vforge.models.loading import (
        PretrainedWeightSource,
        stage_pretrained_weights,
    )

    source = tmp_path / "mounted-transformer"
    source.mkdir()
    (source / "model.safetensors").touch()
    destination = tmp_path / "local"
    calls = []

    def fake_stage(path, *, config):
        calls.append((path, config))
        return node_staging.StagingResult(
            source=path,
            destination=str(destination),
            copied=True,
            files=1,
            bytes=1,
        )

    monkeypatch.setattr(node_staging, "stage_on_node", fake_stage)
    staged = stage_pretrained_weights(PretrainedWeightSource(path=str(source), stage_to_local=True))

    assert staged.path == str(destination)
    assert calls[0][0] == str(source)


def test_cast_module_parameters_preserves_buffer_dtypes() -> None:
    from vforge.models.loading import cast_module_parameters

    model = nn.Linear(2, 2, dtype=torch.bfloat16)
    model.register_buffer(
        "rope",
        torch.ones(2, dtype=torch.float64),
        persistent=False,
    )

    cast_module_parameters(model, torch.float32)

    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}
    assert model.rope.dtype is torch.float64
    assert "rope" not in model.state_dict()


def test_cast_module_parameters_honors_keep_in_fp32_modules() -> None:
    from vforge.models.loading import cast_module_parameters

    class Block(nn.Module):
        _keep_in_fp32_modules = ["time_embedder", "scale_shift_table"]

        def __init__(self) -> None:
            super().__init__()
            self.time_embedder = nn.Linear(2, 2)
            self.proj = nn.Linear(2, 2)
            self.scale_shift_table = nn.Parameter(torch.ones(2))
            self.register_buffer("freqs", torch.ones(2, dtype=torch.float64))

    class Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = Block()

    model = Root()
    cast_module_parameters(model, torch.bfloat16)

    assert model.transformer.time_embedder.weight.dtype is torch.float32
    assert model.transformer.scale_shift_table.dtype is torch.float32
    assert model.transformer.proj.weight.dtype is torch.bfloat16
    assert model.transformer.freqs.dtype is torch.float64


def test_huggingface_safetensor_keys_reads_sharded_index(tmp_path) -> None:
    from vforge.models.loading import _huggingface_safetensor_keys

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "patch_embedding.weight": "model-00001-of-00002.safetensors",
                    "blocks.0.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    assert _huggingface_safetensor_keys(str(tmp_path)) == {
        "patch_embedding.weight",
        "blocks.0.weight",
    }


def test_selective_safetensors_loader_only_reads_indexed_shard(tmp_path) -> None:
    from safetensors.torch import save_file

    from vforge.models.loading import load_safetensors_tensors

    target_name = "model-00001-of-00002.safetensors"
    unused_name = "model-00002-of-00002.safetensors"
    expected = torch.arange(6).reshape(2, 3)
    save_file({"patch_embedding.weight": expected}, tmp_path / target_name)
    (tmp_path / unused_name).write_bytes(b"must not be opened")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "patch_embedding.weight": target_name,
                    "blocks.0.weight": unused_name,
                }
            }
        )
    )

    loaded = load_safetensors_tensors(
        str(tmp_path),
        {"patch_embedding.weight"},
    )

    assert set(loaded) == {"patch_embedding.weight"}
    assert torch.equal(loaded["patch_embedding.weight"], expected)


def test_selective_safetensors_loader_scans_metadata_without_index(
    tmp_path,
) -> None:
    from safetensors.torch import save_file

    from vforge.models.loading import load_safetensors_tensors

    expected = torch.arange(4)
    save_file({"first": torch.ones(1)}, tmp_path / "first.safetensors")
    save_file({"wanted": expected}, tmp_path / "second.safetensors")

    loaded = load_safetensors_tensors(str(tmp_path), {"wanted", "missing"})

    assert set(loaded) == {"wanted"}
    assert torch.equal(loaded["wanted"], expected)


def test_optimizer_main_param_state_preserves_fp32_checkpoint_values(
    monkeypatch,
    tmp_path,
) -> None:
    from safetensors.torch import save_file

    from vforge.models.loading import (
        PretrainedWeightSource,
        load_pretrained_optimizer_main_param_state,
    )

    source_dir = tmp_path / "transformer"
    source_dir.mkdir()
    expected = torch.tensor(
        [[1.001234, 2.003456], [3.005678, 4.007891]],
        dtype=torch.float32,
    )
    save_file({"weight": expected}, source_dir / "model.safetensors")
    model = nn.Module()
    model.tower = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    monkeypatch.setattr(
        "vforge.distributed.parallel_state.get_parallel_state",
        lambda: SimpleNamespace(config=SimpleNamespace(tp_size=1), tp_rank=0),
    )

    state = load_pretrained_optimizer_main_param_state(
        model,
        (PretrainedWeightSource(path=str(tmp_path), model_path="tower"),),
    )

    assert set(state) == {"tower.weight"}
    assert state["tower.weight"].dtype is torch.float32
    assert torch.equal(state["tower.weight"], expected)
    assert not torch.equal(state["tower.weight"], expected.bfloat16().float())


def test_optimizer_main_param_state_builds_tp_local_slice(monkeypatch, tmp_path) -> None:
    from safetensors.torch import save_file

    from vforge.models.loading import (
        PretrainedWeightSource,
        load_pretrained_optimizer_main_param_state,
    )

    source_dir = tmp_path / "transformer"
    source_dir.mkdir()
    expected = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    save_file({"weight": expected}, source_dir / "model.safetensors")
    model = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    model.weight.tensor_model_parallel = True
    model.weight.partition_dim = 1
    model.weight.partition_stride = 1
    monkeypatch.setattr(
        "vforge.distributed.parallel_state.get_parallel_state",
        lambda: SimpleNamespace(config=SimpleNamespace(tp_size=2), tp_rank=1),
    )

    state = load_pretrained_optimizer_main_param_state(
        model,
        (PretrainedWeightSource(path=str(tmp_path)),),
    )

    assert torch.equal(state["weight"], expected[:, 2:])
    assert state["weight"].is_contiguous()


def test_pretrained_weight_source_loads_fallback_into_wrapper(tmp_path) -> None:
    from vforge.models.loading import (
        PretrainedWeightSource,
        load_pretrained_weights,
    )

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Linear(2, 2)

    expected = nn.Linear(2, 2)
    with torch.no_grad():
        expected.weight.fill_(3)
        expected.bias.fill_(4)
    initialized = []
    source = PretrainedWeightSource(
        path=str(tmp_path / "missing"),
        model_prefix="transformer.",
        initialize_missing=lambda _model: initialized.append(True),
        fallback_model_factory=lambda: expected,
    )
    model = Wrapper()

    load_pretrained_weights(model, source)

    assert initialized == [True]
    assert torch.equal(model.transformer.weight, expected.weight)
    assert torch.equal(model.transformer.bias, expected.bias)


def test_pretrained_pt_without_factory_loads_selected_state_into_submodule(tmp_path) -> None:
    from vforge.models.loading import PretrainedWeightSource, load_pretrained_weights

    expected = nn.Linear(2, 2)
    checkpoint = tmp_path / "weights.pt"
    torch.save(
        {"weights": {f"module.{key}": value for key, value in expected.state_dict().items()}},
        checkpoint,
    )
    model = nn.ModuleDict({"tower": nn.Linear(2, 2)})
    load_pretrained_weights(
        model,
        PretrainedWeightSource(
            path=str(checkpoint),
            model_path="tower",
            state_dict_key="weights",
            source_prefix="module.",
        ),
        strict=True,
    )
    for key, value in model["tower"].state_dict().items():
        assert torch.equal(value, expected.state_dict()[key]), key


def test_after_load_initializer_sees_loaded_weights_and_exact_missing_keys(tmp_path) -> None:
    from vforge.models.loading import (
        PretrainedWeightSource,
        load_pretrained_weights,
    )

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.main = nn.Linear(2, 2)
            self.extra = nn.Linear(2, 2)

    loaded = nn.Module()
    loaded.main = nn.Linear(2, 2)
    with torch.no_grad():
        loaded.main.weight.fill_(9.0)
        loaded.main.bias.fill_(8.0)
    observed = []

    def initialize_after_load(model, missing):
        observed.append((model.main.weight.detach().clone(), missing))

    model = Model()
    load_pretrained_weights(
        model,
        PretrainedWeightSource(
            path=str(tmp_path / "missing"),
            fallback_model_factory=lambda: loaded,
            initialize_missing_after_load=initialize_after_load,
        ),
    )

    assert torch.equal(observed[0][0], loaded.main.weight)
    assert observed[0][1] == frozenset({"extra.weight", "extra.bias"})


def test_distributed_safetensors_compatible_load_skips_shape_mismatch(
    tmp_path,
) -> None:
    from safetensors.torch import save_file

    from vforge.models.loading import load_huggingface_safetensors_distributed

    source = nn.Linear(2, 2)
    source_dir = tmp_path / "transformer"
    source_dir.mkdir()
    save_file(source.state_dict(), source_dir / "model.safetensors")
    target = nn.Linear(3, 2)
    initial_weight = target.weight.detach().clone()

    with pytest.warns(RuntimeWarning, match="shape_mismatch"):
        load_huggingface_safetensors_distributed(
            target,
            str(tmp_path),
            subfolder="transformer",
            strict=False,
        )

    assert torch.equal(target.weight, initial_weight)
    assert torch.equal(target.bias, source.bias)
    with pytest.raises(RuntimeError, match="strictly match"):
        load_huggingface_safetensors_distributed(
            target,
            str(tmp_path),
            subfolder="transformer",
            strict=True,
        )


def test_distributed_safetensors_resolves_declared_missing_keys(tmp_path) -> None:
    from safetensors.torch import save_file

    from vforge.models.loading import load_huggingface_safetensors_distributed

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embedding = nn.Linear(4, 2)
            self.patch_embedding_ref = nn.Linear(2, 2)

    source_dir = tmp_path / "transformer"
    source_dir.mkdir()
    source = nn.Linear(4, 2)
    with torch.no_grad():
        source.weight.copy_(torch.arange(8).reshape(2, 4))
        source.bias.copy_(torch.tensor([8.0, 9.0]))
    save_file(
        {
            "patch_embedding.weight": source.weight.detach(),
            "patch_embedding.bias": source.bias.detach(),
        },
        source_dir / "model.safetensors",
    )

    def resolve_missing(key, target, resolved):
        main = resolved.get(key.replace("patch_embedding_ref", "patch_embedding"))
        if main is None:
            return None
        return main.clone() if target.ndim == 1 else main[:, : target.shape[1]].clone()

    target = Model()
    with pytest.warns(RuntimeWarning, match="resolved_missing"):
        load_huggingface_safetensors_distributed(
            target,
            str(tmp_path),
            subfolder="transformer",
            resolve_missing_key=resolve_missing,
            strict=True,
        )

    assert torch.equal(target.patch_embedding_ref.weight, source.weight[:, :2])
    assert torch.equal(target.patch_embedding_ref.bias, source.bias)


def test_compatible_load_callbacks_resolve_shape_and_missing_keys() -> None:
    from vforge.models.loading import (
        CompatibleLoadCallbacks,
        filter_compatible_state_dict,
    )

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embedding = nn.Linear(4, 2)
            self.patch_embedding_ref = nn.Linear(2, 2)

    model = Model()
    target_state = model.state_dict()
    target_state["patch_embedding.weight"] = torch.full((2, 4), -123.0)
    source = {
        "patch_embedding.weight": torch.arange(
            6,
            dtype=torch.float32,
        ).reshape(2, 3),
        "patch_embedding.bias": torch.tensor([7.0, 8.0]),
    }

    def resolve_shape(_key, source_tensor, target_tensor):
        resolved = target_tensor.detach().clone()
        resolved[:, : source_tensor.shape[1]].copy_(source_tensor)
        return resolved

    def resolve_missing(key, target_tensor, resolved_state):
        main_key = key.replace("patch_embedding_ref", "patch_embedding")
        main = resolved_state.get(main_key)
        if main is None:
            return None
        if target_tensor.ndim == 1:
            return main.clone()
        return main[:, : target_tensor.shape[1]].clone()

    with pytest.warns(RuntimeWarning, match="resolved_missing"):
        resolved = filter_compatible_state_dict(
            model,
            source,
            source="checkpoint.pt",
            callbacks=CompatibleLoadCallbacks(
                resolve_shape_mismatch=resolve_shape,
                resolve_missing_key=resolve_missing,
            ),
            target_state=target_state,
        )

    assert torch.equal(
        resolved["patch_embedding.weight"][:, :3],
        source["patch_embedding.weight"],
    )
    assert torch.equal(
        resolved["patch_embedding.weight"][:, 3:],
        target_state["patch_embedding.weight"][:, 3:],
    )
    assert torch.equal(
        resolved["patch_embedding_ref.weight"],
        source["patch_embedding.weight"][:, :2],
    )
    assert torch.equal(
        resolved["patch_embedding_ref.bias"],
        source["patch_embedding.bias"],
    )


def test_keep_in_fp32_module_paths_expand_fragments_to_outermost_targets() -> None:
    from vforge.models.loading import keep_in_fp32_module_paths

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(2)
            self.attn = nn.Linear(2, 2)
            self.scale_shift_table = nn.Parameter(torch.zeros(1, 2))

    class Root(nn.Module):
        _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1"]

        def __init__(self) -> None:
            super().__init__()
            self.time_embedder = nn.Sequential(nn.Linear(2, 2))
            self.time_proj = nn.Linear(2, 2)
            self.blocks = nn.ModuleList([Block()])

    assert keep_in_fp32_module_paths(Root()) == (
        "time_embedder",
        "blocks.0.norm1",
    )


def test_keep_in_fp32_module_paths_skip_targets_without_a_module() -> None:
    from vforge.models.loading import keep_in_fp32_module_paths

    class Root(nn.Module):
        _keep_in_fp32_modules = ["scale_shift_table", "norm1"]

        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(2)
            self.scale_shift_table = nn.Parameter(torch.zeros(1, 2))

    assert keep_in_fp32_module_paths(Root()) == ("norm1",)


def test_keep_in_fp32_module_paths_are_empty_without_a_declaration() -> None:
    from vforge.models.loading import keep_in_fp32_module_paths

    class Declared(nn.Module):
        _keep_in_fp32_modules = ["norm"]

        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(2)
            self.other = nn.LayerNorm(2)

    class Silent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(2)
            self.other = nn.LayerNorm(2)

    assert keep_in_fp32_module_paths(Declared()) == ("norm",)
    # The list is the only way to ask, so a tree that stays silent gets nothing.
    assert keep_in_fp32_module_paths(Silent()) == ()
