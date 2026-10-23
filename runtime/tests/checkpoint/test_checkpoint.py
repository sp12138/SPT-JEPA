import os
import random
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import replace

import numpy as np
import pytest
import torch

from vforge.training.checkpoint import (
    CheckpointConfig,
    CheckpointManager,
    CheckpointManifest,
    apply_retention,
    cast_state_dict_for_weights_only,
    checkpoint_path,
    discover_checkpoints,
    load_dcp,
    load_model_component_weights,
    load_model_weights,
    read_latest_checkpoint,
    read_manifest,
    resolve_checkpoint_path,
    save_dcp,
    save_weights_only_pt,
)
from vforge.utils.rng import RandomStreams


def test_state_dict_mapper_strips_wrapper_prefix() -> None:
    from vforge.training.checkpoint import StateDictMapper

    mapper = StateDictMapper.strip_prefix("transformer.")
    internal = {"transformer.block.weight": torch.ones(1)}

    external = mapper.export(internal)

    assert set(external) == {"block.weight"}
    assert mapper.import_(external).keys() == internal.keys()


def test_checkpoint_config_rejects_full_pt():
    try:
        CheckpointConfig(
            enabled=True,
            every_n_steps=20,
            dirpath="/tmp/ckpts",
            state="full",
            format="pt",
        )
    except ValueError as exc:
        assert "full+pt" in str(exc)
    else:
        raise AssertionError("expected ValueError for full+pt")


def test_checkpoint_config_limits_async_save_to_distributed_formats():
    with pytest.raises(ValueError, match="requires format='dcp'"):
        CheckpointConfig(
            enabled=True,
            every_n_steps=1,
            dirpath="/tmp/ckpts",
            async_save=True,
        )

    config = CheckpointConfig(
        enabled=True,
        every_n_steps=1,
        dirpath="/tmp/ckpts",
        format="dcp",
        async_save=True,
    )
    assert config.async_save

    native = CheckpointConfig(
        enabled=True,
        every_n_steps=1,
        dirpath="/tmp/ckpts",
        format="mcore",
        async_save=True,
    )
    assert native.async_save


def test_checkpoint_config_remote_upload_requires_async_dcp():
    with pytest.raises(ValueError, match="async_save=true"):
        CheckpointConfig(
            enabled=True,
            every_n_steps=1,
            dirpath="/s3/checkpoints",
            state="full",
            format="dcp",
            remote_upload=True,
        )

    config = CheckpointConfig(
        enabled=True,
        every_n_steps=1,
        dirpath="/s3/checkpoints",
        state="full",
        format="dcp",
        async_save=True,
        remote_upload=True,
    )
    assert config.remote_upload


def test_component_integrity_hashes_shared_mcore_payload_once(tmp_path, monkeypatch):
    from vforge.training.checkpoint import storage as checkpoint_storage

    payload = tmp_path / "mcore"
    payload.mkdir()
    (payload / "native.distcp").write_bytes(b"checkpoint")
    manifest = replace(
        CheckpointManifest.create(state="full", completed_steps=1, format="mcore"),
        components={"weights": "mcore", "optimizer": "mcore"},
        components_present={"weights": True, "optimizer": True, "trainer": False},
    )
    calls = []
    original = checkpoint_storage._component_integrity

    def observe(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(checkpoint_storage, "_component_integrity", observe)
    updated = checkpoint_storage._with_component_integrity(str(tmp_path), manifest)

    assert calls == [str(payload)]
    integrity = updated.fingerprints["component_integrity"]
    assert integrity["weights"] == integrity["optimizer"]


def test_should_save_uses_completed_steps():
    config = CheckpointConfig(
        enabled=True,
        every_n_steps=20,
        dirpath="/tmp/ckpts",
    )
    assert not config.should_save(0)
    assert not config.should_save(19)
    assert config.should_save(20)
    assert config.should_save(60)
    assert config.should_save(61, is_final=True)
    assert not config.should_save(61, is_final=False)


def test_cast_state_dict_for_weights_only_fp32():
    payload = cast_state_dict_for_weights_only(
        {
            "w": torch.ones(2, dtype=torch.float32),
            "bf16": torch.ones(2, dtype=torch.bfloat16),
            "mask": torch.ones(2, dtype=torch.int64),
        }
    )
    assert payload["w"].dtype == torch.float32
    assert payload["bf16"].dtype == torch.float32
    assert payload["mask"].dtype == torch.int64


def test_save_weights_only_pt_roundtrip(tmp_path):
    model = torch.nn.Linear(3, 2)
    path = checkpoint_path(
        CheckpointConfig(
            enabled=True,
            every_n_steps=1,
            dirpath=str(tmp_path),
        ),
        20,
    )
    saved = save_weights_only_pt(model, path, is_rank_zero=True)
    assert saved == path
    assert os.path.isfile(path)
    loaded = torch.load(path, map_location="cpu")
    assert set(loaded) == set(model.state_dict())
    assert loaded["weight"].dtype == torch.float32


def test_mapped_pt_uses_local_staging_for_upload_and_download(
    monkeypatch,
    tmp_path,
):
    import vforge.utils.s3_staging as s3_staging
    from vforge.utils.s3_staging import (
        S3MountConfig,
        StorageConfig,
        activate_storage_config,
        restore_storage_config,
    )

    mount = tmp_path / "s3"
    path = mount / "checkpoints" / "00001.pt"
    local_root = tmp_path / "local"
    monkeypatch.setenv("VFORGE_CKPT_LOCAL_ROOT", str(local_root))
    monkeypatch.setenv("VFORGE_NODE_IO_INLINE", "1")
    uploaded = {}

    def fake_upload(source, *, s3, transfer_concurrency):
        assert transfer_concurrency == 32
        uploaded[(s3.bucket, s3.key_prefix)] = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )

    def fake_download(destination, *, s3, transfer_concurrency):
        assert transfer_concurrency == 32
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        torch.save(uploaded[(s3.bucket, s3.key_prefix)], destination)

    monkeypatch.setattr(s3_staging, "upload_file_crt", fake_upload)
    monkeypatch.setattr(s3_staging, "download_file_crt", fake_download)
    token = activate_storage_config(
        StorageConfig(s3_mounts=(S3MountConfig(str(mount), "bucket", "region"),))
    )
    try:
        source = torch.nn.Linear(3, 2)
        save_weights_only_pt(source, str(path), is_rank_zero=True)
        assert not path.exists()
        assert ("bucket", "checkpoints/00001.pt") in uploaded
        target = torch.nn.Linear(3, 2)
        load_model_weights(target, str(path), format="pt")
    finally:
        restore_storage_config(token)

    assert set(target.state_dict()) == set(source.state_dict())
    assert any(local_root.rglob("00001.pt"))


def test_unmapped_pt_honors_filesystem_staging_fallback(
    monkeypatch,
    tmp_path,
):
    source_model = torch.nn.Linear(2, 1)
    source = tmp_path / "efs" / "model.pt"
    source.parent.mkdir()
    torch.save(source_model.state_dict(), source)
    local_root = tmp_path / "local"
    monkeypatch.setenv("VFORGE_CKPT_LOCAL_ROOT", str(local_root))

    target = torch.nn.Linear(2, 1)
    load_model_weights(
        target,
        str(source),
        format="pt",
        stage_remote=True,
    )

    assert any(local_root.rglob("model.pt"))
    assert torch.equal(target.weight, source_model.weight)


def test_pt_cpu_loader_prefers_weights_only_mmap(monkeypatch):
    import vforge.training.checkpoint.storage as storage

    calls = []

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return {"weight": torch.ones(1)}

    monkeypatch.setattr(storage.torch, "load", fake_load)

    payload = storage._load_pt_cpu("/remote/model.pt")

    assert set(payload) == {"weight"}
    assert calls == [
        (
            "/remote/model.pt",
            {
                "map_location": "cpu",
                "weights_only": True,
                "mmap": True,
            },
        )
    ]


def test_distributed_pt_load_reads_only_on_global_rank_zero(monkeypatch):
    import torch.distributed.checkpoint.state_dict as state_dict_api

    import vforge.training.checkpoint.storage as storage

    model = torch.nn.Linear(1, 1)
    loaded_ranks = []
    distributed_calls = []

    monkeypatch.setattr(storage.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(storage.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(
        storage,
        "_load_pt_cpu",
        lambda _path: loaded_ranks.append(1),
    )

    def fake_set_model_state_dict(model, *, model_state_dict, options):
        distributed_calls.append((model, model_state_dict, options))

    monkeypatch.setattr(
        state_dict_api,
        "set_model_state_dict",
        fake_set_model_state_dict,
    )

    storage.load_model_weights(model, "/remote/model.pt", format="pt")

    assert loaded_ranks == []
    assert len(distributed_calls) == 1
    assert distributed_calls[0][1] == {}
    assert distributed_calls[0][2].full_state_dict is True
    assert distributed_calls[0][2].broadcast_from_rank0 is True


def test_distributed_pt_callback_gathers_target_but_filters_only_on_rank_zero(
    monkeypatch,
):
    import torch.distributed.checkpoint.state_dict as state_dict_api

    import vforge.models.loading as loading
    import vforge.training.checkpoint.storage as storage
    from vforge.models.loading import CompatibleLoadCallbacks

    model = torch.nn.Linear(1, 1)
    gathered = []
    monkeypatch.setattr(storage.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(storage.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(
        storage,
        "gather_full_model_state_dict",
        lambda current: gathered.append(current) or current.state_dict(),
    )
    monkeypatch.setattr(
        loading,
        "filter_compatible_state_dict",
        lambda *_args, **_kwargs: pytest.fail("nonzero rank filtered PT state"),
    )
    monkeypatch.setattr(
        state_dict_api,
        "set_model_state_dict",
        lambda *_args, **_kwargs: None,
    )

    storage.load_model_weights(
        model,
        "/remote/model.pt",
        format="pt",
        compatible_load_callbacks=CompatibleLoadCallbacks(),
    )

    assert gathered == [model]


def test_weights_only_pt_uses_explicit_state_dict_mapper(tmp_path):
    from vforge.training.checkpoint import StateDictMapper

    class Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = torch.nn.Linear(2, 2)

    source = Wrapper()
    path = str(tmp_path / "mapped.pt")
    mapper = StateDictMapper.strip_prefix("transformer.")

    save_weights_only_pt(
        source,
        path,
        is_rank_zero=True,
        state_dict_mapper=mapper,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload) == {"weight", "bias"}

    restored = Wrapper()
    load_model_weights(
        restored,
        path,
        format="pt",
        state_dict_mapper=mapper,
    )
    assert torch.equal(
        restored.transformer.weight,
        source.transformer.weight,
    )


def test_pt_compatible_load_preserves_shape_mismatch(tmp_path):
    source = torch.nn.Linear(2, 2)
    target = torch.nn.Linear(3, 2)
    original_weight = target.weight.detach().clone()
    path = str(tmp_path / "compatible.pt")
    torch.save(source.state_dict(), path)

    with pytest.warns(RuntimeWarning, match="shape_mismatch"):
        load_model_weights(target, path, format="pt", strict=False)

    assert torch.equal(target.weight, original_weight)
    assert torch.equal(target.bias, source.bias)
    with pytest.raises(RuntimeError):
        load_model_weights(target, path, format="pt", strict=True)


def test_pt_compatible_load_applies_application_callback(tmp_path):
    from vforge.models.loading import CompatibleLoadCallbacks

    source = torch.nn.Linear(2, 2)
    target = torch.nn.Linear(3, 2)
    original_weight = target.weight.detach().clone()
    path = str(tmp_path / "compatible-callback.pt")
    torch.save(source.state_dict(), path)

    def resolve_shape(_key, source_tensor, target_tensor):
        resolved = target_tensor.detach().clone()
        resolved[:, : source_tensor.shape[1]].copy_(source_tensor)
        return resolved

    load_model_weights(
        target,
        path,
        format="pt",
        strict=False,
        compatible_load_callbacks=CompatibleLoadCallbacks(
            resolve_shape_mismatch=resolve_shape,
        ),
    )

    assert torch.equal(target.weight[:, :2], source.weight)
    assert torch.equal(target.weight[:, 2:], original_weight[:, 2:])


@pytest.mark.parametrize("format", ["pt", "dcp"])
def test_component_weight_load_writes_only_its_global_subtree(tmp_path, format):
    class Pair(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wan = torch.nn.Linear(2, 2, bias=False)
            self.wan_2 = torch.nn.Linear(2, 2, bias=False)

    source = Pair()
    target = Pair()
    with torch.no_grad():
        source.wan.weight.fill_(7)
        source.wan_2.weight.fill_(9)
        target.wan.weight.zero_()
        target.wan_2.weight.fill_(3)
    path = str(tmp_path / ("high.pt" if format == "pt" else "00001.dcp"))
    if format == "pt":
        torch.save({"wan.weight": source.wan.weight.detach()}, path)
    else:
        save_dcp(
            source,
            None,
            path,
            manifest=CheckpointManifest.create(state="weights", completed_steps=1),
        )

    load_model_component_weights(target, path, component_path="wan", format=format)

    assert torch.equal(target.wan.weight, source.wan.weight)
    assert torch.count_nonzero(target.wan_2.weight == 3) == target.wan_2.weight.numel()


def test_pt_component_load_rejects_keys_from_another_subtree(tmp_path):
    path = str(tmp_path / "wrong.pt")
    torch.save({"wan_2.weight": torch.ones(2, 2)}, path)
    model = torch.nn.Module()
    model.wan = torch.nn.Linear(2, 2, bias=False)

    with pytest.raises(ValueError, match="contains keys outside"):
        load_model_component_weights(model, path, component_path="wan", format="pt")


def test_mcore_component_load_exposes_only_the_selected_global_subtree(tmp_path, monkeypatch):
    from vforge.training.checkpoint import backend
    from vforge.training.checkpoint.manifest import write_manifest

    class Pair(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wan = torch.nn.Linear(2, 2, bias=False)
            self.wan_2 = torch.nn.Linear(2, 2, bias=False)

    path = tmp_path / "high.mcore"
    path.mkdir()
    manifest = CheckpointManifest.create(state="weights", completed_steps=3, format="mcore")
    write_manifest(str(path), manifest)
    seen = []

    def fake_load(model, optimizer, checkpoint_path, **kwargs):
        assert optimizer is None
        assert checkpoint_path == str(path)
        assert set(model.state_dict()) == {"wan.weight"}
        seen.append(kwargs)
        return manifest, None

    monkeypatch.setattr(backend, "load_mcore_training_state", fake_load)
    model = Pair()

    loaded = load_model_component_weights(
        model,
        str(path),
        component_path="wan",
        format="mcore",
    )

    assert loaded is manifest
    assert seen[0]["weights_strict"] is True
    assert seen[0]["resume"].optimizer is False


def test_dcp_compatible_load_preserves_shape_mismatch(tmp_path):
    source = torch.nn.Linear(2, 2)
    path = str(tmp_path / "00001.dcp")
    save_dcp(
        source,
        None,
        path,
        manifest=CheckpointManifest.create(state="weights", completed_steps=1),
    )
    target = torch.nn.Linear(3, 2)
    original_weight = target.weight.detach().clone()

    with pytest.warns(RuntimeWarning, match="shape_mismatch"):
        load_model_weights(target, path, format="dcp", strict=False)

    assert torch.equal(target.weight, original_weight)
    assert torch.equal(target.bias, source.bias)
    with pytest.raises(ValueError, match="fingerprint"):
        load_model_weights(target, path, format="dcp", strict=True)


def test_dcp_compatible_load_resolves_missing_model_key(tmp_path):
    from vforge.models.loading import CompatibleLoadCallbacks

    class Source(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embedding = torch.nn.Linear(4, 2)

    class Target(Source):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embedding_ref = torch.nn.Linear(2, 2)

    source = Source()
    with torch.no_grad():
        source.patch_embedding.weight.copy_(torch.arange(8).reshape(2, 4))
        source.patch_embedding.bias.copy_(torch.tensor([8.0, 9.0]))
    path = str(tmp_path / "00002.dcp")
    save_dcp(
        source,
        None,
        path,
        manifest=CheckpointManifest.create(state="weights", completed_steps=2),
    )

    def resolve_missing(key, target, resolved):
        main = resolved.get(key.replace("patch_embedding_ref", "patch_embedding"))
        if main is None:
            return None
        return main.clone() if target.ndim == 1 else main[:, : target.shape[1]].clone()

    target = Target()
    with pytest.warns(RuntimeWarning, match="missing"):
        load_model_weights(
            target,
            path,
            format="dcp",
            strict=False,
            compatible_load_callbacks=CompatibleLoadCallbacks(
                resolve_missing_key=resolve_missing,
            ),
        )

    assert torch.equal(
        target.patch_embedding_ref.weight,
        source.patch_embedding.weight[:, :2],
    )
    assert torch.equal(
        target.patch_embedding_ref.bias,
        source.patch_embedding.bias,
    )


def test_dcp_path_uses_directory_suffix_by_default(tmp_path):
    config = CheckpointConfig(
        enabled=True,
        every_n_steps=1,
        dirpath=str(tmp_path),
        state="full",
        format="dcp",
    )
    assert checkpoint_path(config, 12).endswith("00012.dcp")


def test_mcore_path_and_manifest_are_discoverable(tmp_path):
    config = CheckpointConfig(
        enabled=True,
        every_n_steps=1,
        dirpath=str(tmp_path),
        state="full",
        format="mcore",
    )
    path = tmp_path / "00012.mcore"
    path.mkdir()
    manifest = CheckpointManifest.create(
        state="full",
        completed_steps=12,
        format="mcore",
    )
    (path / "manifest.json").write_text(
        __import__("json").dumps(manifest.to_dict()),
        encoding="utf-8",
    )

    assert checkpoint_path(config, 12) == str(path)
    assert read_manifest(str(path)).format == "mcore"
    assert discover_checkpoints(str(tmp_path)) == [(12, str(path))]


def test_mcore_remote_upload_is_rejected() -> None:
    with pytest.raises(ValueError, match="remote_upload requires format='dcp'"):
        CheckpointConfig(
            state="full",
            format="mcore",
            async_save=True,
            remote_upload=True,
        )


def test_discovery_ignores_incomplete_and_invalid_dcp(tmp_path):
    torch.save({}, tmp_path / "00001.pt")
    (tmp_path / "00002.dcp.incomplete").mkdir()
    (tmp_path / "00003.dcp").mkdir()
    valid = tmp_path / "00004.dcp"
    valid.mkdir()
    manifest = CheckpointManifest.create(state="weights", completed_steps=4)
    (valid / "manifest.json").write_text(
        __import__("json").dumps(manifest.to_dict()),
        encoding="utf-8",
    )

    assert discover_checkpoints(str(tmp_path)) == [
        (1, str(tmp_path / "00001.pt")),
        (4, str(valid)),
    ]
    (tmp_path / "latest").write_text("00004.dcp\n", encoding="utf-8")
    assert read_latest_checkpoint(str(tmp_path)) == str(valid)
    assert resolve_checkpoint_path("auto", str(tmp_path)) == str(valid)


def test_rng_state_roundtrip():
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    streams = RandomStreams()
    state = streams.state_dict()
    expected = (random.random(), np.random.rand(), torch.rand(3))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    streams.load_state_dict(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_weights_dcp_roundtrip_and_manifest(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = str(tmp_path / "00001.dcp")
    manifest = CheckpointManifest.create(state="weights", completed_steps=1)

    assert save_dcp(model, None, path, manifest=manifest) == path
    assert os.path.isdir(os.path.join(path, "weights"))
    assert not os.path.exists(os.path.join(path, "state"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    loaded_manifest, rank_state = load_dcp(model, None, path)
    assert rank_state is None
    assert loaded_manifest.completed_steps == 1
    durable_manifest = read_manifest(path)
    assert durable_manifest.state == "weights"
    assert durable_manifest.model_fingerprint is not None
    assert "weights" in durable_manifest.fingerprints["component_integrity"]
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])


def test_full_dcp_roundtrips_named_optimizers(tmp_path):
    from vforge.training.phased_trainer import NamedOptimizerState

    model = torch.nn.ModuleDict(
        {
            "generator": torch.nn.Linear(2, 2),
            "critic": torch.nn.Linear(2, 1),
        }
    )
    optimizers = NamedOptimizerState(
        optimizers={
            "generator": torch.optim.AdamW(model["generator"].parameters(), lr=2e-6),
            "critic": torch.optim.AdamW(model["critic"].parameters(), lr=4e-7),
        }
    )
    for name, module in model.items():
        optimizers.optimizers[name].zero_grad(set_to_none=True)
        module(torch.ones(1, 2)).square().mean().backward()
        optimizers.optimizers[name].step()
    expected_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_steps = {
        name: [float(state["step"]) for state in optimizer.state.values() if "step" in state]
        for name, optimizer in optimizers.optimizers.items()
    }
    path = str(tmp_path / "00001.dcp")
    save_dcp(
        model,
        optimizers,
        path,
        manifest=CheckpointManifest.create(state="full", completed_steps=1),
        rank_state={"scheduler": None, "dataloader": {}, "rng": RandomStreams().state_dict()},
    )

    restored_model = torch.nn.ModuleDict(
        {
            "generator": torch.nn.Linear(2, 2),
            "critic": torch.nn.Linear(2, 1),
        }
    )
    restored_optimizers = NamedOptimizerState(
        optimizers={
            "generator": torch.optim.AdamW(
                restored_model["generator"].parameters(),
                lr=2e-6,
            ),
            "critic": torch.optim.AdamW(
                restored_model["critic"].parameters(),
                lr=4e-7,
            ),
        }
    )

    load_dcp(restored_model, restored_optimizers, path)

    for name, value in restored_model.state_dict().items():
        assert torch.equal(value, expected_model[name])
    assert {
        name: [float(state["step"]) for state in optimizer.state.values() if "step" in state]
        for name, optimizer in restored_optimizers.optimizers.items()
    } == expected_steps
    manifest = read_manifest(path)
    assert manifest.optimizer_fingerprint["type"] == "named_optimizer_collection"


def test_full_dcp_integrity_detects_same_size_payload_corruption(tmp_path):
    model = torch.nn.Linear(3, 2)
    path = str(tmp_path / "00001.dcp")
    save_dcp(
        model,
        None,
        path,
        manifest=CheckpointManifest.create(state="weights", completed_steps=1),
    )
    shard = next((tmp_path / "00001.dcp" / "weights").glob("*.distcp"))
    payload = bytearray(shard.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    shard.write_bytes(payload)

    with pytest.raises(ValueError, match="integrity validation"):
        load_model_weights(model, path, format="dcp", integrity="full")


def test_metadata_dcp_integrity_does_not_read_payload(tmp_path, monkeypatch):
    from vforge.training.checkpoint import storage as checkpoint_storage

    component = tmp_path / "weights"
    component.mkdir()
    (component / "shard.distcp").write_bytes(b"payload")
    saved = checkpoint_storage._component_integrity(str(component))
    manifest = replace(
        CheckpointManifest.create(state="weights", completed_steps=1),
        fingerprints={"component_integrity": {"weights": saved}},
    )
    monkeypatch.setattr(
        checkpoint_storage,
        "_sha256_file",
        lambda _path: pytest.fail("metadata validation read file payload"),
    )

    checkpoint_storage._validate_component_integrity(
        str(tmp_path),
        manifest,
        "weights",
        mode="metadata",
    )


def test_distributed_metadata_integrity_is_local_without_broadcast(
    monkeypatch,
):
    from vforge.training.checkpoint import storage as checkpoint_storage

    metadata = {
        "file_count": 2,
        "total_bytes": 123,
        "metadata_digest": "metadata",
    }
    manifest = replace(
        CheckpointManifest.create(state="weights", completed_steps=1),
        fingerprints={"component_integrity": {"weights": metadata}},
    )
    reads = []
    monkeypatch.setattr(checkpoint_storage.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        checkpoint_storage.dist,
        "broadcast_object_list",
        lambda *_args, **_kwargs: pytest.fail(
            "metadata validation used a distributed object broadcast"
        ),
    )
    monkeypatch.setattr(
        checkpoint_storage,
        "_component_metadata",
        lambda path: reads.append(path) or metadata,
    )

    checkpoint_storage._validate_component_integrity(
        "/checkpoint",
        manifest,
        "weights",
        distributed=True,
        mode="metadata",
    )

    assert reads == ["/checkpoint/weights"]


@pytest.mark.parametrize(("rank", "expected_reads"), [(0, 1), (1, 0)])
def test_distributed_integrity_reads_component_only_on_rank_zero(
    monkeypatch,
    rank,
    expected_reads,
):
    from vforge.training.checkpoint import storage as checkpoint_storage

    integrity = {
        "file_count": 2,
        "total_bytes": 123,
        "digest": "digest",
    }
    manifest = replace(
        CheckpointManifest.create(state="weights", completed_steps=1),
        fingerprints={"component_integrity": {"weights": integrity}},
    )
    reads = []

    def component_integrity(path):
        reads.append(path)
        return integrity

    def broadcast_object_list(values, src):
        assert src == 0
        if rank != 0:
            values[0] = {"integrity": integrity, "error": None}

    monkeypatch.setattr(checkpoint_storage.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(checkpoint_storage.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(
        checkpoint_storage.dist,
        "broadcast_object_list",
        broadcast_object_list,
    )
    monkeypatch.setattr(
        checkpoint_storage,
        "_component_integrity",
        component_integrity,
    )

    checkpoint_storage._validate_component_integrity(
        "/checkpoint",
        manifest,
        "weights",
        distributed=True,
        mode="full",
    )

    assert len(reads) == expected_reads


def test_dcp_conversion_validates_input_only_once(tmp_path, monkeypatch):
    from torch.distributed.checkpoint import format_utils

    from vforge.training.checkpoint import storage as checkpoint_storage

    manifest = CheckpointManifest.create(state="weights", completed_steps=1)
    validations = []
    monkeypatch.setattr(
        checkpoint_storage,
        "read_manifest",
        lambda _path: manifest,
    )
    monkeypatch.setattr(
        checkpoint_storage,
        "_validate_component_integrity",
        lambda *args, **kwargs: validations.append((args, kwargs)),
    )
    monkeypatch.setattr(
        checkpoint_storage,
        "_component_path",
        lambda *_args: str(tmp_path / "weights"),
    )
    monkeypatch.setattr(
        checkpoint_storage,
        "_conversion_estimate",
        lambda *_args: {
            "estimated_peak_bytes": 1,
            "available_bytes": 2,
        },
    )

    def dcp_to_torch_save(_weights_dir, output):
        torch.save({"model": {"weight": torch.ones(1)}}, output)

    monkeypatch.setattr(
        format_utils,
        "dcp_to_torch_save",
        dcp_to_torch_save,
    )

    output = tmp_path / "model.pt"
    assert checkpoint_storage.convert_dcp_weights_to_pt(
        str(tmp_path / "input.dcp"),
        str(output),
    ) == str(output)
    assert len(validations) == 1


@pytest.fixture
def async_checkpoint(tmp_path, monkeypatch):
    from vforge.training.checkpoint import storage as checkpoint_storage

    uploads = []
    all_uploads = []
    monkeypatch.setattr(
        checkpoint_storage,
        "_get_dcp_state",
        lambda model, optimizer: {"model": {}},
    )
    monkeypatch.setattr(
        checkpoint_storage,
        "_dcp_save",
        lambda state, checkpoint_id, **kwargs: uploads.pop(0).result(),
    )
    monkeypatch.setattr(
        checkpoint_storage,
        "_with_component_integrity",
        lambda root, manifest: manifest,
    )
    config = CheckpointConfig(
        enabled=True,
        every_n_steps=1,
        dirpath=str(tmp_path),
        format="dcp",
        async_save=True,
    )
    manager = CheckpointManager(config)
    model = torch.nn.Linear(2, 2)

    def save(step, upload):
        uploads.append(upload)
        all_uploads.append(upload)
        path = str(tmp_path / f"{step:05d}.dcp")
        pending = manager.save(
            model,
            None,
            path,
            manifest=CheckpointManifest.create(
                state="weights",
                completed_steps=step,
            ),
        )
        return path, pending

    yield manager, save
    for upload in all_uploads:
        if not upload.done():
            upload.set_exception(RuntimeError("test teardown"))
    try:
        manager.close()
    except BaseException:
        pass


def test_async_dcp_commits_only_after_upload(tmp_path, async_checkpoint):
    manager, save = async_checkpoint
    upload = Future()
    path, pending = save(1, upload)

    assert not pending.done()
    assert not os.path.exists(path)
    assert os.path.isdir(path + ".incomplete")
    upload.set_result(None)
    completed = manager.wait()

    assert completed.path == path
    assert completed.completed_steps == 1
    assert completed.foreground_s >= completed.stage_s
    assert read_manifest(path).completed_steps == 1
    assert read_latest_checkpoint(str(tmp_path)) == path


def test_async_dcp_propagates_upload_error(tmp_path, async_checkpoint):
    manager, save = async_checkpoint
    upload = Future()
    upload.set_exception(OSError("storage unavailable"))
    path, _ = save(1, upload)

    with pytest.raises(OSError, match="storage unavailable"):
        manager.wait()
    assert not os.path.exists(path)
    assert os.path.isdir(path + ".incomplete")


def test_async_dcp_retention_error_does_not_invalidate_commit(
    tmp_path,
    monkeypatch,
    async_checkpoint,
):
    from vforge.training.checkpoint import storage as checkpoint_storage

    manager, save = async_checkpoint
    upload = Future()
    upload.set_result(None)
    monkeypatch.setattr(
        checkpoint_storage,
        "apply_retention",
        lambda config: (_ for _ in ()).throw(OSError("retention denied")),
    )
    path, _ = save(1, upload)

    with pytest.warns(RuntimeWarning, match="retention failed"):
        completed = manager.wait()
    assert completed.path == path
    assert read_manifest(path).completed_steps == 1


def test_async_dcp_waits_for_previous_request(tmp_path, async_checkpoint):
    manager, save = async_checkpoint
    first_upload = Future()
    second_upload = Future()
    first_path, _ = save(1, first_upload)

    with ThreadPoolExecutor(max_workers=1) as executor:
        second_save = executor.submit(save, 2, second_upload)
        with pytest.raises(TimeoutError):
            second_save.result(timeout=0.05)
        first_upload.set_result(None)
        second_path, second_pending = second_save.result(timeout=2)

    assert second_pending.wait_s > 0.0
    second_upload.set_result(None)
    manager.wait()
    assert os.path.isdir(first_path)
    assert os.path.isdir(second_path)


def test_full_dcp_loads_sparse_optimizer_state(tmp_path):
    class SparseGradientModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Linear(3, 2)
            self.unused = torch.nn.Linear(3, 2)

        def forward(self, inputs):
            return self.used(inputs)

    torch.manual_seed(23)
    model = SparseGradientModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    assert model.unused.weight not in optimizer.state
    assert model.unused.bias not in optimizer.state

    expected_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_optimizer = {
        name: {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in optimizer.state[parameter].items()
        }
        for name, parameter in model.named_parameters()
        if parameter in optimizer.state
    }
    path = str(tmp_path / "00001.dcp")
    save_dcp(
        model,
        optimizer,
        path,
        manifest=CheckpointManifest.create(state="full", completed_steps=1),
        rank_state={"scheduler": None, "dataloader": {}, "rng": RandomStreams().state_dict()},
    )

    restored_model = SparseGradientModel()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    load_dcp(restored_model, restored_optimizer, path)

    for name, value in restored_model.state_dict().items():
        assert torch.equal(value, expected_model[name])
        if value.is_floating_point():
            assert value.dtype is torch.float32
    for name, parameter in restored_model.named_parameters():
        if name not in expected_optimizer:
            assert restored_optimizer.state.get(parameter, {}) == {}
            continue
        actual_state = restored_optimizer.state[parameter]
        expected_state = expected_optimizer[name]
        assert set(actual_state) == set(expected_state)
        for key, expected in expected_state.items():
            actual = actual_state[key]
            if torch.is_tensor(expected):
                assert torch.equal(actual, expected)
                if key in {"exp_avg", "exp_avg_sq"}:
                    assert actual.dtype is torch.float32
            else:
                assert actual == expected


def test_full_dcp_builds_and_applies_combined_state_once(
    monkeypatch,
    tmp_path,
):
    import vforge.training.checkpoint.storage as storage

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    path = str(tmp_path / "00001.dcp")
    save_dcp(
        model,
        optimizer,
        path,
        manifest=CheckpointManifest.create(state="full", completed_steps=1),
        rank_state={"scheduler": None, "dataloader": {}, "rng": RandomStreams().state_dict()},
    )
    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    original_get = storage._get_dcp_state
    original_set = storage._set_dcp_state
    get_calls = []
    set_calls = []

    def tracked_get(live_model, live_optimizer):
        get_calls.append(live_optimizer is not None)
        return original_get(live_model, live_optimizer)

    def tracked_set(live_model, live_optimizer, state, **kwargs):
        set_calls.append(live_optimizer is not None)
        return original_set(live_model, live_optimizer, state, **kwargs)

    monkeypatch.setattr(storage, "_get_dcp_state", tracked_get)
    monkeypatch.setattr(storage, "_set_dcp_state", tracked_set)

    load_dcp(restored_model, restored_optimizer, path)

    assert get_calls == [True]
    assert set_calls == [True]


def test_full_dcp_keeps_model_load_strict_with_partial_optimizer(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    path = str(tmp_path / "00001.dcp")
    save_dcp(
        model,
        optimizer,
        path,
        manifest=CheckpointManifest.create(state="full", completed_steps=1),
        rank_state={"scheduler": None, "dataloader": {}, "rng": RandomStreams().state_dict()},
    )

    incompatible_model = torch.nn.Sequential(
        torch.nn.Linear(3, 2),
        torch.nn.Linear(2, 1),
    )
    incompatible_optimizer = torch.optim.AdamW(
        incompatible_model.parameters(),
        lr=0.01,
    )
    with pytest.raises(
        ValueError,
        match="model fingerprint does not match checkpoint",
    ):
        load_dcp(
            incompatible_model,
            incompatible_optimizer,
            path,
        )


def test_full_dcp_topology_change_requires_weights_only_plan(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = str(tmp_path / "00001.dcp")
    manifest = replace(
        CheckpointManifest.create(state="full", completed_steps=1),
        world_size=2,
    )
    save_dcp(
        model,
        optimizer,
        path,
        manifest=manifest,
        rank_state={
            "scheduler": None,
            "dataloader": {},
            "rng": RandomStreams().state_dict(),
        },
    )
    with pytest.raises(ValueError, match="supports weights only"):
        load_dcp(
            model,
            optimizer,
            path,
            bit_exact=False,
        )


def test_retention_keeps_newest_matching_format(tmp_path):
    config = CheckpointConfig(
        enabled=True,
        every_n_steps=1,
        dirpath=str(tmp_path),
        keep_last=2,
    )
    for step in (1, 2, 3):
        torch.save({}, checkpoint_path(config, step))

    apply_retention(config)

    assert not (tmp_path / "00001.pt").exists()
    assert (tmp_path / "00002.pt").exists()
    assert (tmp_path / "00003.pt").exists()
