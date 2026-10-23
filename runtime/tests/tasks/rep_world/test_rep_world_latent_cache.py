import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module

from vforge.data.loader import build_resumable_loader
from vforge.ops import OpsConfig, initialize_ops, reset_ops
from vforge.tasks.rep_world.data import VideoClipDataset
from vforge.tasks.rep_world.latent_cache import SHARD_SCHEMA, CachedVideoLatents
from vforge.tasks.rep_world.shared import parse_config, validate_train_request
from vforge.tasks.rep_world.train import build_train_components
from vforge.tasks.requests import DataSourceSpec, TrainRequest
from vforge.training.recipe import StepContext


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data))


class Reader:
    def __init__(self, path):
        self.value = int(Path(path).stem)

    def __len__(self):
        return 32

    def get_batch(self, indices):
        pixels = np.full((len(indices), 256, 256, 3), self.value, dtype=np.uint8)
        return SimpleNamespace(asnumpy=lambda: pixels)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """A complete small artifact using the same public schema as extraction."""
    root = tmp_path / "cache"
    (root / "shards").mkdir(parents=True)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(dict(sample_id=str(i), rgb_path=f"{i}.mp4")) for i in range(8))
    )
    checkpoint, statistics = tmp_path / "weights.pth", tmp_path / "stats.pt"
    checkpoint.write_bytes(b"frozen codec identity")
    statistics.write_bytes(b"normalization identity")
    extras = dict(
        codec=dict(
            family="videorae", variant="3d", checkpoint=str(checkpoint), statistics=str(statistics)
        ),
        states=2,
        use_muon=False,
        world=dict(width=48, depth=1, heads=4, condition_dropout=0.1),
        conditioner=dict(
            input="latent", width=24, depth=1, heads=4, feature_layers=[0], context_mask_ratio=0.8
        ),
    )
    contract = dict(
        schema=SHARD_SCHEMA,
        source_manifest_sha256=digest(manifest),
        samples=8,
        codec=extras["codec"],
        latent_shape=[256, 2, 16, 16],
        latent_dtype="float32",
        stored_key="latents",
        storage_order="C",
        channel_index="plane * 64 + channel",
        states_per_clip=2,
        rgb_shape=[3, 32, 256, 256],
        frame_interval=1,
        frames=list(range(32)),
        resize=[256, 256],
        shard_size=4,
        precision="bf16 autocast and codec weights; FP32 return",
        assets={
            key: dict(path=str(path), sha256=digest(path))
            for key, path in [("checkpoint", checkpoint), ("statistics", statistics)]
        },
    )
    fp = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    contract["fingerprint"] = fp
    write_json(root / "contract.json", contract)
    latent = torch.randn(8, 256, 2, 16, 16, generator=torch.Generator().manual_seed(17))
    descriptors, index = [], []
    for sid in range(2):
        value = latent[sid * 4 : (sid + 1) * 4].numpy()
        dest = root / "shards" / f"{sid:06d}.npy"
        np.save(dest, value)
        records = []
        for position in range(4):
            i = sid * 4 + position
            h = hashlib.sha256(memoryview(value[position]).cast("B")).hexdigest()
            records.append(
                dict(
                    index=i,
                    row=position,
                    sample_id=str(i),
                    rgb_path=str(tmp_path / f"{i}.mp4"),
                    latent_sha256=h,
                    source_frames=32,
                    frame_start=0,
                    frame_interval=1,
                )
            )
            index.append(
                dict(
                    index=i,
                    row=position,
                    sample_id=str(i),
                    latent_sha256=h,
                    latent_path=f"shards/{sid:06d}.npy",
                )
            )
        descriptor = dict(
            shard=sid,
            count=4,
            file=dest.name,
            fingerprint=fp,
            sha256=digest(dest),
            verified_readback=True,
        )
        write_json(dest.with_suffix(".json"), {**descriptor, "records": records})
        descriptors.append(descriptor)
    (root / "index.jsonl").write_text("\n".join(json.dumps(row) for row in index))
    write_json(
        root / "COMPLETE.json",
        dict(
            status="complete",
            fingerprint=fp,
            samples=8,
            shards=descriptors,
            index_sha256=digest(root / "index.jsonl"),
        ),
    )
    monkeypatch.setattr("vforge.tasks.rep_world.data.open_video", Reader)
    state = SimpleNamespace(
        batch_group=None, is_batch_leader=True, dp_rank=0, config=SimpleNamespace(dp_size=1)
    )
    monkeypatch.setattr("vforge.tasks.rep_world.train.get_parallel_state", lambda: state)
    request = TrainRequest(
        extras=extras,
        train_manifest=str(manifest),
        sources=(DataSourceSpec(cache_dir=str(root)),),
        latent_cache_mode="offline",
        precision="bf16",
        batch_size=2,
        sampler_seed=31,
        learning_rate=1e-4,
    )
    return SimpleNamespace(
        root=root, manifest=manifest, request=request, latent=latent, state=state, fp=fp
    )


def reader(prepared, **kwargs):
    return CachedVideoLatents(
        prepared.root,
        prepared.manifest,
        config=parse_config(prepared.request.extras),
        precision="bf16",
        **kwargs,
    )


@pytest.mark.parametrize("random_clip", [True, False])
def test_cache_matches_sample_layout_rng_and_owns_storage(prepared, random_clip):
    extras = {**prepared.request.extras, "random_clip": random_clip}
    original = VideoClipDataset(prepared.manifest, frames=32, random_clip=random_clip)
    cached = CachedVideoLatents(
        prepared.root,
        prepared.manifest,
        config=parse_config(extras),
        precision="bf16",
        expected_fingerprint=prepared.fp,
    )
    for i in [0, 4, 7]:
        torch.manual_seed(18)
        before = original[i]
        expected_rng = torch.get_rng_state()
        torch.manual_seed(18)
        actual = cached[i]
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert actual["sample_id"] == before["sample_id"]
        assert actual["latents"].dtype == torch.float32 and actual["latents"].is_contiguous()
        assert torch.equal(actual["latents"], prepared.latent[i])
        actual["latents"].zero_()
        assert torch.equal(cached[i]["latents"], prepared.latent[i])


@pytest.mark.parametrize(
    "fault", ["incomplete", "manifest", "fingerprint", "statistics", "missing_shard", "index"]
)
def test_cache_rejects_mismatched_or_incomplete_artifacts(prepared, fault):
    if fault == "incomplete":
        (prepared.root / "COMPLETE.json").unlink()
    if fault == "manifest":
        prepared.manifest.write_text(prepared.manifest.read_text() + "\n")
    if fault == "statistics":
        Path(prepared.request.extras["codec"]["statistics"]).write_bytes(b"different normalization")
    if fault == "missing_shard":
        (prepared.root / "shards/000001.npy").unlink()
    if fault == "index":
        (prepared.root / "index.jsonl").write_text("{}\n")
    with pytest.raises(ValueError):
        reader(prepared, expected_fingerprint="wrong" if fault == "fingerprint" else prepared.fp)


@pytest.mark.parametrize("fault", ["dtype", "corruption", "window", "order", "path"])
def test_cache_never_substitutes_a_corrupt_row(prepared, fault):
    path = prepared.root / "shards/000000.npy"
    if fault == "dtype":
        np.save(path, np.load(path).astype(np.float16))
    if fault == "corruption":
        value = np.load(path)
        value[0, 0, 0, 0, 0] += 1
        np.save(path, value)
    if fault in {"window", "order", "path"}:
        meta = path.with_suffix(".json")
        value = json.loads(meta.read_text())
        value["records"][0][
            {"window": "frame_start", "order": "sample_id", "path": "rgb_path"}[fault]
        ] = 1
        write_json(meta, value)
    with pytest.raises(RuntimeError, match="Cached sample 0"):
        reader(prepared)[0]


def test_offline_preflight_and_normal_hydra_overrides(prepared):
    validate_train_request(prepared.request, "fsdp2")
    for candidate in [
        replace(prepared.request, sources=()),
        replace(prepared.request, sources=prepared.request.sources * 2),
    ]:
        with pytest.raises(ValueError):
            validate_train_request(candidate, "fsdp2")
    from vforge.tasks.common import build_and_snapshot
    from vforge.tasks.translate import train_request

    with initialize_config_module(config_module="vforge.conf", version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=rep_world",
                "latent_cache.mode=offline",
                "model.extras.conditioner.input=latent",
                f"data.train_manifest={prepared.manifest}",
                f"data.sources=[{{cache_dir:{prepared.root}}}]",
                "checkpoint.enabled=false",
            ],
        )
    built = build_and_snapshot(cfg, world_size=1, write_snapshot=False)
    translated = train_request(built.config)
    validate_train_request(translated, "fsdp2")
    assert translated.sources[0].cache_dir == str(prepared.root)


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("with_rgb", [False, True])
def test_native_cached_loader_resumes_exact_next_samples(prepared, workers, with_rgb):
    if with_rgb:
        extras = copy.deepcopy(prepared.request.extras)
        extras["conditioner"].update(input="rgb", temporal_patch=4, context_mask_ratio=0)
        prepared.request = replace(prepared.request, extras=extras)

    def build():
        return build_resumable_loader(
            reader(prepared),
            parallel_state=prepared.state,
            batch_size=2,
            num_workers=workers,
            drop_last=True,
            seed=31,
            shuffle=True,
            persistent_workers=workers > 0,
            prefetch_factor=2 if workers else None,
        )

    first = build()
    it = iter(first)
    next(it)
    saved, rng = first.state_dict(), torch.get_rng_state()
    expected = next(it)
    restored = build()
    restored.load_state_dict(saved)
    torch.set_rng_state(rng)
    actual = next(iter(restored))
    assert actual["sample_id"] == expected["sample_id"]
    assert torch.equal(actual["latents"], expected["latents"])
    if with_rgb:
        assert torch.equal(actual["rgb"], expected["rgb"])


@pytest.mark.parametrize("condition_input", ["latent", "rgb", "rgb_split"])
def test_online_and_cached_native_steps_match_loss_gradients_and_rng(
    prepared, monkeypatch, condition_input
):
    if condition_input != "latent":
        extras = copy.deepcopy(prepared.request.extras)
        extras["conditioner"].update(input="rgb", temporal_patch=4, context_mask_ratio=0)
        if condition_input == "rgb_split":
            extras["conditioner"].update(
                architecture="split_ec", depth=2, encoder_depth=1, feature_layers=[1]
            )
        prepared.request = replace(prepared.request, extras=extras)
    original = replace(prepared.request, latent_cache_mode="online", sources=())
    calls = []

    class Codec:
        def encode(self, rgb):
            assert not torch.is_grad_enabled()
            indices = ((rgb[:, 0, 0, 0, 0] + 1) * 127.5).round().long()
            calls.append(len(indices))
            return prepared.latent[indices].clone()

    monkeypatch.setattr(
        "vforge.tasks.rep_world.shared.CodecProvider.get", lambda self, device: Codec()
    )
    online = build_train_components(original)
    offline = build_train_components(prepared.request)
    torch.manual_seed(411)
    raw = next(iter(online.dataloader))
    raw_rng = torch.get_rng_state()
    torch.manual_seed(411)
    cached = next(iter(offline.dataloader))
    assert raw["sample_id"] == cached["sample_id"] and torch.equal(raw_rng, torch.get_rng_state())
    model = online.model.build()
    model.to_empty(device="cpu")
    online.model_initializer(model)
    cloned = copy.deepcopy(model)
    initialize_ops(
        OpsConfig(backend="sdpa_math", allow_fallback=False),
        device=torch.device("cpu"),
        parallel_state=None,
    )
    try:
        torch.manual_seed(101)
        expected = online.step_fn(model, raw, 0, StepContext(torch.device("cpu"), prepared.state))
        expected.loss.backward()
        rng = torch.get_rng_state()
        monkeypatch.setattr(
            "vforge.tasks.rep_world.shared.CodecProvider.get",
            lambda *args: pytest.fail("offline loaded codec"),
        )
        torch.manual_seed(101)
        actual = offline.step_fn(
            cloned, cached, 0, StepContext(torch.device("cpu"), prepared.state)
        )
        actual.loss.backward()
        assert torch.equal(actual.loss, expected.loss) and torch.equal(torch.get_rng_state(), rng)
        assert actual.metadata == expected.metadata and actual.workload == expected.workload
        for (name, left), (_, right) in zip(
            model.named_parameters(), cloned.named_parameters(), strict=True
        ):
            if left.grad is None:
                assert right.grad is None, name
            else:
                assert torch.equal(left.grad, right.grad), name
        assert calls == [2]
    finally:
        reset_ops()


@pytest.mark.parametrize("random_clip", [False, True])
def test_cached_rgb_pairs_exact_source_and_rng(prepared, random_clip):
    extras = copy.deepcopy(prepared.request.extras)
    extras["random_clip"] = random_clip
    extras["conditioner"].update(input="rgb", temporal_patch=4, context_mask_ratio=0)
    prepared.request = replace(prepared.request, extras=extras)
    validate_train_request(prepared.request, "fsdp2")
    cached = reader(prepared)
    original = VideoClipDataset(prepared.manifest, frames=32, random_clip=random_clip)
    assert cached.resume_contract["dataset"] == "rep_world.latent_shards_with_rgb.v1"
    for index in [0, 4, 7]:
        torch.manual_seed(19)
        expected = original[index]
        state = torch.get_rng_state()
        torch.manual_seed(19)
        actual = cached[index]
        assert torch.equal(torch.get_rng_state(), state)
        assert torch.equal(actual["rgb"], expected["rgb"])
        assert torch.equal(actual["latents"], prepared.latent[index])
        assert actual["sample_id"] == expected["sample_id"]


def test_cached_rgb_rejects_changed_source_length(prepared, monkeypatch):
    extras = copy.deepcopy(prepared.request.extras)
    extras["conditioner"].update(input="rgb", temporal_patch=4, context_mask_ratio=0)
    prepared.request = replace(prepared.request, extras=extras)
    cached = reader(prepared)

    class LongerReader(Reader):
        def __len__(self):
            return 33

    monkeypatch.setattr("vforge.tasks.rep_world.data.open_video", LongerReader)
    with pytest.raises(RuntimeError, match="source length changed"):
        cached[0]
