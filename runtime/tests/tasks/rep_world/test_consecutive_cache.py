import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vforge.models.rep_world.config import codec_geometry
from vforge.tasks.rep_world.data import VideoClipDataset
from vforge.tasks.rep_world.latent_cache import (
    CONSECUTIVE_SCHEMA,
    CachedConsecutiveVideoLatents,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def split_cache(tmp_path):
    root, data = tmp_path / "cache", tmp_path / "data"
    data.mkdir()
    checkpoint, statistics = tmp_path / "codec.pth", tmp_path / "stats.pt"
    checkpoint.write_bytes(b"codec")
    statistics.write_bytes(b"stats")
    config = SimpleNamespace(
        geometry=codec_geometry("videorae", "3d"),
        states=10,
        frame_interval=1,
        random_clip=False,
        conditioner=SimpleNamespace(input="latent"),
        codec={"checkpoint": str(checkpoint), "statistics": str(statistics)},
    )
    owners = {}
    train, test = [], []
    for owner_index, owner in enumerate(("alpha", "beta")):
        directory = root / owner
        (directory / "shards").mkdir(parents=True)
        values = np.full((2, 256, 10, 16, 16), owner_index + 1, dtype=np.float32)
        values[1] += 1
        array_path = directory / "shards/000000.npy"
        np.save(array_path, values)
        contract = {
            "schema": CONSECUTIVE_SCHEMA,
            "dataset": owner,
            "samples": 2,
            "shard_size": 2,
            "latent_shape": [256, 10, 16, 16],
            "latent_dtype": "float32",
            "channel_index": "plane * 64 + channel",
            "window_frames": 160,
            "frame_interval": 1,
            "rgb_shape": [3, 160, 256, 256],
            "precision": "bf16 codec",
            "assets": {
                key: {"sha256": digest(path)}
                for key, path in (("checkpoint", checkpoint), ("statistics", statistics))
            },
        }
        fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        contract["fingerprint"] = fingerprint
        write_json(directory / "contract.json", contract)
        index, metadata = [], []
        for row in range(2):
            sample_id = f"{owner}:{row}"
            checksum = hashlib.sha256(memoryview(values[row]).cast("B")).hexdigest()
            path = str(tmp_path / f"{owner}.mp4")
            start = row * 160
            record = dict(
                dataset=owner,
                sample_id=sample_id,
                rgb_path=path,
                start_frame=start,
                end_frame_exclusive=start + 160,
                cache_index=row,
                latent_sha256=checksum,
            )
            (test if owner == "alpha" and row == 1 else train).append(record)
            index.append(
                dict(
                    index=row,
                    sample_id=sample_id,
                    rgb_path=path,
                    start_frame=start,
                    end_frame_exclusive=start + 160,
                    latent_path="shards/000000.npy",
                    row=row,
                    latent_sha256=checksum,
                )
            )
            metadata.append(
                dict(
                    index=row,
                    row=row,
                    sample_id=sample_id,
                    rgb_path=path,
                    frame_start=start,
                    frame_end_exclusive=start + 160,
                    latent_sha256=checksum,
                )
            )
        index_path = directory / "index.jsonl"
        index_path.write_text("".join(json.dumps(row) + "\n" for row in index))
        descriptor = dict(
            shard=0,
            count=2,
            fingerprint=fingerprint,
            sha256=digest(array_path),
            verified_readback=True,
        )
        write_json(array_path.with_suffix(".json"), {**descriptor, "records": metadata})
        write_json(
            directory / "COMPLETE.json",
            dict(
                status="complete",
                samples=2,
                fingerprint=fingerprint,
                index_sha256=digest(index_path),
                shards=[descriptor],
            ),
        )
        owners[owner] = dict(fingerprint=fingerprint, index_sha256=digest(index_path), samples=2)
    manifests = {}
    for name, records in (("train.jsonl", train), ("test.jsonl", test)):
        path = data / name
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        manifests[name] = dict(sha256=digest(path), samples=len(records))
    write_json(
        data / "split_contract.json",
        dict(
            schema="rep_world.consecutive160.clip_split.v1",
            cache_root=str(root),
            owners=owners,
            manifests=manifests,
        ),
    )
    return SimpleNamespace(root=root, data=data, config=config, train=train, test=test)


def test_multi_owner_cache_reads_only_selected_windows(split_cache):
    fixture = split_cache
    reader = CachedConsecutiveVideoLatents(
        fixture.root, fixture.data / "train.jsonl", config=fixture.config, precision="bf16"
    )
    assert len(reader) == 3
    assert [reader[i]["sample_id"] for i in range(3)] == ["alpha:0", "beta:0", "beta:1"]
    assert [float(reader[i]["latents"].mean()) for i in range(3)] == [1.0, 2.0, 3.0]
    reader[0]["latents"].zero_()
    assert float(reader[0]["latents"].mean()) == 1.0
    heldout = CachedConsecutiveVideoLatents(
        fixture.root, fixture.data / "test.jsonl", config=fixture.config, precision="bf16"
    )
    assert heldout[0]["sample_id"] == "alpha:1"


def test_nine_state_view_reads_original_ten_state_cache(split_cache):
    config = SimpleNamespace(**vars(split_cache.config))
    config.states = 9
    config.cached_state_view = "random_9_from_10"
    reader = CachedConsecutiveVideoLatents(
        split_cache.root, split_cache.data / "train.jsonl", config=config, precision="bf16"
    )
    assert reader[0]["latents"].shape == (256, 10, 16, 16)
    assert reader.resume_contract["model_states"] == 9
    assert reader.resume_contract["cached_state_view"] == "random_9_from_10"


def test_multi_owner_cache_rejects_changed_window(split_cache):
    fixture = split_cache
    reader = CachedConsecutiveVideoLatents(
        fixture.root, fixture.data / "train.jsonl", config=fixture.config, precision="bf16"
    )
    metadata_path = fixture.root / "beta/shards/000000.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["records"][1]["frame_start"] = 0
    write_json(metadata_path, metadata)
    with pytest.raises(RuntimeError, match="beta:1"):
        reader[2]


def test_eval_decodes_declared_window_start(split_cache, monkeypatch):
    class FakeReader:
        def __len__(self):
            return 320

        def get_batch(self, indices):
            frames = np.empty((len(indices), 256, 256, 3), dtype=np.uint8)
            for i, frame in enumerate(indices):
                frames[i].fill(frame % 256)
            return SimpleNamespace(asnumpy=lambda: frames)

    monkeypatch.setattr("vforge.tasks.rep_world.data.open_video", lambda path: FakeReader())
    dataset = VideoClipDataset(split_cache.data / "test.jsonl", frames=160, random_clip=False)
    rgb = dataset[0]["rgb"]
    assert rgb.shape == (3, 160, 256, 256)
    assert float(rgb[0, 0, 0, 0]) == pytest.approx(160 / 255 * 2 - 1)
    assert float(rgb[0, -1, 0, 0]) == pytest.approx(319 % 256 / 255 * 2 - 1)
