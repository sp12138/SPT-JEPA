"""CPU tests for the indexed cache layout Wan's offline caches use."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_indexed_shards(cache_dir: Path, indices: list[int]) -> None:
    for index in indices:
        path = cache_dir / "train" / f"{index // 1000:04d}" / f"{index:08d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def test_indexed_cache_uses_manifest_order_without_flat_media_fields(
    tmp_path: Path,
) -> None:
    from vforge.data.cache import save_latent_bundle
    from vforge.data.dataset import (
        EncodedCacheDataset,
        load_manifest_sample_ids,
        manifest_file_digest,
    )
    from vforge.data.taichu_index_cache import indexed_path_resolver
    from vforge.tasks.wan.v21.shared import (
        WAN21_ENCODED_BATCH_CODEC,
        project_to_encoded_batch,
    )

    manifest_path = tmp_path / "train.json"
    manifest_path.write_text(
        json.dumps(
            [
                {"id": "chunked-a", "chunks": {"0": {"video_path": "a.mp4"}}},
                {"id": "chunked-b", "chunks": {"0": {"video_path": "b.mp4"}}},
            ]
        )
    )
    sample_ids = load_manifest_sample_ids(manifest_path)
    bundle = project_to_encoded_batch(
        {
            "prompt": torch.ones(2, 4),
            "frame_latents": torch.ones(3, 2, 2, 2),
            "image_condition_latent_vae": None,
            "image_condition_latent_clip": None,
            "image_condition_latent_dino": None,
            "condition_latents_normal": torch.ones(5, 2, 2, 2),
        }
    )
    cache_path = tmp_path / "cache" / "train" / "0000" / "00000001.pt"
    cache_path.parent.mkdir(parents=True)
    save_latent_bundle(bundle, str(cache_path))

    dataset = EncodedCacheDataset(
        str(tmp_path / "cache"),
        sample_ids,
        WAN21_ENCODED_BATCH_CODEC,
        path_for=indexed_path_resolver(),
    )
    loaded = dataset[1]
    assert loaded["sample_id"] == "chunked-b"
    assert torch.equal(loaded["frame_latents"], bundle["frame_latents"])
    assert (
        manifest_file_digest(manifest_path)
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )


def test_scan_indexed_cache_derives_ordered_ids_from_dense_shards(tmp_path: Path) -> None:
    from vforge.data.taichu_index_cache import scan_indexed_cache

    # Crosses a shard boundary, so a scan that only reads train/0000 fails here.
    _write_indexed_shards(tmp_path, list(range(1005)))
    (tmp_path / "train" / "0000" / "notes.txt").write_text("ignored")
    (tmp_path / "train" / "0000" / "0001.pt").write_bytes(b"")

    sample_ids = scan_indexed_cache(tmp_path)
    assert len(sample_ids) == 1005
    assert sample_ids[0] == "00000000"
    assert sample_ids[-1] == "00001004"


def test_scan_result_is_a_function_of_the_count(tmp_path: Path) -> None:
    # What the leader broadcasts is the count alone, so a follower's rebuilt
    # list is only the scan's list if these two agree for every count.
    from vforge.data.taichu_index_cache import indexed_sample_ids, scan_indexed_cache

    _write_indexed_shards(tmp_path, list(range(1005)))

    assert scan_indexed_cache(tmp_path) == indexed_sample_ids(1005)
    assert indexed_sample_ids(0) == []
    with pytest.raises(ValueError, match="non-negative"):
        indexed_sample_ids(-1)


def test_scan_indexed_cache_rejects_holes(tmp_path: Path) -> None:
    from vforge.data.taichu_index_cache import scan_indexed_cache

    _write_indexed_shards(tmp_path, [index for index in range(10) if index not in (3, 7)])

    with pytest.raises(ValueError, match="missing 2 of 10 bundles") as excinfo:
        scan_indexed_cache(tmp_path)
    assert "00000003" in str(excinfo.value)
    assert "00000007" in str(excinfo.value)


def test_scan_indexed_cache_rejects_empty_and_absent_split(tmp_path: Path) -> None:
    from vforge.data.taichu_index_cache import scan_indexed_cache

    with pytest.raises(ValueError, match="no train/ directory"):
        scan_indexed_cache(tmp_path)

    (tmp_path / "train" / "0000").mkdir(parents=True)
    with pytest.raises(ValueError, match="holds no 8-digit"):
        scan_indexed_cache(tmp_path)


def test_open_encoded_cache_dataset_scans_when_manifest_absent(tmp_path: Path) -> None:
    from vforge.data.cache import save_latent_bundle
    from vforge.data.taichu_index_cache import open_encoded_cache_dataset
    from vforge.tasks.wan.v21.shared import (
        WAN21_ENCODED_BATCH_CODEC,
        project_to_encoded_batch,
    )

    bundle = project_to_encoded_batch(
        {
            "prompt": torch.ones(2, 4),
            "frame_latents": torch.ones(3, 2, 2, 2),
            "image_condition_latent_vae": None,
            "image_condition_latent_clip": None,
            "image_condition_latent_dino": None,
            "condition_latents_normal": torch.ones(5, 2, 2, 2),
        }
    )
    for index in range(2):
        path = tmp_path / "train" / "0000" / f"{index:08d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_latent_bundle(bundle, str(path))

    dataset = open_encoded_cache_dataset(str(tmp_path), "", codec=WAN21_ENCODED_BATCH_CODEC)
    assert dataset.sample_ids == ["00000000", "00000001"]
    assert torch.equal(dataset[1]["frame_latents"], bundle["frame_latents"])
