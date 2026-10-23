from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vforge.data.cache import EncodedBatchCodec, save_latent_bundle
from vforge.data.mixed_dataset import MixedDataset
from vforge.tasks.wan_taichu.cache import (
    open_taichu_encoded_cache_dataset,
    open_taichu_mixed_cache_dataset,
)

_CODEC = EncodedBatchCodec(
    name="wan-taichu-cache-test",
    required_keys=("value",),
    optional_keys=("ref_frame_latents",),
)


def _write(cache_dir: Path, index: int, value: float) -> None:
    path = cache_dir / "train" / f"{index // 1000:04d}" / f"{index:08d}.pt"
    save_latent_bundle({"value": torch.tensor(value)}, str(path))


def test_single_indexed_cache_restores_dense_taichu_index_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path, 0, 0.0)
    _write(tmp_path, 2, 2.0)
    monkeypatch.setattr("vforge.tasks.wan_taichu.cache.random.randrange", lambda _n: 2)

    dataset = open_taichu_encoded_cache_dataset(str(tmp_path), "", codec=_CODEC)

    assert dataset.sample_ids == ["00000000", "00000001", "00000002"]
    missing = dataset[1]
    assert missing["sample_id"] == "00000001"
    assert float(missing["value"]) == 2.0


def test_mixed_cache_fingerprint_pins_dense_and_written_membership(
    tmp_path: Path,
) -> None:
    _write(tmp_path, 0, 0.0)
    _write(tmp_path, 2, 2.0)

    mixed, fingerprint = open_taichu_mixed_cache_dataset(
        [
            {
                "name": "sparse",
                "cache_dir": str(tmp_path),
                "weight": 1.0,
                "min_latents": 1,
                "max_latents": 3,
            }
        ],
        codec=_CODEC,
    )

    assert isinstance(mixed, MixedDataset)
    assert mixed.sub_lengths == [3]
    source = fingerprint["sources"][0]
    assert source["sample_count"] == 3
    assert source["written_sample_count"] == 2
    assert source["indexed_miss_fallback"] is True


def test_ref_sidecar_is_attached_before_online_ref_fallback(tmp_path: Path) -> None:
    _write(tmp_path, 0, 0.0)
    sidecar = tmp_path / "ref_images" / "train" / "0000" / "00000000.pt"
    expected = torch.randn(48, 2, 3, 4)
    save_latent_bundle({"ref_latent": expected}, str(sidecar))

    dataset = open_taichu_encoded_cache_dataset(
        str(tmp_path),
        "",
        codec=_CODEC,
        load_ref_sidecar=True,
    )

    torch.testing.assert_close(dataset[0]["ref_frame_latents"], expected)
