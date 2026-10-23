"""Cache pairing must support both SFT/TF and video-only DMD sources."""

import pytest
import torch

from vforge.tasks.ltx.data import LtxPrecomputedDataset


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("dmd", [False, True])
def test_nested_legacy_cache_pairs_selected_sources(tmp_path, sharded, dmd):
    root = tmp_path / ".precomputed"
    shards = (root / "0002", root / "0001") if sharded else (root,)
    sources = ("latents", "conditions") if dmd else ("latents", "conditions", "audio_latents")
    latent = torch.arange(24).reshape(6, 4)
    for shard in shards:
        offset = int(shard.name) if sharded else 0
        for source in sources:
            name = "condition_clip.pt" if source == "conditions" else "latent_clip.pt"
            path = shard / source / "nested" / name
            path.parent.mkdir(parents=True)
            value = (
                {"latents": latent + offset, "num_frames": 1, "height": 2, "width": 3}
                if source == "latents"
                else {"value": torch.tensor([offset])}
            )
            torch.save(value, path)
    dataset = (
        LtxPrecomputedDataset(tmp_path, sources=sources) if dmd else LtxPrecomputedDataset(tmp_path)
    )
    assert len(dataset) == len(shards)
    expected_ids = (
        ("0001/nested/latent_clip", "0002/nested/latent_clip")
        if sharded
        else ("nested/latent_clip",)
    )
    assert dataset.sample_ids == expected_ids
    assert LtxPrecomputedDataset(root, sources=sources).sample_ids == expected_ids
    for index, sample_id in enumerate(expected_ids):
        sample = dataset[index]
        offset = index + 1 if sharded else 0
        assert sample["sample_id"] == sample_id
        torch.testing.assert_close(
            sample["video_latents"]["latents"],
            (latent + offset).reshape(1, 2, 3, 4).permute(3, 0, 1, 2),
        )
        assert sample["conditions"]["value"].item() == offset
        assert ("audio_latents" in sample) is not dmd
    if dmd:
        with pytest.raises(FileNotFoundError, match="required LTX source directories"):
            LtxPrecomputedDataset(tmp_path)
