from __future__ import annotations

import json

import pytest

from vforge.data.dataset import (
    InferenceRequestDataset,
    RawSampleDataset,
    load_manifest_sample_ids,
    load_sample_manifest,
)


def test_load_sample_manifest_accepts_taichu_nested_media_schema(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "id": "chunk_0000000_0000000",
                    "file": {
                        "video": {
                            "video_path": "/asset/rgb.mp4",
                            "normal_path": "/asset/normal.mp4",
                        },
                        "text_caption": [{"caption": "a test prompt"}],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    [sample] = load_sample_manifest(manifest)

    assert sample.sample_id == "chunk_0000000_0000000"
    assert sample.video == "/asset/rgb.mp4"
    assert sample.normal == "/asset/normal.mp4"
    assert sample.prompt == "a test prompt"


def test_flat_manifest_fields_take_precedence_over_nested_schema(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "id": "sample",
                    "video": "/flat/rgb.mp4",
                    "normal": "/flat/normal.mp4",
                    "prompt": "flat prompt",
                    "file": {
                        "video": {
                            "video_path": "/nested/rgb.mp4",
                            "normal_path": "/nested/normal.mp4",
                        },
                        "text_caption": [{"caption": "nested prompt"}],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    [sample] = load_sample_manifest(manifest)

    assert sample.video == "/flat/rgb.mp4"
    assert sample.normal == "/flat/normal.mp4"
    assert sample.prompt == "flat prompt"


def test_taichu_uuid_and_id_form_joint_sample_identity(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    records = [
        {
            "uuid": uuid,
            "id": "chunk_0000000_0000000",
            "file": {"video": {"video_path": f"/asset/{uuid}.mp4"}},
        }
        for uuid in ("video-a", "video-b")
    ]
    manifest.write_text(json.dumps(records), encoding="utf-8")

    samples = load_sample_manifest(manifest)

    assert [sample.sample_id for sample in samples] == [
        "video-a:chunk_0000000_0000000",
        "video-b:chunk_0000000_0000000",
    ]
    assert load_manifest_sample_ids(manifest) == [
        "video-a:chunk_0000000_0000000",
        "video-b:chunk_0000000_0000000",
    ]


def test_inference_request_dataset_preserves_model_fields(tmp_path) -> None:
    path = tmp_path / "infer.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "id": "request-1",
                        "image": "/images/first.png",
                        "normal": "/conditions/normal.mp4",
                        "prompt": "turn around",
                    }
                ),
                json.dumps({"image": "/images/second.png"}),
            )
        ),
        encoding="utf-8",
    )

    dataset = InferenceRequestDataset.from_manifest(path)

    assert dataset[0] == {
        "id": "request-1",
        "sample_id": "request-1",
        "image": "/images/first.png",
        "normal": "/conditions/normal.mp4",
        "prompt": "turn around",
    }
    assert dataset[1]["sample_id"] == "sample-000001"
    item = dataset[0]
    item["image"] = "mutated"
    assert dataset[0]["image"] == "/images/first.png"


def test_inference_request_dataset_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate sample_id"):
        InferenceRequestDataset([{"id": "same"}, {"sample_id": "same"}])


def test_inference_request_dataset_resolves_relative_media_paths(tmp_path) -> None:
    path = tmp_path / "infer.jsonl"
    path.write_text(
        json.dumps(
            {
                "video": "video/target.mp4",
                "image": "image/first.png",
                "normal": "normal/condition.mp4",
            }
        ),
        encoding="utf-8",
    )

    sample = InferenceRequestDataset.from_manifest(path)[0]

    assert sample["video"] == str(tmp_path / "video/target.mp4")
    assert sample["image"] == str(tmp_path / "image/first.png")
    assert sample["normal"] == str(tmp_path / "normal/condition.mp4")


def test_raw_sample_dataset_still_requires_target_video(tmp_path) -> None:
    path = tmp_path / "validation.jsonl"
    path.write_text(json.dumps({"image": "/images/first.png"}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing a video path"):
        RawSampleDataset.from_manifest(path)
