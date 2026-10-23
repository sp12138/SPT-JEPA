from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vforge.tasks.wan_taichu.eval_task.gridnormal.artifacts import (
    GridNormalOutputWriter,
    gridnormal_sample_context,
)
from vforge.tasks.wan_taichu.eval_task.gridnormal.variants import (
    resolve_gridnormal_variants,
)


def test_writer_publishes_completion_and_variant_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "vforge.data.video_io.write_video_mp4",
        lambda _video, path, *, fps: Path(path).write_bytes(f"variant:{fps}".encode()),
    )
    selected = resolve_gridnormal_variants({})
    writer = GridNormalOutputWriter(fps=24.0, variants=selected)
    result = {
        "variants": {
            variant.name: {
                "video": torch.zeros(1, 3, 2, 2),
                "metadata": {
                    "gridnormal_condition": {
                        "grid": {"selection": variant.grid},
                        "normal": {"selection": variant.normal},
                        "ref": {"enabled": variant.ref},
                    }
                },
            }
            for variant in selected
        },
        "metadata": {"global_manifest_index": 4},
    }
    provenance = {
        "resolved_config_hash": "cfg",
        "checkpoint": {"id": "checkpoint"},
    }

    path = writer.write(
        result,
        str(tmp_path),
        sample_id="sample/a",
        seed=21,
        provenance=provenance,
    )

    assert Path(path) == tmp_path / ".gridnormal" / "sample_a.complete"
    assert Path(path).read_text(encoding="utf-8") == "complete\n"
    completion_sidecar = json.loads(Path(path + ".json").read_text(encoding="utf-8"))
    assert completion_sidecar["provenance"]["seed"] == 21
    for variant in selected:
        video = tmp_path / variant.name / "sample_a.mp4"
        sidecar = json.loads(Path(f"{video}.json").read_text(encoding="utf-8"))
        assert video.read_bytes() == b"variant:24.0"
        assert sidecar["metadata"]["variant"] == variant.name
        assert sidecar["metadata"]["gridnormal_condition"]["ref"]["enabled"] is variant.ref
        assert sidecar["provenance"]["checkpoint"] == {"id": "checkpoint"}
    assert writer.is_complete(
        str(tmp_path),
        sample_id="sample/a",
        seed=21,
        provenance=provenance,
    )


def test_shared_inputs_are_written_once_across_checkpoint_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vforge.data.video_io as video_io

    shared_video_writes: list[str] = []
    shared_image_writes: list[str] = []
    monkeypatch.setattr(
        video_io,
        "read_video_tensor",
        lambda _path, *, max_frames=None: torch.zeros(2, 3, 2, 2),
    )
    monkeypatch.setattr(
        video_io,
        "read_image_tensor",
        lambda _path: torch.zeros(3, 2, 2),
    )

    def write_frames(_frames: torch.Tensor, path: str, *, fps: float) -> None:
        shared_video_writes.append(path)
        Path(path).write_bytes(f"shared:{fps}".encode())

    def write_image(_image: torch.Tensor, path: str) -> None:
        shared_image_writes.append(path)
        Path(path).write_bytes(b"image")

    monkeypatch.setattr(video_io, "write_video_frames_mp4", write_frames)
    monkeypatch.setattr(video_io, "write_image_png", write_image)
    monkeypatch.setattr(
        video_io,
        "write_video_mp4",
        lambda _video, path, *, fps: Path(path).write_bytes(f"variant:{fps}".encode()),
    )

    selected = resolve_gridnormal_variants({})
    writer = GridNormalOutputWriter(
        fps=24.0,
        variants=selected,
        data_source="validation",
        manifest="/data/manifest.jsonl",
    )
    result = {
        "variants": {
            variant.name: {
                "video": torch.zeros(1, 3, 2, 2),
                "metadata": {},
            }
            for variant in selected
        },
        "shared_sources": {
            "gt": "/data/gt.mp4",
            "normal": "/data/normal.mp4",
            "normal_fg": "/data/normal_fg.mp4",
            "normal_bg": "/data/normal_bg.mp4",
            "grid": "/data/grid.mp4",
            "ref": ["/data/ref-0.png", "/data/ref-1.png"],
        },
        "metadata": {"global_manifest_index": 7},
    }

    for checkpoint in ("checkpoint-00001", "checkpoint-00002"):
        writer.write(
            result,
            str(tmp_path / checkpoint),
            sample_id="sample/a",
            seed=17,
            provenance={"checkpoint": {"id": checkpoint}},
        )

    assert len(shared_video_writes) == 5
    assert len(shared_image_writes) == 2
    marker = json.loads(
        (tmp_path / ".gridnormal-inputs" / "sample_a.json").read_text(encoding="utf-8")
    )
    assert marker["global_manifest_index"] == 7
    assert marker["unavailable"] == []
    assert marker["streams"]["ref"] == [
        "ref/sample_a-000.png",
        "ref/sample_a-001.png",
    ]
    assert marker["schema_version"] == 2


def test_shared_inputs_accept_native_gameplay_source_descriptors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vforge.data.gameplay_dataset as gameplay
    import vforge.data.video_io as video_io
    from vforge.data.gameplay_dataset import GAMEPLAY_MEDIA_SOURCES_KEY

    descriptor = {
        "kind": "gameplay_window",
        "stream": "video",
        "path": "/data/chunk.mp4",
        "start_frame": 3,
        "end_frame": 7,
        "height": 2,
        "width": 2,
    }
    context = gridnormal_sample_context({GAMEPLAY_MEDIA_SOURCES_KEY: {"gt": descriptor}})
    assert context["__wan_taichu_gridnormal_shared_sources__"]["gt"] == descriptor

    decoded = torch.zeros(5, 3, 2, 2)
    monkeypatch.setattr(gameplay, "decode_gameplay_media_source", lambda source: decoded)
    monkeypatch.setattr(
        video_io,
        "read_video_tensor",
        lambda *_args, **_kwargs: pytest.fail("descriptor must not use path-only decoding"),
    )
    monkeypatch.setattr(
        video_io,
        "write_video_frames_mp4",
        lambda frames, path, *, fps: Path(path).write_bytes(
            f"{tuple(frames.shape)}:{fps}".encode()
        ),
    )
    monkeypatch.setattr(
        video_io,
        "write_video_mp4",
        lambda _video, path, *, fps: Path(path).write_bytes(f"variant:{fps}".encode()),
    )
    selected = resolve_gridnormal_variants({"variant": "wan22_a14b"}, ["full"])
    writer = GridNormalOutputWriter(fps=24.0, variants=selected)
    writer.write(
        {
            "variants": {"full": {"video": torch.zeros(1, 3, 2, 2), "metadata": {}}},
            "shared_sources": {"gt": descriptor},
            "metadata": {"global_manifest_index": 0},
        },
        str(tmp_path),
        sample_id="native",
        seed=42,
        provenance={"checkpoint": {"id": "checkpoint"}},
    )

    assert (tmp_path / "gt/native.mp4").read_bytes() == b"(5, 3, 2, 2):24.0"


def test_online_reference_frames_are_exported_from_the_target_video(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vforge.data.video_io as video_io

    frames = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1)
    written_images: list[torch.Tensor] = []
    monkeypatch.setattr(
        video_io,
        "read_video_tensor",
        lambda _path, *, max_frames=None: frames,
    )
    monkeypatch.setattr(
        video_io,
        "read_image_tensor",
        lambda _path: pytest.fail("explicit ref must not replace online refs"),
    )

    def write_frames(_frames: torch.Tensor, path: str, *, fps: float) -> None:
        Path(path).write_bytes(f"shared:{fps}".encode())

    def write_image(image: torch.Tensor, path: str) -> None:
        written_images.append(image.clone())
        Path(path).write_bytes(b"image")

    monkeypatch.setattr(video_io, "write_video_frames_mp4", write_frames)
    monkeypatch.setattr(video_io, "write_image_png", write_image)
    monkeypatch.setattr(
        video_io,
        "write_video_mp4",
        lambda _video, path, *, fps: Path(path).write_bytes(f"variant:{fps}".encode()),
    )

    selected = resolve_gridnormal_variants({})
    writer = GridNormalOutputWriter(
        fps=24.0,
        variants=selected,
        data_source="validation",
        manifest="/data/manifest.jsonl",
    )
    result = {
        "variants": {
            variant.name: {
                "video": torch.zeros(1, 3, 2, 2),
                "metadata": {},
            }
            for variant in selected
        },
        "shared_sources": {
            "gt": "/data/target.mp4",
            "ref": "/data/explicit.png",
        },
        "metadata": {
            "global_manifest_index": 7,
            "ref_selection": {
                "source": "target_video",
                "indices": [1, 3],
                "count": 2,
            },
        },
    }

    writer.write(
        result,
        str(tmp_path / "checkpoint-00001"),
        sample_id="sample/a",
        seed=17,
        provenance={"checkpoint": {"id": "checkpoint-00001"}},
    )

    assert [image.item() for image in written_images] == [1.0, 3.0]
    marker = json.loads(
        (tmp_path / ".gridnormal-inputs" / "sample_a.json").read_text(encoding="utf-8")
    )
    assert marker["streams"]["ref"] == [
        "ref/sample_a-000.png",
        "ref/sample_a-001.png",
    ]
    assert marker["ref_selection"]["indices"] == [1, 3]
