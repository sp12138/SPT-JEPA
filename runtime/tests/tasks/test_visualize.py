from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from vforge.tasks import visualize


def _stub_media_io(monkeypatch, calls: list[tuple]) -> None:
    def read_video(path):
        calls.append(("read_video", path))
        return f"frames:{path}"

    def write_video(frames, path, *, fps):
        calls.append(("write_video", frames, fps))
        Path(path).write_bytes(b"mp4")

    def read_image(path):
        calls.append(("read_image", path))
        return f"image:{path}"

    def write_image(image, path):
        calls.append(("write_image", image))
        Path(path).write_bytes(b"png")

    monkeypatch.setattr(visualize, "read_video_tensor", read_video)
    monkeypatch.setattr(visualize, "write_video_frames_mp4", write_video)
    monkeypatch.setattr(visualize, "read_image_tensor", read_image)
    monkeypatch.setattr(visualize, "write_image_png", write_image)


def test_visualize_json_exports_recognized_media_and_text(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "rgb": "rgb.mp4",
                    "normal": "normal.mp4",
                    "grid": "grid.mp4",
                    "normal_fg": "normal-fg.mp4",
                    "normal_bg": "normal-bg.mp4",
                    "ref": "reference.jpg",
                    "image": "condition.webp",
                    "text": "walk forward",
                }
            ]
        ),
        encoding="utf-8",
    )
    calls: list[tuple] = []
    _stub_media_io(monkeypatch, calls)

    output_dir = tmp_path / "artifacts"
    exported = visualize.visualize_manifest(
        manifest,
        num_samples=1,
        fps=12.0,
        output_dir=output_dir,
    )

    assert {path.name for path in exported} == {
        "sample_0000_video.mp4",
        "sample_0000_normal.mp4",
        "sample_0000_grid.mp4",
        "sample_0000_normal_fg.mp4",
        "sample_0000_normal_bg.mp4",
        "sample_0000_ref.png",
        "sample_0000_image.png",
        "sample_0000_prompt.txt",
        "sample_0000_metadata.json",
    }
    assert all(path.is_file() for path in exported)
    assert (output_dir / "sample_0000_prompt.txt").read_text(encoding="utf-8") == ("walk forward")
    metadata = json.loads((output_dir / "sample_0000_metadata.json").read_text(encoding="utf-8"))
    assert metadata["sample_index"] == 0
    assert metadata["present"] == {
        "grid": True,
        "image": True,
        "normal": True,
        "normal_bg": True,
        "normal_fg": True,
        "prompt": True,
        "ref": True,
        "video": True,
    }
    assert ("read_video", str(tmp_path / "rgb.mp4")) in calls
    assert ("read_image", str(tmp_path / "reference.jpg")) in calls
    assert sum(call[0] == "write_video" and call[2] == 12.0 for call in calls) == 5


def test_visualize_jsonl_sample_index_exports_only_requested_record(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            (
                json.dumps({"video": "first.mp4", "prompt": "first"}),
                json.dumps(
                    {
                        "video": "second.mp4",
                        "reference": "reference.mp4",
                        "prompt": "second",
                    }
                ),
            )
        ),
        encoding="utf-8",
    )
    calls: list[tuple] = []
    _stub_media_io(monkeypatch, calls)

    output_dir = tmp_path / "selected"
    exported = visualize.visualize_manifest(
        manifest,
        sample_index=1,
        num_samples=None,
        output_dir=output_dir,
    )

    assert {path.name for path in exported} == {
        "sample_0001_video.mp4",
        "sample_0001_ref.mp4",
        "sample_0001_prompt.txt",
        "sample_0001_metadata.json",
    }
    assert not (output_dir / "sample_0000_prompt.txt").exists()
    assert (output_dir / "sample_0001_prompt.txt").read_text(encoding="utf-8") == "second"
    assert not any("first.mp4" in str(part) for call in calls for part in call)


def test_visualize_gameplay_index_exports_decoded_tensors_without_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[tuple] = []
    _stub_media_io(monkeypatch, calls)
    opened: list[dict] = []

    class FakeGameplayDataset:
        def __init__(self, **kwargs):
            opened.append(kwargs)
            self.samples = [
                {
                    "sample_id": "native",
                    "video_frames": torch.zeros(5, 3, 4, 6),
                    "normal_condition": torch.full((5, 3, 4, 6), 0.5),
                    "image_condition": torch.zeros(1, 3, 4, 6),
                    "prompt": "native pkl",
                }
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    monkeypatch.setattr(visualize, "GameplayDataset", FakeGameplayDataset)
    output_dir = tmp_path / "native"
    exported = visualize.visualize_gameplay_index(
        ["/data/items.pkl"],
        frames=5,
        height=4,
        width=6,
        path_maps=(("/efs", "/gwm-efs"),),
        num_samples=1,
        fps=24.0,
        output_dir=output_dir,
    )

    assert opened[0]["index_files"] == ["/data/items.pkl"]
    assert opened[0]["extras"]["path_replacements"] == [("/efs", "/gwm-efs")]
    assert {path.name for path in exported} == {
        "sample_0000_video.mp4",
        "sample_0000_normal.mp4",
        "sample_0000_image.png",
        "sample_0000_prompt.txt",
        "sample_0000_metadata.json",
    }
    assert sum(call[0] == "write_video" and call[2] == 24.0 for call in calls) == 2


def test_visualize_evaluation_preset_hides_pkl_geometry_and_path_maps(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[tuple] = []
    _stub_media_io(monkeypatch, calls)
    opened: list[dict] = []

    class FakeGameplayDataset:
        def __init__(self, **kwargs):
            opened.append(kwargs)
            self.samples = [
                {
                    "video_frames": torch.zeros(5, 3, 4, 6),
                    "prompt": "preset",
                }
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    sources = OmegaConf.create(
        [
            {
                "name": "gameplay-20s",
                "kind": "gameplay",
                "index_files": ["/efs/shared/items.pkl"],
                "frames": 5,
                "height": 4,
                "width": 6,
                "extras": {
                    "use_partial_normal_aug": True,
                },
            }
        ]
    )
    monkeypatch.setattr(visualize, "GameplayDataset", FakeGameplayDataset)
    monkeypatch.setattr(
        "vforge.tasks.common.compose_standalone_config",
        lambda _name, _overrides: SimpleNamespace(
            evaluation=SimpleNamespace(
                sources=sources,
                path_replacements=OmegaConf.create([]),
            )
        ),
    )

    exported = visualize.visualize_evaluation_preset(
        "wan_taichu_a14b_20s_prod",
        num_samples=1,
        output_dir=tmp_path / "preset",
    )

    assert opened[0]["index_files"] == ["/efs/shared/items.pkl"]
    assert opened[0]["frames"] == 5
    assert "path_replacements" not in opened[0]["extras"]
    assert {path.name for path in exported} == {
        "sample_0000_video.mp4",
        "sample_0000_prompt.txt",
        "sample_0000_metadata.json",
    }
