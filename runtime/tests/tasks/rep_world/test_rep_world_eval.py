import json
from dataclasses import replace

import pytest
import torch

from vforge.tasks.rep_world.eval import build_eval_components, collate_sample
from vforge.tasks.requests import EvalRequest


def test_eval_builder_uses_checkpoint_and_only_reads_observed_prefix(tmp_path, monkeypatch):
    manifest = tmp_path / "clips.jsonl"
    manifest.write_text(json.dumps(dict(video="clip.mp4", sample_id="case")))
    opened = []

    class Dataset:
        def __init__(self, path, *, frames, **kwargs):
            opened.append((path, frames))

        def __len__(self):
            return 1

    monkeypatch.setattr("vforge.tasks.rep_world.eval.VideoClipDataset", Dataset)
    monkeypatch.setattr("vforge.tasks.rep_world.eval.get_parallel_state", lambda: None)
    monkeypatch.setattr("vforge.tasks.rep_world.eval.build_eval_loader", lambda *args, **kwargs: [])
    request = EvalRequest(
        extras=dict(codec=dict(family="vrae", variant="eupe"), states=20),
        checkpoint_path="/not-loaded/dcp",
        data_source="inference",
        manifest=str(manifest),
        latent_cache_mode="online",
    )
    components = build_eval_components(request)
    assert components.initial_weights.path == "/not-loaded/dcp"
    assert opened[-1][1] == 4
    build_eval_components(replace(request, data_source="validation"))
    assert opened[-1][1] == 80
    assert all(p.is_meta for p in components.model.build().parameters())
    with pytest.raises(ValueError, match="requires evaluation.checkpoint_path"):
        build_eval_components(replace(request, checkpoint_path=""))


def test_eval_writer_declared_path_matches_atomic_video_output(tmp_path, monkeypatch):
    from vforge.evaluation.runner import EvalOutputWriter, sample_artifact_name
    from vforge.tasks.rep_world.eval import write_prediction

    calls = []

    def write(video, path, *, fps):
        from pathlib import Path

        calls.append((path, fps, video.shape))
        Path(path).write_bytes(b"video-content")

    monkeypatch.setattr("vforge.tasks.rep_world.eval.write_video_mp4", write)
    writer = EvalOutputWriter(
        artifact_name=lambda sample_id: sample_artifact_name(sample_id, suffix=".mp4"),
        write_output=lambda result, output_dir: write_prediction(result, output_dir, fps=16),
    )
    prediction = torch.zeros(1, 3, 8, 16, 16)
    result = dict(sample_id="../unsafe/id", prediction=prediction, metadata={"cfg_scale": 2})
    path = writer.write(result, str(tmp_path), sample_id=result["sample_id"], seed=4, provenance={})
    assert path == writer.expected_path(str(tmp_path), result["sample_id"])
    assert writer.is_complete(str(tmp_path), sample_id=result["sample_id"], seed=4, provenance={})
    assert calls[0][0].endswith(".tmp.mp4")


def test_eval_collation_keeps_padding_and_batches_one_clip():
    sample = dict(sample_id="case", rgb=torch.zeros(3, 4, 16, 16), __vforge_eval_padding__=True)
    batch = collate_sample([sample])
    assert batch["rgb"].shape == (1, 3, 4, 16, 16)
    assert batch["sample_id"] == "case" and batch["__vforge_eval_padding__"]
    with pytest.raises(ValueError):
        collate_sample([sample, sample])
