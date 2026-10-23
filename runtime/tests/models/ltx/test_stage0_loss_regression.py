import json
import os
from pathlib import Path

import pytest

from tests.models.ltx.loss_regression import encode_loss_trace, load_loss_golden, verify_loss_trace

DEFAULT_MODEL_PATH = Path("/efs/dannsu/vforge/ci/ltx23-stage1/model/ltx-2.3-22b-dev.safetensors")
DEFAULT_DATA_ROOT = Path("/efs/dannsu/vforge/ci/ltx23-stage1/data/.precomputed")
GOLDEN_PATH = Path(__file__).with_name("stage0_loss_b300_dp2_usp2.json")


def test_stage0_and_stage1_fallback_reuse_staged_checkpoint(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file, save_file

    import vforge.models.ltx as ltx
    import vforge.utils.node_staging as staging
    from vforge.models.loading import load_pretrained_weights, stage_pretrained_weights
    from vforge.tasks.ltx.train import _pretrained_source

    remote = tmp_path / "remote"
    remote.mkdir()
    path = remote / "model.safetensors"
    expected = torch.nn.Linear(2, 2)
    save_file(expected.state_dict(), path)
    config = staging.NodeStagingConfig(root=str(tmp_path / "ssd"), minimum_free_bytes=0)
    monkeypatch.setattr(staging, "NodeStagingConfig", lambda: config)
    monkeypatch.setenv("VFORGE_NODE_IO_INLINE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    original_stage = staging.stage_on_node
    copies, loaded_paths = [], []

    def tracked_stage(*args, **kwargs):
        result = original_stage(*args, **kwargs)
        copies.append(result.copied)
        return result

    def loader(*, pretrained_path, dtype, connector_dtype):
        local = Path(pretrained_path)
        assert local.is_relative_to(tmp_path / "ssd")
        loaded_paths.append(local)
        model = torch.nn.Linear(2, 2, dtype=dtype)
        model.load_state_dict(load_file(str(local)))
        return model

    monkeypatch.setattr(staging, "stage_on_node", tracked_stage)
    monkeypatch.setattr(ltx, "load_pretrained_model", loader)
    # Both Stage0 and Stage1 use this source factory and loading mode.
    for _ in range(2):
        source = stage_pretrained_weights(_pretrained_source(str(path), "rank0_broadcast"))
        model = torch.nn.Linear(2, 2)
        load_pretrained_weights(model, source, strict=True)
        for name, value in model.state_dict().items():
            assert torch.equal(value, expected.state_dict()[name])
    assert copies == [True, False, False, False]
    assert loaded_paths == [Path(source.path)] * 2


@pytest.mark.gpu(gpus=4)
def test_ltx23_stage0_b300_dp2_usp2_loss_is_bitwise_stable(tmp_path, torchrun):
    assert os.environ.get("CI_MACHINE") == "b300", "LTX-2.3 golden is specific to B300"
    model_path = Path(os.environ.get("LTX23_CI_MODEL_PATH", DEFAULT_MODEL_PATH))
    data_root = Path(os.environ.get("LTX23_CI_PRECOMPUTED_ROOT", DEFAULT_DATA_ROOT))
    assert model_path.is_file(), f"missing LTX-2.3 CI checkpoint: {model_path}"
    assert data_root.is_dir(), f"missing LTX-2.3 CI dataset: {data_root}"

    worker = Path(__file__).with_name("loss_worker.py")
    golden = load_loss_golden(GOLDEN_PATH, steps=10)
    output_dir = tmp_path / "ltx23-stage0"
    torchrun(
        worker,
        "--experiment=ltx23_ti2av_sft",
        f"--model-path={model_path}",
        f"--data-root={data_root}",
        f"--output-dir={output_dir}",
        timeout=3600,
    )

    actual = encode_loss_trace(output_dir / "train_steps.jsonl", steps=10)
    print("LTX23_STAGE0_LOSS_TRACE=" + json.dumps(actual, sort_keys=True), flush=True)
    verify_loss_trace(actual, golden)
