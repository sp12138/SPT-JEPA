import json
import os
from pathlib import Path

import pytest

from tests.models.ltx.loss_regression import (
    encode_loss_trace,
    load_loss_golden,
    verify_loss_trace,
)

DEFAULT_MODEL_PATH = Path("/efs/dannsu/vforge/ci/ltx23-stage1/model/ltx-2.3-22b-dev.safetensors")
DEFAULT_DATA_ROOT = Path("/efs/dannsu/vforge/ci/ltx23-stage1/data/.precomputed")
GOLDEN_PATH = Path(__file__).with_name("stage1_loss_b300_dp2_usp2.json")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _row(step, *, loss, video_loss, audio_loss):
    return {
        "event": "step",
        "step": step,
        "loss_global": loss,
        "metrics": {
            "video_loss": video_loss,
            "audio_loss": audio_loss,
        },
    }


def test_encode_loss_trace_preserves_exact_float64_values(tmp_path):
    trace = tmp_path / "train_steps.jsonl"
    _write_jsonl(
        trace,
        [
            _row(0, loss=1.0, video_loss=0.25, audio_loss=0.75),
            _row(1, loss=-0.0, video_loss=0.1, audio_loss=0.2),
        ],
    )

    assert encode_loss_trace(trace, steps=2) == [
        {
            "step": 0,
            "loss": "0x1.0000000000000p+0",
            "video_loss": "0x1.0000000000000p-2",
            "audio_loss": "0x1.8000000000000p-1",
        },
        {
            "step": 1,
            "loss": "-0x0.0p+0",
            "video_loss": (0.1).hex(),
            "audio_loss": (0.2).hex(),
        },
    ]


def test_encode_loss_trace_requires_ten_consecutive_steps(tmp_path):
    trace = tmp_path / "train_steps.jsonl"
    _write_jsonl(
        trace,
        [_row(step, loss=1.0, video_loss=0.25, audio_loss=0.75) for step in range(9)],
    )

    with pytest.raises(AssertionError, match="expected steps 0..9"):
        encode_loss_trace(trace, steps=10)


def test_verify_loss_trace_reports_the_first_bitwise_mismatch():
    expected = [
        {
            "step": 0,
            "loss": "0x1.0000000000000p+0",
            "video_loss": "0x1.0000000000000p-2",
            "audio_loss": "0x1.8000000000000p-1",
        }
    ]
    actual = [{**expected[0], "video_loss": "0x1.0000000000001p-2"}]

    with pytest.raises(AssertionError, match=r"step 0 video_loss bitwise mismatch"):
        verify_loss_trace(actual, expected)


def test_load_loss_golden_reads_a_valid_contract(tmp_path):
    expected = [
        {
            "step": 0,
            "loss": "0x1.0000000000000p+0",
            "video_loss": "0x1.0000000000000p-2",
            "audio_loss": "0x1.8000000000000p-1",
        }
    ]
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "machine": "b300",
                "topology": "DP2xUSP2",
                "steps": 1,
                "losses": expected,
            }
        )
    )

    assert load_loss_golden(path, steps=1) == expected


@pytest.mark.gpu(gpus=4)
def test_ltx23_stage1_b300_dp2_usp2_loss_is_bitwise_stable(tmp_path, torchrun):
    assert os.environ.get("CI_MACHINE") == "b300", "LTX-2.3 golden is specific to B300"
    model_path = Path(os.environ.get("LTX23_CI_MODEL_PATH", DEFAULT_MODEL_PATH))
    data_root = Path(os.environ.get("LTX23_CI_PRECOMPUTED_ROOT", DEFAULT_DATA_ROOT))
    assert model_path.is_file(), f"missing LTX-2.3 CI checkpoint: {model_path}"
    assert data_root.is_dir(), f"missing LTX-2.3 CI dataset: {data_root}"

    worker = Path(__file__).with_name("loss_worker.py")
    golden = load_loss_golden(GOLDEN_PATH, steps=10)
    output_dir = tmp_path / "ltx23-stage1"
    torchrun(
        worker,
        "--experiment=ltx23_causal_ar_tf",
        f"--model-path={model_path}",
        f"--data-root={data_root}",
        f"--output-dir={output_dir}",
        timeout=3600,
    )

    actual = encode_loss_trace(output_dir / "train_steps.jsonl", steps=10)
    print("LTX23_STAGE1_LOSS_TRACE=" + json.dumps(actual, sort_keys=True), flush=True)
    verify_loss_trace(actual, golden)
