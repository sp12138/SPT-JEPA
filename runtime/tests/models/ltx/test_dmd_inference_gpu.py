"""B300 native DMD generation, USP and audio/video decode regression."""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.gpu(gpus=8)
@pytest.mark.parametrize("usp_size", [1, 2], ids=["dp8_sp1", "dp4_sp2"])
def test_dmd_eval_end_to_end(torchrun, tmp_path, dmd_assets, usp_size):
    assets = dmd_assets("base", "generator", "cache")

    prompt_args = []
    if usp_size == 2:
        assets.update(dmd_assets("gemma"))
        prompts = tmp_path / "prompts.txt"
        prompts.write_text("A person turns toward the camera.\nA person waves hello.\n")
        prompt_args = [
            f"model.extras.text_encoder_path={assets['gemma']}",
            f"model.extras.prompt_file={prompts}",
        ]
    run_dmd_eval(
        torchrun,
        tmp_path / "eval",
        assets,
        usp_size=usp_size,
        prompt_args=prompt_args,
        expected_samples=2 if usp_size == 2 else 1,
    )


def run_dmd_eval(
    torchrun, output, assets, *, usp_size=1, checkpoint=None, prompt_args=(), expected_samples=1
):
    # Mandatory dependencies fail before allocating the generator.
    import ltx_core.model.audio_vae  # noqa: F401

    base, cache = assets["base"], assets["cache"]
    generator = checkpoint or assets["generator"]
    check_reference = checkpoint is None and usp_size == 1 and not prompt_args
    # The archived reference predates global PyTorch/cuDNN determinism. Its FA4
    # flag controlled backward only; eval runs forward without gradients. Keep
    # that numerical policy for the exact reference check, while SP2 repeat and
    # DCP inference exercise runtime.deterministic=true.
    deterministic = not check_reference
    with patch.dict(
        os.environ,
        {
            "LTX_DMD_EVAL_REFERENCE": "1" if check_reference else "0",
            "LTX_DMD_EVAL_REPEAT": "1" if usp_size == 2 else "0",
        },
    ):
        torchrun(
            Path(__file__),
            "experiment=ltx23_dmd_eval",
            f"model.pretrained_path={base}",
            f"evaluation.checkpoint_path={generator}",
            f"evaluation.sources.0.cache_dir={cache}",
            "model.extras.seconds=2",
            "attention.backend=flash_attention_4",
            f"runtime.deterministic={str(deterministic).lower()}",
            f"parallel.usp_size={usp_size}",
            f"runtime.output_dir={output}",
            *prompt_args,
            timeout=1200,
        )
    import av

    videos = list((output / "samples").glob("*.mp4"))
    assert len(videos) == expected_samples
    assert not list(output.rglob("*.tmp.mp4"))
    for path in videos:
        with av.open(str(path)) as container:
            assert len(container.streams.video) == len(container.streams.audio) == 1
    assert not list(output.rglob("*.dcp"))


if __name__ == "__main__":
    import torch
    from transformers import AutoModelForImageTextToText

    from vforge.models.ltx.decoding import LtxDecoder
    from vforge.models.ltx.text_encoder import encode_prompts
    from vforge.tasks.eval import main
    from vforge.tasks.ltx.dmd.eval import LtxDmdInference

    _original_infer = LtxDmdInference.infer_sample
    _original_decode = LtxDecoder.decode

    repeat_infer = os.environ.get("LTX_DMD_EVAL_REPEAT") == "1"
    repeat_decode = repeat_infer

    def checked_infer(self, sample, *, seed):
        global repeat_infer
        rng = torch.cuda.get_rng_state().clone()
        first = _original_infer(self, sample, seed=seed)
        for key, value in first["latents"].items():
            assert torch.isfinite(value).all(), key
        if repeat_infer:
            second = _original_infer(self, sample, seed=seed)
            for key in first["latents"]:
                assert torch.equal(first["latents"][key], second["latents"][key]), key
            repeat_infer = False
            print("LTX_DMD_EVAL_REPEAT_BITWISE", flush=True)
        assert torch.equal(rng, torch.cuda.get_rng_state()), "inference consumed global RNG"
        if os.environ.get("LTX_DMD_EVAL_REFERENCE") == "1":
            baseline = json.loads(Path(__file__).with_name("dmd_eval_b300.json").read_text())
            for key, expected in baseline["latents"].items():
                value = first["latents"][key].contiguous()
                assert list(value.shape) == expected["shape"]
                assert str(value.dtype) == expected["dtype"]
                digest = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
                assert digest == expected["sha256"], f"reference {key} latent changed: {digest}"
            print("LTX_DMD_EVAL_REFERENCE_BITWISE", flush=True)
        return first

    def checked_decode(self, values, layout):
        global repeat_decode
        first = _original_decode(self, values, layout)
        for value in first[:2]:
            assert torch.isfinite(value).all()
        if repeat_decode:
            second = _original_decode(self, values, layout)
            for actual, expected in zip(first[:2], second[:2], strict=True):
                assert torch.equal(actual, expected), (
                    "decoded tensor changed between identical calls"
                )
            assert first[2] == second[2]
            repeat_decode = False
            print("LTX_DMD_DECODE_REPEAT_BITWISE", flush=True)
        return first

    with (
        patch.object(LtxDmdInference, "infer_sample", checked_infer),
        patch.object(LtxDecoder, "decode", checked_decode),
        patch.object(
            AutoModelForImageTextToText,
            "from_pretrained",
            wraps=AutoModelForImageTextToText.from_pretrained,
        ) as gemma_load,
        patch("vforge.tasks.ltx.dmd.eval.encode_prompts", wraps=encode_prompts) as encode,
    ):
        main()
        assert gemma_load.call_count == encode.call_count <= 1
        if os.environ.get("RANK", "0") == "0":
            assert not repeat_infer and not repeat_decode, "repeat checks did not run"
