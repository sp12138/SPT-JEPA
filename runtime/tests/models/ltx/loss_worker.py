"""Launch a real LTX-2.3 training entrypoint for a loss gate."""

from __future__ import annotations

import argparse
import os
import runpy
import sys

import torch

EXPERIMENTS = ("ltx23_ti2av_sft", "ltx23_causal_ar_tf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["WANDB_MODE"] = "disabled"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.argv = [
        "vforge.tasks.train",
        f"experiment={args.experiment}",
        f"model.pretrained_path={args.model_path}",
        f"data.cache_dir={args.data_root}",
        "data.num_workers=0",
        "data.shuffle=false",
        "data.persistent_workers=false",
        "data.prefetch_factor=null",
        f"runtime.output_dir={args.output_dir}",
        "runtime.max_steps=10",
        "scheduler.total_iters=10",
        "parallel.dp_size=2",
        "parallel.usp_size=2",
        "parallel.cp_size=1",
        "parallel.tp_size=1",
        "attention.backend=flash_attention_4",
        "attention.allow_fallback=false",
        "runtime.deterministic=true",
        "attention.blas_backend=cublas",
        "checkpoint.enabled=false",
        "resume.path=none",
    ]
    runpy.run_module("vforge.tasks.train", run_name="__main__")


if __name__ == "__main__":
    main()
