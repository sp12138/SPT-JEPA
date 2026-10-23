"""Generate held-out RGB16 rollouts through the Native evaluator."""
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from vforge.tasks.eval import run_eval

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("checkpoint")
parser.add_argument("output")
parser.add_argument("--samples", type=int, default=8)
args = parser.parse_args()
cfg = OmegaConf.load(args.config)
cfg.runtime.output_dir = args.output
cfg.runtime.weights_strict = True
cfg.checkpoint.enabled = False
cfg.resume.path = "none"
cfg.evaluation.data_source = "validation"
cfg.evaluation.manifest = cfg.data.val_manifest
cfg.evaluation.checkpoint_path = args.checkpoint
cfg.evaluation.checkpoint_format = "dcp"
cfg.evaluation.output_dir = args.output
cfg.evaluation.num_samples = args.samples
cfg.evaluation.seed = 20260920
cfg.evaluation.num_inference_steps = 50
cfg.evaluation.guidance_scale = 1
cfg.model.extras.ema.evaluation_weights = "ema"
cfg.model.extras.conditioner.context_mask_ratio = 0
run_eval(cfg)
