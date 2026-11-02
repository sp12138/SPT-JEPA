"""Offline clean/noisy physical probes, without the renderer audit main entry point."""
import argparse
from pathlib import Path
import torch
from common import read, write, sha


def main(checkpoint, config, output):
    from as_jepa_tools.evaluate import load_model
    from as_jepa_tools.benchmark_information_audit import run_probes
    from experiments.trajectory_search_20261101.common import state_hash
    from vforge.ops import OpsConfig, initialize_ops
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=torch.device('cuda'), parallel_state=None)
    model = load_model(checkpoint); cfg = read(config); before = state_hash(model)
    probes = run_probes(model, cfg, 256, 1042)
    assert before == state_hash(model)
    write(Path(output) / 'content.json', dict(passed=True, probes=probes, feature_precision='FP32',
          state_before=before, state_after=state_hash(model), model_updates=0,
          checkpoint_sha256=sha(Path(checkpoint) / 'weights.pt'), config_sha256=sha(config)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); main(a.checkpoint, a.config, a.output)
