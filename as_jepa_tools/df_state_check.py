"""Decoder-free TwoRoom state gate; frozen encoders and diagnostic labels only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


@torch.no_grad()
def check(model, cfg, count=256, seed=1042):
    from as_jepa_tools import benchmark_information_audit as audit
    from as_jepa_tools.benchmark_gap import make_benchmark_dataset
    from experiments.trajectory_search_20261101.common import state_hash
    before = state_hash(model)
    captured = {}
    original = audit.probe_features
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured[args[2]] = result
        return result
    audit.probe_features = capture
    try:
        probes = audit.run_probes(model, cfg, count, seed)
    finally:
        audit.probe_features = original
    validation = captured['validation'][0]
    spread = float(validation.reshape(-1, validation.shape[-1]).std(0).mean() / model.sigma0)
    assert probes['label_keys'][0] == 'pos_agent', probes['label_keys']
    clean = probes['scores']['clean']['r2_against_train_mean'][:2]
    noisy = probes['scores'][f'noisy_sigma{model.sigma0:g}']['r2_against_train_mean'][:2]
    clean_agent, noisy_agent = float(np.mean(clean)), float(np.mean(noisy))
    ds = make_benchmark_dataset(cfg, split='validation')
    indices = audit.episode_indices(ds, 128, seed)
    means, anchors, rows = [], [], []
    for batch in DataLoader(Subset(ds, indices), batch_size=8, num_workers=0):
        rows.extend(dict(episode=int(ep), start=int(start), anchor_row=int(ar))
                    for ep, start, ar in zip(batch['episode'], batch['start'], batch['anchor_row']))
        batch = audit.device_batch(batch)
        mu = model.encode(dict(pixels=model.frames(batch)))['emb'].float()
        mu = mu.reshape(len(mu), model.window + 1, model.state_slots, model.state_width)
        means.append(mu[:, 1:model.window].cpu())
        anchors.append(mu[:, model.window].cpu())
    mu, anchor = torch.cat(means).cuda(), torch.cat(anchors).cuda()
    diagnostics = {key: float(value) for key, value in model.information_diagnostics(mu, anchor).items()}
    info = float(model.information(mu, anchor))
    assert before == state_hash(model)
    return dict(feature_precision='FP32', feature_std_over_sigma0=spread, clean_agent_r2=clean_agent,
                noisy_agent_r2=noisy_agent, state_gate=spread >= .3 and noisy_agent >= .5,
                information_nats=info, information_diagnostics=diagnostics, information_windows=len(indices),
                information_rows=rows, probes=probes, state_before=before, state_after=state_hash(model),
                model_updates=0, scope='Single-seed state gate. Both seeds must pass before6000. No renderer or planning is invoked; labels are offline diagnostics only.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=256)
    p.add_argument('--seed', type=int, default=1042)
    a = p.parse_args()
    from as_jepa_tools.evaluate import load_model
    from vforge.ops import OpsConfig, initialize_ops
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=torch.device('cuda'), parallel_state=None)
    cfg = json.loads(Path(a.config).read_text())
    model = load_model(a.checkpoint).eval()
    assert cfg['model_class'] == 'decoder_free' and model.renderer is None
    result = check(model, cfg, a.count, a.seed)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'state-check.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['probes', 'information_rows']}))


if __name__ == '__main__':
    main()
