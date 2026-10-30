"""Read-only functional audit of the clean-latent auxiliary query path.

Recorded actions are inputs, never optimal-action labels. The same target and
per-clip projector are retained for every intervention. No physical labels.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from as_jepa_lewm.robust import RobustLeWM
from as_jepa_tools.benchmark_data import BenchmarkWindows
from as_jepa_tools.benchmark_health import finite
from vforge.ops import OpsConfig, initialize_ops


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root / 'config.json').read_text())
    torch.set_num_threads(2)
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=torch.device('cuda'), parallel_state=None)
    model = RobustLeWM(**cfg['model']).cuda().eval()
    model.load_state_dict(torch.load(root / 'model-step0000600/weights.pt', map_location='cpu', weights_only=True))
    assert model.response_query_mode != 'none'
    ds = BenchmarkWindows(cfg['data'], window=model.window, split='validation')
    indices = np.random.default_rng(1042).choice(len(ds), 64, replace=False).tolist()
    rows = []
    for batch in DataLoader(Subset(ds, indices), batch_size=4, num_workers=0):
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        pixels, render, _ = model.prepare(batch)
        mu = model.encode(dict(pixels=pixels))['emb']
        actions = batch['action']
        b, w, _ = mu.shape
        target = F.adaptive_avg_pool2d(render.float().flatten(0, 1), (8, 8)).reshape(b, w, -1)
        projector, rank = model.response_projector(actions)

        def forecast(initial, control):
            embedded = model.action_encoder(control)
            history = [initial]
            for t in range(w - 1):
                history.append(model.predict(torch.stack(history, 1), embedded[:, :t+1])[:, -1])
            return torch.stack(history, 1)

        wrong_actions = actions.clone()
        wrong_actions[:, :-1] = actions[:, :-1].roll(1, 1)
        states = dict(forecast=forecast(mu[:, 0], actions),
            other_initial=forecast(mu.roll(1, 0)[:, 0], actions),
            zero_initial=forecast(torch.zeros_like(mu[:, 0]), actions),
            wrong_actions=forecast(mu[:, 0], wrong_actions),
            encoded=mu, carry=mu[:, :1].expand_as(mu), zero=torch.zeros_like(mu))
        values = {}
        for name, state in states.items():
            error = model.response_query_head(state).float() - target
            values[name+'_anchor_mse'] = error[:, 0].square().mean(-1)
            values[name+'_direct_mse'] = error.square().mean((1, 2))
            residual = torch.bmm(projector, error)
            values[name+'_annihilator_mse'] = residual.square().sum((1, 2)) / (rank.clamp_min(1) * error.shape[-1])
        projected_target = torch.bmm(projector, target)
        values['target_direct_variation'] = (target - target[:, :1]).square().mean((1, 2))
        values['target_projected_energy'] = projected_target.square().sum((1, 2)) / (rank.clamp_min(1) * target.shape[-1])
        for i in range(b):
            rows.append(dict(episode=int(batch['episode'][i]), start=int(batch['start'][i]),
                residual_rank=int(rank[i]), **{k: float(v[i]) for k, v in values.items()}))
    report = dict(run=str(root), mode=model.response_query_mode, cases=rows,
        summary={k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in {'episode','start'}},
        scope='Frozen600 checkpoint; clean planner-path readout. Fixed targets/projector under all interventions. No update, no oracle labels, no task-performance claim.')
    assert finite(report)
    (out/'query-audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
