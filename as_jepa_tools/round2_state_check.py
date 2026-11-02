"""Matched-noise renderer-use gate with the uploaded forecast channel and actual gap anchors."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def state_conditions(model, mu, actions, noise, forecast_noise):
    """Swap means, keep each recipient's noise/actions. Forecast calls the unchanged training helper."""
    b, w, k, d = mu.shape
    s = mu + model.sigma0 * noise
    swapped = mu.flip(0) + model.sigma0 * noise
    act = model.action_encoder(actions)
    scale = model.sigma0 if model.forecast_noise == 'sample' else 0.
    one_step = model._prior_mean(s.flatten(2)[:, :-1], act[:, :-1]).reshape(b, w - 1, k, d)
    other_step = model._prior_mean(swapped.flatten(2)[:, :-1], act[:, :-1]).reshape_as(one_step)
    result = dict(posterior=s[:, 1:], posterior_shuffled=swapped[:, 1:],
                  one_step=one_step + scale * forecast_noise,
                  one_step_shuffled=other_step + scale * forecast_noise,
                  forecast=model._forecast(s, act, forecast_noise, scale),
                  forecast_shuffled=model._forecast(swapped, act, forecast_noise, scale),
                  wrong_actions=model._forecast(s, model.action_encoder(actions.flip(0)), forecast_noise, scale))
    result['zero_state'] = torch.zeros_like(result['forecast'])
    return result


@torch.no_grad()
def check(model, cfg, cases=16, seed=1042, fp32=True):
    from as_jepa_tools.benchmark_gap import make_benchmark_dataset
    from as_jepa_lewm.forecast_scored import ar1
    from vforge.models.as_jepa.pixels import pack
    from experiments.trajectory_search_20261101.common import state_hash
    before = state_hash(model)
    ds = make_benchmark_dataset(cfg, split='validation')
    indices = np.random.default_rng(seed).choice(len(ds), cases, replace=False).tolist()
    rows, embeddings = [], []
    for batch in DataLoader(Subset(ds, indices), batch_size=2, num_workers=0):
        metadata = [dict(episode=int(ep), start=int(start), anchor_row=int(ar), anchor_fallback=bool(fb))
                    for ep, start, ar, fb in zip(batch['episode'], batch['start'], batch['anchor_row'], batch['anchor_fallback'])]
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        torch.manual_seed(seed + len(rows))
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=not fp32):
            pixels, render, anchor = model.prepare(batch)
            mu = model.encode(dict(pixels=pixels))['emb'].float()
            b, w, _ = mu.shape
            mu = mu.reshape(b, w, model.state_slots, model.state_width)
            noise = ar1(torch.randn_like(mu), model.rho)
            xi = torch.randn_like(mu[:, 1:])
            conditions = state_conditions(model, mu, batch['action'], noise, xi)
            y = pack(render[:, 1:].flatten(0, 1))
            anc = pack(anchor).repeat_interleave(w - 1, 0)
            eps = torch.randn_like(y)
            errors = {}
            for level in [.2, .5, .8, .95]:
                t = torch.full((len(y), 1), level, device='cuda')
                for name, state in conditions.items():
                    with model.renderer.anchored(anc):
                        velocity = model.renderer((1 - level) * y + level * eps, t,
                                                  condition=state.flatten(0, 1))
                    errors[f't{level}_{name}'] = (y - eps - velocity).float().square().flatten(1).mean(1).reshape(b, w - 1).mean(1).cpu().tolist()
                with model.renderer.anchored(anc.flip(0)):
                    state = conditions['forecast'].flatten(0, 1)
                    velocity = model.renderer((1 - level) * y + level * eps, t, condition=state)
                errors[f't{level}_wrong_anchor'] = (y - eps - velocity).float().square().flatten(1).mean(1).reshape(b, w - 1).mean(1).cpu().tolist()
        embeddings.append(mu.cpu().flatten(0, 1).flatten(1))
        rows.extend(dict(meta, **{key: value[i] for key, value in errors.items()}) for i, meta in enumerate(metadata))
    summary = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key.startswith('t')}
    spread = float(torch.cat(embeddings).std(0).mean() / model.sigma0)
    channel = 'posterior' if model.lam == 0 else model.horizon if model.horizon == 'one_step' else 'forecast'
    correct = summary[f't0.95_{channel}']
    swapped = summary[f't0.95_{channel}_shuffled']
    increase = (swapped - correct) / max(correct, 1e-30)
    assert before == state_hash(model)
    return dict(feature_precision='FP32' if fp32 else 'BF16 autocast', feature_std_over_sigma0=spread,
                trained_condition=channel, state_swap_relative_error_increase=increase,
                use_gate=spread >= .3 and increase >= .05, summary=summary, cases=rows,
                state_before=before, state_after=state_hash(model), model_updates=0,
                scope='Fixed t/noise, same recipient actions, swap encoded means only. Actual configured gap anchors. Both seeds must pass. This is a renderer-use diagnostic, not a task-success result.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--cases', type=int, default=16)
    p.add_argument('--bf16', action='store_true')
    a = p.parse_args()
    from as_jepa_tools.evaluate import load_model
    from vforge.ops import OpsConfig, initialize_ops
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest' if not a.bf16 else 'high')
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=torch.device('cuda'), parallel_state=None)
    cfg = json.loads(Path(a.config).read_text())
    model = load_model(a.checkpoint).eval()
    assert cfg['model_class'] == 'forecast_scored'
    result = check(model, cfg, a.cases, fp32=not a.bf16)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'state-check.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
