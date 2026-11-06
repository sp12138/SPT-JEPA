"""Frozen future-observation diagnostics; no action target or optimizer."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def run(a):
    from as_jepa_lewm.factory import build_model
    from as_jepa_tools.predictive_kernel_data import KernelTransitions
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads((a.run / 'config.json').read_text())
    model = build_model(cfg).to(a.device).eval().requires_grad_(False)
    weights = a.run / f'model-step{a.step:07d}/weights.pt'
    model.load_state_dict(torch.load(weights, map_location=a.device, weights_only=True))
    data = KernelTransitions(cfg, 'test')
    candidates = []
    for index in range(len(data)):
        record, t = data.locate(index)
        if t + 2 < record['length']:
            candidates.append(index)
    chosen = np.asarray(candidates)[np.linspace(0, len(candidates) - 1, min(512, len(candidates)), dtype=int)]
    rows, identities = [], []
    with torch.inference_mode():
        for start in range(0, len(chosen), 32):
            items = []
            for index in chosen[start:start + 32]:
                record, t = data.locate(int(index))
                z = data.shards[record['shard']]['features']
                offset = record['offset']
                def feature(i):
                    return torch.from_numpy(np.asarray(z[offset + i]).copy())
                items.append(dict(previous=feature(max(t - 1, 0)), current=feature(t), future=feature(t + 1),
                                  goal=feature(record['length'] - 1), valid=float(t > 0),
                                  control=torch.from_numpy(np.array(data.shards[record['shard']]['inputs'][offset + t]))))
                identities.append(dict(episode=record['episode'], start=t, target=t + 1, goal=record['length'] - 1))
            batch = {key: torch.stack([x[key] if torch.is_tensor(x[key]) else torch.tensor(x[key]) for x in items]).to(a.device)
                     for key in items[0]}
            previous, current, future, goal = [batch[key] for key in ['previous', 'current', 'future', 'goal']]
            control, valid = batch['control'], batch['valid']
            zero = current.new_zeros(len(current))
            drift, effect, variance = model.fields(previous, current, torch.zeros_like(goal), control, zero, valid)
            if model.affine:
                active = torch.einsum('na,nachw->nchw', control, effect)
                predicted = drift + active
                passive = drift
                reverse = drift - active
            else:
                predicted = drift
                passive = model.fields(previous, current, torch.zeros_like(goal), torch.zeros_like(control), zero, valid)[0]
                reverse = model.fields(previous, current, torch.zeros_like(goal), -control, zero, valid)[0]
            delta = future - current
            metrics = dict(copy_mse=delta.square().flatten(1).mean(1),
                           factual_mse=(predicted-delta).square().flatten(1).mean(1),
                           zero_input_mse=(passive-delta).square().flatten(1).mean(1),
                           reverse_input_mse=(reverse-delta).square().flatten(1).mean(1),
                           predicted_delta_rms=predicted.square().flatten(1).mean(1).sqrt(),
                           observed_delta_rms=delta.square().flatten(1).mean(1).sqrt(),
                           input_effect_rms=(predicted-passive).square().flatten(1).mean(1).sqrt())
            if model.boundary:
                desired, _, boundary_variance = model.fields(previous, current, goal, torch.zeros_like(control), zero + 1, valid)
                metrics['boundary_mse'] = (desired-delta).square().flatten(1).mean(1)
                metrics['boundary_delta_rms'] = desired.square().flatten(1).mean(1).sqrt()
                if model.affine:
                    command = model.box_projection(drift, effect, desired, variance+boundary_variance)
                    achieved = drift + torch.einsum('na,nachw->nchw', command, effect)
                    metrics['projection_residual_mse'] = (achieved-desired).square().flatten(1).mean(1)
                    metrics['requested_delta_mse'] = (desired-drift).square().flatten(1).mean(1)
                    metrics['inferred_control_rms'] = command.square().mean(1).sqrt()
            for k in range(len(current)):
                rows.append({name: float(value[k]) for name, value in metrics.items()})
    assert all(np.isfinite(v) for r in rows for v in r.values())
    report = dict(passed=True, step=a.step, windows=len(rows), rows=rows,
                  medians={key: float(np.median([r[key] for r in rows])) for key in rows[0]},
                  prediction_beats_copy=sum(r['factual_mse'] < r['copy_mse'] for r in rows),
                  current_input_beats_zero=sum(r['factual_mse'] < r['zero_input_mse'] for r in rows),
                  identities=identities, manifest_sha256=hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest(),
                  weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(), model_updates=0,
                  scope='Observed future features only. Current executed command is a dynamics INPUT, never a control target/reference.'
                        'Zero/reversed inputs are model queries, not simulated counterfactual evidence. No performance claim.')
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['passed', 'step', 'windows', 'medians', 'prediction_beats_copy', 'current_input_beats_zero']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    run(p.parse_args())
