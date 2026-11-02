"""Renderer-free evaluation: train-only readout, noisy information, physical rollouts and action identification."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def fit_probe(x, y):
    sx, sy = StandardScaler().fit(x), StandardScaler().fit(y)
    probe = RidgeCV(alphas=np.logspace(-4, 4, 9), cv=5).fit(sx.transform(x), sy.transform(y))
    def read(value):
        shape = value.shape[:-1]
        result = sy.inverse_transform(probe.predict(sx.transform(value.reshape(-1, value.shape[-1]))))
        return result.reshape(*shape, y.shape[-1])
    return read, float(probe.alpha_)


def targets(state):
    # Same six pose units as evaluate.pose_net: xy /512, then sin/cos(theta).
    return torch.cat([state[..., :4] / 512, state[..., 4:5].sin(), state[..., 4:5].cos()], -1)


def batch_cuda(batch):
    return {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def information(model, data, seed=20261029):
    from as_jepa_tools.data import PushTWindows
    from vforge.models.as_jepa.codes import entry_nats, surprise_nats
    ds = PushTWindows(data, window=model.window, gap=model.anchor_gap, split='test')
    indices = np.random.default_rng(seed).choice(len(ds), 128, replace=False).tolist()
    means, anchors, predictions, samples, sample_predictions, records = [], [], [], [], [], []
    torch.manual_seed(seed)
    for batch in DataLoader(Subset(ds, indices), batch_size=8, num_workers=0):
        records.extend(dict(episode=int(ep), start=int(start), anchor_row=int(row))
                       for ep, start, row in zip(batch['episode'], batch['start'], batch['anchor_row']))
        batch = batch_cuda(batch)
        flat = model.encode(dict(pixels=model.frames(batch)))['emb'].float()
        all_mu = flat.reshape(len(flat), model.window + 1, model.state_slots, model.state_width)
        mu, anchor = all_mu[:, :model.window], all_mu[:, model.window]
        s = mu + model.sigma0 * torch.randn_like(mu)
        action = model.action_encoder(batch['action'])
        sample_predicted = model.predict(s.flatten(2)[:, :-1], action[:, :-1]).reshape_as(mu[:, 1:])
        inputs = s if model.prior_input == 'sample' else mu
        predicted = model.predict(inputs.flatten(2)[:, :-1], action[:, :-1]).reshape_as(mu[:, 1:])
        for destination, value in [(means, mu), (anchors, anchor), (predictions, predicted),
                                   (samples, s), (sample_predictions, sample_predicted)]:
            destination.append(value.cpu())
    mu, anchor, predicted, s, sample_predicted = [torch.cat(v).cuda() for v in [means, anchors, predictions, samples, sample_predictions]]
    diff = (mu[:, 1:] - anchor[:, None]).flatten(2).double().flatten(0, 1)
    covariance = diff.T @ diff / (2 * len(diff) * model.sigma0 ** 2)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    inputs = s if model.prior_input == 'sample' else mu
    def slope(pred, prev):
        truth, estimate = (mu[:, 1:] - prev).double(), (pred - prev).double()
        return float((truth * estimate).sum() / estimate.square().sum().clamp_min(1e-30))
    entry = float(entry_nats(mu[:, 0], model.sigma0).mean())
    surprise = float(surprise_nats(mu[:, 1:], predicted, model.sigma0).sum(1).mean())
    return dict(windows=128, records=records, eigenvalues_over_sigma2=eigenvalues.cpu().tolist(),
                state_spread_over_sigma0=float(mu.flatten(0, 1).flatten(1).std(0).mean() / model.sigma0),
                nats_per_direction=(.5 * torch.log1p(eigenvalues)).cpu().tolist(),
                diagnostics={k: float(v) for k, v in model.information_diagnostics(mu[:, 1:], anchor).items()},
                information_nats=float(model.information(mu[:, 1:], anchor)),
                rate_nats_per_window=entry + surprise, entry_nats_per_window=entry, surprise_nats_per_window=surprise,
                calibration_slope=slope(predicted, inputs[:, :-1]),
                calibration_slope_sample_input=slope(sample_predicted, s[:, :-1]))


@torch.no_grad()
def rollouts(model, data, readout, windows, fp32):
    from as_jepa_tools.data import PushTWindows
    from as_jepa_tools.evaluate import predict_states, select_windows
    ds = PushTWindows(data, window=16, gap=model.anchor_gap, split='test')
    indices = select_windows(ds, windows, seed=20261029)
    prediction, truth, copy, latent, last, encoded, moving, records = [], [], [], [], [], [], [], []
    for batch in DataLoader(Subset(ds, indices), batch_size=8, num_workers=0):
        state = batch['state'].float()
        moving.extend(((state[:, :, 2:4] - state[:, :1, 2:4]).norm(dim=-1).amax(1) > 10).tolist())
        records.extend(dict(episode=int(ep), start=int(start)) for ep, start in zip(batch['episode'], batch['start']))
        batch = batch_cuda(batch)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=not fp32):
            pixels, _, _ = model.prepare(batch)
            mu = model.encode(dict(pixels=pixels))['emb'].float()
            predicted = predict_states(model, pixels[:, :3], batch['action'], 13).float()
        prediction.append(readout(predicted.cpu().numpy()))
        copy.append(np.repeat(readout(mu[:, 2:3].cpu().numpy()), 13, axis=1))
        truth.append(targets(batch['state'][:, 3:].float()).cpu().numpy())
        latent.append(predicted.cpu().numpy()); encoded.append(mu[:, 3:].cpu().numpy()); last.append(mu[:, 2:3].cpu().numpy())
    pred, target, copied = np.concatenate(prediction), np.concatenate(truth), np.concatenate(copy)
    errors, baseline = {}, {}
    for name, start in [('agent', 0), ('block', 2)]:
        errors[name] = np.linalg.norm(pred[..., start:start+2] - target[..., start:start+2], axis=-1)
        baseline[name] = np.linalg.norm(copied[..., start:start+2] - target[..., start:start+2], axis=-1)
    horizons = [1, 3, 7, 13]
    summary = {}
    for name in errors:
        summary[name] = {f'h{h}': dict(mean_error=float(errors[name][:, h-1].mean()),
                     copy_error=float(baseline[name][:, h-1].mean()),
                     ratio_to_copy=float(errors[name][:, h-1].mean() / max(baseline[name][:, h-1].mean(), 1e-30))) for h in horizons}
    motion = np.asarray(moving, dtype=bool)
    moving_ratio = float(errors['block'][motion, 6].mean() / max(baseline['block'][motion, 6].mean(), 1e-30)) if motion.any() else None
    latent, encoded, last = np.concatenate(latent), np.concatenate(encoded), np.concatenate(last)
    drift = ((latent - encoded) ** 2).mean((0, 2)) / np.maximum(((encoded - last) ** 2).mean((0, 2)), 1e-30)
    return dict(windows=len(records), selection_seed=20261029, context=3, horizon=13,
                feature_precision='FP32' if fp32 else 'BF16 autocast', pose_units='position /512',
                errors=summary, h7_moving_block_ratio=moving_ratio, moving_block_windows=int(motion.sum()),
                drift_vs_copy=drift.tolist(), cases=records)


@torch.no_grad()
def counterfactual(model, data, readout, count, fp32):
    """Identical case/donor selection and simulator reset as counterfactual.py; no renderer."""
    from as_jepa_tools.data import PushTWindows
    from as_jepa_tools.evaluate import predict_states
    from as_jepa_tools.counterfactual import costs
    from stable_worldmodel.envs.pusht import PushT as PushTEnv
    ds = PushTWindows(data, window=16, gap=8, split='test'); ds.open()
    rng = np.random.default_rng(20261029)
    candidates = rng.choice(len(ds), min(5000, len(ds)), replace=False)
    eligible = [int(i) for i in candidates if ds.locate(i)[1] >= 8][:count]
    root = Path(data)
    raw = np.load(root / 'action_raw.npy', mmap_mode='r')
    raw_rows = np.load(root / 'raw_rows.npy')
    state = np.load(root / 'states.npy', mmap_mode='r')
    card = json.loads((root / 'dataset_card.json').read_text())
    mean, std = np.asarray(card['action_mean']), np.asarray(card['action_std'])
    env = PushTEnv(render_mode='rgb_array')
    records = []
    try:
        for case, index in enumerate(eligible):
            batch = ds[index]; ep, start = ds.locate(index); rows = batch['rows'].numpy()
            origin_raw = int(raw_rows[int(rows[2])]); initial = np.array(state[int(rows[2])])
            donors = [origin_raw] + [int(raw_rows[int(ds[int(rng.integers(len(ds)))]['rows'][2])]) for _ in range(3)]
            truth, normalized = [], []
            for donor in donors:
                sequence = np.nan_to_num(np.array(raw[donor:donor+40]), nan=0.)
                assert len(sequence) == 40
                env.reset(seed=case); env._set_state(initial.copy()); future = []
                for t, action in enumerate(sequence):
                    env.step(action)
                    if (t + 1) % 5 == 0:
                        future.append(np.r_[np.array(env.agent.position), np.array(env.block.position)])
                truth.append(future)
                normalized.append(((sequence - mean) / std).reshape(8, 10))
            actions = torch.from_numpy(np.stack([np.concatenate([batch['action'][:2].numpy(), value], 0) for value in normalized])).cuda().float()
            context = model.normalize(batch['pixels'][:3][None].cuda()).expand(4, -1, -1, -1, -1)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=not fp32):
                predicted = predict_states(model, context, actions, 8).float()
            decoded = readout(predicted.cpu().numpy())[..., :4] * 512
            matrix = costs(decoded, np.asarray(truth))
            assignment = matrix.argmin(1)
            records.append(dict(episode=ep, start=start, donor_raw_rows=donors, assigned=assignment.tolist(),
                                correct=(assignment == np.arange(4)).tolist(), cost_matrix=matrix.tolist()))
    finally:
        env.close()
    return dict(cases=len(records), selection_seed=20261029, chance=.25,
                accuracy=float(np.mean([r['correct'] for r in records])), records=records,
                scope='Counterfactual.py case/donor/reset protocol; latent linear readout. Reset cannot restore unsaved block velocity.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--data', required=True); p.add_argument('--output', required=True)
    p.add_argument('--windows', type=int, default=256); p.add_argument('--fp32', action='store_true')
    p.add_argument('--train', type=int, default=20000); p.add_argument('--test', type=int, default=5000)
    a = p.parse_args()
    from as_jepa_tools.evaluate import load_model
    from as_jepa_tools.probes import features
    from experiments.trajectory_search_20261101.common import state_hash
    from vforge.ops import OpsConfig, initialize_ops
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest'); torch.manual_seed(20261027)
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=torch.device('cuda'), parallel_state=None)
    model = load_model(a.checkpoint).eval()
    assert hasattr(model, 'information_diagnostics') and model.renderer is None
    before = state_hash(model)
    x, y, names = features(model, a.data, 'train', a.train, 17, fp32=True)
    tx, ty, _ = features(model, a.data, 'test', a.test, 18, fp32=True)
    y, ty = y[:, :6].copy(), ty[:, :6].copy(); y[:, :4] /= 512; ty[:, :4] /= 512
    readout, alpha = fit_probe(x, y)
    rng = np.random.default_rng(20261027)
    noisy_readout, noisy_alpha = fit_probe(x + model.sigma0 * rng.standard_normal(x.shape), y)
    clean_prediction = readout(tx)
    noisy_prediction = noisy_readout(tx + model.sigma0 * rng.standard_normal(tx.shape))
    probes = dict(targets=names[:6], pose_units='position /512', train_frames=len(x), test_frames=len(tx),
                  clean_r2=r2_score(ty, clean_prediction, multioutput='raw_values').tolist(),
                  noisy_r2=r2_score(ty, noisy_prediction, multioutput='raw_values').tolist(),
                  clean_floor_mse=((clean_prediction - ty) ** 2).mean(0).tolist(),
                  clean_alpha=alpha, noisy_alpha=noisy_alpha,
                  scope='Both scalers and RidgeCV fit only training frames; noisy codes use declared sigma0. No rollout/success label fits.')
    result = dict(checkpoint=a.checkpoint, probes=probes, rollout=rollouts(model, a.data, readout, a.windows, a.fp32),
                  information=information(model, a.data),
                  counterfactual=counterfactual(model, a.data, readout, 16 if a.windows == 64 else 64, a.fp32) if model.domain == 'standard' else None)
    assert before == state_hash(model)
    result.update(state_before=before, state_after=state_hash(model), model_updates=0)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    (out / 'latent-eval.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(probes=probes, rollout=result['rollout']['errors'], h7_moving_block_ratio=result['rollout']['h7_moving_block_ratio'])))


if __name__ == '__main__':
    main()
