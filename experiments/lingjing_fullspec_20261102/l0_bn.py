"""The existing precise-BN moment control, adapted to FullPushT and its actual L0 window."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from common import read, write, sha


@torch.no_grad()
def main(run, step, output):
    from as_jepa_tools.evaluate import load_model
    from as_jepa_tools.data import FullPushTWindows
    from experiments.native_baseline_20261031.bn_calibrate import Moments
    from vforge.ops import OpsConfig, initialize_ops
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=torch.device('cuda'), parallel_state=None)
    run, out = Path(run), Path(output); out.mkdir(parents=True, exist_ok=False)
    cfg = read(run / 'config.json'); cp = run / f'model-step{step:07d}'; before = sha(cp / 'weights.pt')
    model = load_model(cp); assert model.mode == 'L0'
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ds = FullPushTWindows(cfg['data'], window=model.window, gap=8, split='train')
    rng = np.random.default_rng(1042); chosen = rng.choice(len(ds.episodes), 128, replace=False)
    indices = [int((ds.ends[i-1] if i else 0) + rng.integers(ds.counts[i])) for i in chosen]
    cls, actions, records = [], [], []
    for batch in DataLoader(Subset(ds, indices), batch_size=8, num_workers=0):
        records.extend(dict(episode=int(ep), start=int(start), rows=rows.tolist()) for ep, start, rows in zip(batch['episode'], batch['start'], batch['rows']))
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        pixels = model.prepare(batch)[0]
        cls.append(model.encoder(pixels.flatten(0, 1), interpolate_pos_encoding=True).last_hidden_state[:, 0].reshape(len(pixels), model.window, 192))
        actions.append(model.action_encoder(batch['action']))
    allowed = set(np.load(Path(cfg['data']) / 'test_episodes.npy').tolist())
    assert len({r['episode'] for r in records}) == 128 and not {r['episode'] for r in records}.intersection(allowed)
    cls, actions = torch.cat(cls), torch.cat(actions)
    def set_moments(name, value):
        module = dict(model.named_modules())[name]; moment = Moments(); array = value.float().cpu().numpy()
        for lo in range(0, len(array), 32):moment.update(array[lo:lo+32])
        np.testing.assert_allclose(moment.mean, array.astype(np.float64).mean(0), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(moment.variance, array.astype(np.float64).var(0, ddof=1), rtol=1e-9, atol=1e-12)
        module.running_mean.copy_(torch.from_numpy(moment.mean).to(module.running_mean))
        module.running_var.copy_(torch.from_numpy(moment.variance).to(module.running_var))
        return dict(name=name, vectors=moment.n)
    first = set_moments('projector.net.1', model.projector.net[0](cls.flatten(0, 1)))
    observed = model.projector(cls.flatten(0, 1)).reshape(128, model.window, 192)
    raw = model.predictor(observed[:, :-1], actions[:, :-1])
    second = set_moments('pred_proj.net.1', model.pred_proj.net[0](raw.flatten(0, 1)))
    predicted = model.predict(observed[:, :-1], actions[:, :-1])
    separate = torch.cat([model.predict(observed[i:i+1, :-1], actions[i:i+1, :-1]) for i in range(2)])
    replay = float((separate - predicted[:2]).square().mean()); assert replay < 1e-7
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    permitted = {prefix + '.' + suffix for prefix in ['projector.net.1', 'pred_proj.net.1'] for suffix in ['running_mean', 'running_var']}
    changed = {k for k in state if not torch.equal(state[k], original[k])}; assert changed <= permitted
    assert all(torch.isfinite(v).all() for v in state.values())
    export = out / 'model'; export.mkdir(); torch.save(state, export / 'weights.pt'); shutil.copy2(cp / 'config.json', export / 'config.json')
    assert before == sha(cp / 'weights.pt')
    write(out / 'proof.json', dict(passed=True, input_sha256=before, calibrated_sha256=sha(export / 'weights.pt'),
          first=first, second=second, changed_buffers=sorted(changed), records=records, sample_independence_mse=replay,
          scope='One fixed128-distinct-episode train-only pooled-moment recalibration. Only four BN running-statistic buffers may change; original model and all parameters remain exact. No validation statistics or labels.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', required=True); p.add_argument('--step', type=int, required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); main(a.run, a.step, a.output)
