"""Colour changes at fixed pose, and goal retrieval after recolouring; no fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


@torch.no_grad()
def measure(model, data, count=512, seed=20261029):
    from as_jepa_tools.data import PushTWindows
    from experiments.trajectory_search_20261101.common import state_hash
    before = state_hash(model)
    ds = PushTWindows(data, window=1, gap=0, split='test')
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), min(count, len(ds)), replace=False)
    batches = list(DataLoader(Subset(ds, indices), batch_size=len(indices), num_workers=0))
    assert len(batches) == 1
    raw = batches[0]
    episodes = raw['episode'].numpy()
    candidates = np.unique(ds.episodes)
    assert len(candidates) > 1
    donor = np.array([rng.choice(candidates[candidates != ep]) for ep in episodes])
    other = np.roll(np.arange(len(indices)), 1)
    assert np.all(other != np.arange(len(indices)))
    own, recoloured, moved = [], [], []
    def encode(batch):
        pixels = model.frames(batch)[:, :1] if hasattr(model, 'frames') else model.prepare(batch)[0]
        return model.encode(dict(pixels=pixels))['emb'][:, 0].float().cpu()
    for lo in range(0, len(indices), 32):
        hi = min(lo + 32, len(indices))
        batch = {k: v[lo:hi].cuda() if torch.is_tensor(v) else v for k, v in raw.items()}
        own.append(encode(batch))
        changed = dict(batch, episode=torch.as_tensor(donor[lo:hi], device='cuda'))
        recoloured.append(encode(changed))
        # Same palette A_i on both poses; do not let a different palette inflate the denominator.
        moved_batch = dict(batch, pixels=raw['pixels'][other[lo:hi]].cuda())
        moved.append(encode(moved_batch))
    a, b, pose = torch.cat(own).double(), torch.cat(recoloured).double(), torch.cat(moved).double()
    numerator = (a - b).square().sum(1).mean()
    denominator = (a - pose).square().sum(1).mean()
    own_pair_denominator = (a - a[other]).square().sum(1).mean()
    distance = torch.cdist(b, a, compute_mode='donot_use_mm_for_euclid_dist').square()
    nearest = distance.argmin(1)
    assert before == state_hash(model)
    return dict(frames=len(indices), feature_precision='FP32', seed=seed,
                invariance_index=float(numerator / denominator.clamp_min(1e-30)),
                same_pose_colour_squared_distance=float(numerator), same_palette_pose_squared_distance=float(denominator),
                original_palettes_pair_index=float(numerator / own_pair_denominator.clamp_min(1e-30)),
                recoloured_goal_retrieval=float((nearest == torch.arange(len(a))).double().mean()),
                records=[dict(index=int(idx), episode=int(ep), palette_B_episode=int(db), pose_partner_index=int(indices[j]))
                         for idx, ep, db, j in zip(indices, episodes, donor, other)],
                state_before=before, state_after=state_hash(model), model_updates=0,
                scope='Primary invariance denominator uses different poses under the SAME palette A_i. Also preserve the original-palette denominator. A low ratio alone cannot distinguish invariance from total collapse; retain retrieval and noisy physical probes.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--frames', type=int, default=512)
    a = p.parse_args()
    from as_jepa_tools.evaluate import load_model
    from vforge.ops import OpsConfig, initialize_ops
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=torch.device('cuda'), parallel_state=None)
    model = load_model(a.checkpoint).eval()
    assert model.domain == 'color'
    report = measure(model, a.data, a.frames)
    out = Path(a.output)
    out.mkdir(exist_ok=False, parents=True)
    (out / 'colour-invariance.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}))


if __name__ == '__main__':
    main()
