"""Frozen action-use and within-scene variation diagnostics, no SGD/probes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def run(args):
    old = Path(args.old_root)
    sys.path.insert(0, str(old / 'helper-v8'))
    from bridge import install_paths, cost_adapter
    install_paths(old)
    import torch
    import numpy as np
    from droid_data import DROIDClipReader
    from droid_eval import img_transform
    from train import digest_state
    from vforge.ops import OpsConfig, initialize_ops
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision('high')
    name = 'droid_lewm'
    run_dir = old / 'training' / f'{args.arm}-seed{args.seed}'
    checkpoint = run_dir / f'{name}_epoch_100_object.ckpt'
    core = torch.load(checkpoint, map_location='cpu', weights_only=False).eval().requires_grad_(False)
    before = digest_state(core)
    assert before == json.loads((run_dir / 'epoch100-ready.json').read_text())['model_state_sha256']
    candidate = args.arm != 'lewm'
    if candidate:
        initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=torch.device('cuda'), parallel_state=None)
        model = cost_adapter(core).cuda()
    else:
        model = core.get_level(1).cuda()
    artifact = core.normalizer_artifact
    stats = artifact['stats']['action']
    mean = torch.tensor(np.asarray(stats['mean']).reshape(-1), device='cuda')
    std = torch.tensor(np.asarray(stats['std']).reshape(-1), device='cuda')
    manifest = old / 'public/h_jepa/droid_assets/droid_clips_waypoint_curated16v2_5fps_gw36.json'
    reader = DROIDClipReader(None, ['left_mp4_path'], 37, 5, frozen_clips=str(manifest), deterministic_getitem=True)
    tf = img_transform(256)
    rows, all_z = [], []
    with torch.no_grad():
        for k in range(16):
            obs, actions, _, _, _ = reader[k]
            pixels = tf(obs['visual']).cuda()[None]
            z = model.encode(dict(pixels=pixels))['embed_0']
            all_z.append(z[0].cpu().double())
            normalized = ((actions[:36].cuda() - mean) / std).float()
            alternatives = dict(factual=normalized, zero=((torch.zeros_like(normalized)-mean)/std).float(),
                                reverse=normalized.flip(0))
            values = {}
            for label, action in alternatives.items():
                for h in [1, 7, 36]:
                    info = dict(pixels=pixels[:, None, :1], goal=pixels[:, None, h:h+1],
                                action=torch.zeros(1, 1, 1, 7, device='cuda'))
                    values[f'{label}_{h}'] = float(model.get_cost(info, action[None, None, :h]))
            # teacher-forced one-step prediction versus copying the observed state.
            ae = model.action_embed(normalized[None])
            pred = model.predict(z[:, :-1], ae)
            values['one_step_mse'] = float((pred-z[:, 1:]).square().mean())
            values['copy_mse'] = float((z[:, :-1]-z[:, 1:]).square().mean())
            values['one_step_over_copy'] = values['one_step_mse']/max(values['copy_mse'], 1e-30)
            rows.append(dict(clip=k, values=values,
                             rgb_sha256=hashlib.sha256(obs['visual'].numpy().tobytes()).hexdigest(),
                             action_sha256=hashlib.sha256(actions.numpy().tobytes()).hexdigest()))
    z = torch.stack(all_z)
    within = z.var(1, unbiased=False).mean()
    total = z.flatten(0, 1).var(0, unbiased=False).mean()
    after = digest_state(core)
    assert before == after
    value = dict(passed=True, arm=args.arm, seed=args.seed, model_sha256=before, state_unchanged=True,
                 within_scene_variance_share=float(within/total.clamp_min(1e-30)), rows=rows,
                 interpretation='Observed actions are factual retrospective diagnostics, not optimal labels or simulated counterfactuals. Own-latent costs are not shared physical units.')
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['old-root', 'arm', 'output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--seed', type=int, required=True)
    run(p.parse_args())
