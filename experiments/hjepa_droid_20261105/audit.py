"""Finite corpus/split, real CPU restore and differentiable planning checks."""
import argparse
import hashlib
import json
from pathlib import Path

from bridge import install_paths, candidate_batch, cost_adapter


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def data_audit(root, host):
    from stable_worldmodel.data.utils import get_cache_dir
    from droid_data import DROIDClipReader
    droid = get_cache_dir() / 'droid'
    train = (droid / 'droid_paths_minus16_256p.csv').read_text().splitlines()
    val = (droid / 'droid_val_indist_256p.csv').read_text().splitlines()
    canonical = lambda path: str(Path(path)).split('/', 1)[1]
    train_ids = {canonical(row.split()[0]) for row in train}
    val_ids = {canonical(row.split()[0]) for row in val}
    manifest = root / 'public/h_jepa/droid_assets/droid_clips_waypoint_curated16v2_5fps_gw36.json'
    clips = json.loads(manifest.read_text())
    eval_ids = {canonical(row['episode_path']) for row in clips}
    assert not (train_ids & eval_ids), 'Released training split overlaps the planning clips'
    dataset = DROIDClipReader(None, ['left_mp4_path'], clips[0]['num_frames'], clips[0]['fps'],
                              frozen_clips=str(manifest), deterministic_getitem=True)
    import torch
    from droid_eval import img_transform
    from data import IMAGENET_STATS
    from bridge import candidate_config
    decoded = []
    for k in range(len(dataset)):
        obs, actions, states, reward, info = dataset[k]
        assert obs['visual'].dtype == torch.uint8
        assert torch.isfinite(actions).all() and actions.shape == (37, 7)
        # Public evaluation preprocessing is common; no candidate-specific resize.
        rgb = img_transform(256)(obs['visual'][0])
        assert rgb.shape == (3, 256, 256) and torch.isfinite(rgb).all()
        decoded.append(dict(clip=k, pixels_sha256=hashlib.sha256(obs['visual'].numpy().tobytes()).hexdigest(),
                            actions_sha256=hashlib.sha256(actions.numpy().tobytes()).hexdigest()))
    value = dict(passed=True, train_rows=len(train), val_rows=len(val), val_train_overlap=len(train_ids & val_ids),
                 planning_clips=len(clips), planning_train_overlap=0,
                 manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), decoded=decoded,
                 native_sampler_unseeded=True, validation_is_independent=False,
                 learner_inputs=['RGB', 'executed_pose_deltas'], reward_used=False,
                 proprio_encoder_used=False, privileged_planning_labels=False)
    dump(root / 'audit' / host / 'data.json', value)
    return value


def restore_audit(root, host, arm, run):
    import torch
    from omegaconf import OmegaConf
    from data import build_hdf5_dataset, Compose, get_column_normalizer
    from main_hjepa import _normalize_uint8_images_on_device, _build_hjepa_optimizer_factory
    from train import digest_state

    cfg = OmegaConf.load(run / 'config.yaml')
    name = cfg.output_model_name
    model = torch.load(run / f'{name}_object.ckpt', map_location='cpu', weights_only=False)
    checkpoint = torch.load(run / 'lightning_resume/last.ckpt', map_location='cpu', weights_only=False)
    state = {k[len('model.'):]: v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
    actual = model.state_dict()
    assert set(state) == set(actual)
    assert all(torch.equal(v, actual[k]) for k, v in state.items()), 'Native object differs from full checkpoint'
    model.load_state_dict(state, strict=True)
    candidate = arm in ['temporal-forecast', 'innovation']
    if candidate:
        render, other = [], []
        for key, param in model.named_parameters():
            if param.requires_grad:
                (render if key.startswith('renderer.') else other).append(param)
        optimizer = torch.optim.AdamW([
            dict(params=other, lr=5e-5, weight_decay=1e-3, betas=(.9, .999)),
            dict(params=render, lr=1e-4, weight_decay=0., betas=(.9, .95))], foreach=True)
    else:
        optimizer = _build_hjepa_optimizer_factory(model, cfg)(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_states'][0])
    associated = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert associated == {id(p) for p in model.parameters() if p.requires_grad}
    assert optimizer.state and checkpoint['lr_schedulers']
    assert int(checkpoint['global_step']) >= 3
    for value in optimizer.state.values():
        for v in value.values():
            if torch.is_tensor(v):
                assert torch.isfinite(v).all()
    # Real native training windows, same action transform and random-window semantics.
    ds = build_hdf5_dataset(cfg.data.dataset)
    ds.transform = Compose(*[get_column_normalizer(ds, key, key) for key in cfg.data.dataset.keys_to_load
                             if not key.startswith('pixels')])
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)))
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    model = model.cuda().train()
    if candidate:
        from vforge.ops import OpsConfig, initialize_ops
        initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False),
                       device=torch.device('cuda'), parallel_state=None)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = model.loss(candidate_batch(batch), int(checkpoint['global_step']), 0)
    else:
        from hjepa_utils import create_world_model, hjepa_forward
        _, losses = create_world_model(cfg)
        class Module(torch.nn.Module):
            def log_dict(self, *a, **kw):
                pass
        module = Module()
        module.model = model
        for key, value in losses.items():
            setattr(module, key, value.cuda())
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = hjepa_forward(module, batch, 'audit', cfg,
                                  normalize_batch=_normalize_uint8_images_on_device)['loss']
    assert torch.isfinite(loss)
    loss.backward()
    gradient_norm = float(torch.stack([p.grad.float().square().sum() for p in model.parameters()
                                      if p.grad is not None]).sum().sqrt())
    assert gradient_norm > 0 and __import__('math').isfinite(gradient_norm)
    # Public get_cost must propagate gradients to every planned action including the last one.
    model.eval().requires_grad_(False)
    adapter = cost_adapter(model) if candidate else model.get_level(1)
    from droid_data import DROIDClipReader
    from droid_eval import img_transform
    manifest = root / 'public/h_jepa/droid_assets/droid_clips_waypoint_curated16v2_5fps_gw36.json'
    rows = json.loads(manifest.read_text())
    reader = DROIDClipReader(None, ['left_mp4_path'], rows[0]['num_frames'], rows[0]['fps'],
                            frozen_clips=str(manifest), deterministic_getitem=True)
    obs, _, _, _, _ = reader[0]
    tf = img_transform(256)
    info = dict(pixels=tf(obs['visual'][0])[None, None, None].cuda(),
                goal=tf(obs['visual'][-1])[None, None, None].cuda(),
                action=torch.zeros(1, 1, 1, 7, device='cuda'))
    actions = torch.zeros(1, 2, 36, 7, device='cuda', requires_grad=True)
    cost = adapter.get_cost(dict(info), actions)
    gradient = torch.autograd.grad(cost.sum(), actions)[0]
    assert torch.isfinite(cost).all() and torch.isfinite(gradient).all()
    temporal_norm = gradient.abs().sum((0, 1, 3))
    assert (temporal_norm > 0).all(), temporal_norm
    value = dict(passed=True, arm=arm, checkpoint_step=int(checkpoint['global_step']),
                 cpu_model_exact=True, optimizer_associations_exact=True,
                 optimizer_state_entries=len(optimizer.state), scheduler_states=len(checkpoint['lr_schedulers']),
                 real_gpu_backward_loss=float(loss.detach()), gradient_norm=gradient_norm,
                 planning_actions=36, all_actions_have_gradient=True,
                 action_gradient_norms=temporal_norm.detach().cpu().tolist(),
                 note='Audit backward performs no optimizer step; native random-window sampler is not bitwise resumable.')
    dump(root / 'audit' / host / (arm + '-restore.json'), value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['root', 'host']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--arm')
    parser.add_argument('--run')
    args = parser.parse_args()
    install_paths(args.root)
    if args.arm:
        print(restore_audit(Path(args.root), args.host, args.arm, Path(args.run)))
    else:
        print(data_audit(Path(args.root), args.host))
