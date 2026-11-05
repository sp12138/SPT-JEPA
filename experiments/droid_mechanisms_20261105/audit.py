"""Real full-state restore/backward gates and initialization pairing contracts."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from common import arm_root, bootstrap, run_dir


def optimizer_for(model):
    import torch
    render, other = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (render if name.startswith('renderer.') else other).append(param)
    return torch.optim.AdamW([
        dict(params=other, lr=5e-5, weight_decay=1e-3, betas=(.9, .999)),
        dict(params=render, lr=1e-4, weight_decay=0., betas=(.9, .95))], foreach=True)


def mean_hash(model):
    import torch
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('idm.'):
            continue
        if name in ['pool.head.weight', 'pool.head.bias']:
            value = value[:model.state_width]
        if name in ['predictor.transformer.output_proj.weight', 'predictor.transformer.output_proj.bias']:
            value = value[:model.state_dim]
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def pairing(args):
    root = arm_root(args.root, 'tf'); dump = bootstrap(root)
    import torch
    from as_jepa_lewm.droid_mechanisms import ARMS, build
    from bridge import build_candidate
    torch.set_num_threads(3)
    rows = []
    for seed in [1, 1000]:
        hashes = {}
        for arm in ARMS:
            torch.manual_seed(seed)
            model = build(arm, seed)
            hashes[arm] = mean_hash(model)
            assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for name,m in model.named_modules()
                           if not name.startswith('sigreg')), arm
            del model
        assert len(set(hashes.values())) == 1, hashes
        torch.manual_seed(seed)
        original = build_candidate('temporal-forecast', seed)
        assert mean_hash(original) == hashes['tf']
        rows.append(dict(seed=seed, hashes=hashes, original_tf_mean_exact=True))
    dump(Path(args.root)/'contracts/pairing.json', dict(passed=True, rows=rows))
    print(json.dumps(dict(passed=True, pairing_seeds=2, arms=7)), flush=True)


def restore(args):
    root = arm_root(args.root, args.arm); dump = bootstrap(root)
    import torch
    from bridge import candidate_batch, cost_adapter
    from omegaconf import OmegaConf
    from data import build_hdf5_dataset, Compose, get_column_normalizer
    from vforge.ops import OpsConfig, initialize_ops
    from train import digest_state
    torch.set_num_threads(3)
    directory = Path(args.run) if args.run else run_dir(args.root, args.arm, args.seed)
    ready = json.loads((directory/f'{args.stage}-ready.json').read_text())
    model = torch.load(ready['checkpoint'], map_location='cpu', weights_only=False)
    full = torch.load(directory/'lightning_resume/last.ckpt', map_location='cpu', weights_only=False)
    state = {k[len('model.'):]:v for k,v in full['state_dict'].items() if k.startswith('model.')}
    assert int(full['global_step']) == ready['step']
    assert set(state) == set(model.state_dict())
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in state.items())
    assert digest_state(model) == ready['model_state_sha256']
    optimizer = optimizer_for(model); optimizer.load_state_dict(full['optimizer_states'][0])
    assert optimizer.state and full['lr_schedulers']
    for group in optimizer.state.values():
        for value in group.values():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all()
    cfg = OmegaConf.load(directory/'config.yaml')
    ds = build_hdf5_dataset(cfg.data.dataset)
    ds.transform = Compose(*[get_column_normalizer(ds,key,key) for key in cfg.data.dataset.keys_to_load if not key.startswith('pixels')])
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)))
    batch = {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=torch.device('cuda'), parallel_state=None)
    model = model.cuda().train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, metrics = model.loss(candidate_batch(batch), ready['step'], 0)
    assert torch.isfinite(loss)
    loss.backward()
    prefixes = ['encoder','pool','predictor','action_encoder','renderer'] + (['idm'] if args.arm.endswith('idm') else [])
    grads = {}
    for prefix in prefixes:
        norm = sum(float(p.grad.float().square().sum()) for k,p in model.named_parameters() if k.startswith(prefix+'.') and p.grad is not None)
        assert math.isfinite(norm) and norm > 0, (prefix,norm)
        grads[prefix] = norm**.5
    value = dict(passed=True, arm=args.arm, seed=args.seed, stage=args.stage, step=ready['step'],
                 model_full_checkpoint_exact=True, optimizer_state_entries=len(optimizer.state),
                 real_gpu_backward=True, loss=float(loss), gradients=grads,
                 metrics={k:float(v.detach()) for k,v in metrics.items()}, optimizer_step_performed=False)
    dump(directory/f'{args.stage}-audit.json', value)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['pairing','restore']); p.add_argument('--root',required=True)
    p.add_argument('--arm'); p.add_argument('--seed',type=int); p.add_argument('--stage'); p.add_argument('--run')
    a=p.parse_args(); pairing(a) if a.mode=='pairing' else restore(a)
