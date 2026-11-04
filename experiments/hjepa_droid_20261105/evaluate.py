"""One public DROID clip; GT actions are accessed only by the released scorer."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from bridge import install_paths, cost_adapter


def state_hash(model):
    import torch
    result = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        result.update(key.encode())
        result.update(value.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def main(args):
    install_paths(args.root)
    import torch
    from omegaconf import OmegaConf
    from droid_eval import run_clip_eval, clip_metrics

    torch.set_num_threads(3)
    torch.set_float32_matmul_precision('high')
    candidate = args.arm in ['temporal-forecast', 'innovation']
    root = Path(args.root)
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    else:
        model_name = {'lewm': 'droid_lewm', 'h-jepa': 'droid_hjepa_l2', 'hwm': 'droid_hwm_l2'}[args.arm]
        checkpoint = (root.parent.parent / 'data/hjepa-public-models/droid' / model_name /
                      f'seed{args.model_seed}' / f'{model_name}_object.ckpt')
    assert checkpoint.is_file(), checkpoint
    core = torch.load(checkpoint, map_location='cpu', weights_only=False).eval()
    before = state_hash(core)
    if args.checkpoint and '/training/' in str(checkpoint):
        epoch = int(checkpoint.stem.split('_epoch_')[1].split('_')[0])
        ready = json.loads((checkpoint.parent / f'epoch{epoch}-ready.json').read_text())
        assert ready['passed'] and ready['model_state_sha256'] == before
    core.requires_grad_(False)
    if candidate:
        from vforge.ops import OpsConfig, initialize_ops
        initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False),
                       device=torch.device('cuda'), parallel_state=None)
        model = cost_adapter(core).cuda()
    else:
        model = core.cuda()
    hierarchical = args.arm in ['h-jepa', 'hwm']
    cfg = OmegaConf.load(root / 'public/h_jepa/config/eval' /
                         ('droid_l2.yaml' if hierarchical else 'droid_flat.yaml'))
    cfg.policy = str(checkpoint)
    cfg.seed = int(args.planner_seed)
    cfg.output.dir = str(Path(args.output).resolve())
    cfg.start_index = int(args.clip)
    cfg.num_eval = 1
    if hierarchical and args.arm == 'h-jepa':
        cfg.solver.solvers.level2.optimizer_kwargs.lr = .01
    start = time.time()
    run_clip_eval(cfg, model=model, results_dir=args.output)
    after = state_hash(core)
    assert before == after, 'Evaluation changed learned tensors/buffers'
    ep = torch.load(Path(args.output) / f'ep_{args.clip}/actions.pt', weights_only=False)
    assert ep['planned'].shape == ep['gt'].shape == (36, 7)
    assert torch.isfinite(ep['planned']).all() and torch.isfinite(ep['gt']).all()
    result = dict(passed=True, arm=args.arm, model_seed=args.model_seed,
                  planner_seed=args.planner_seed, clip=args.clip,
                  model_sha256=before, state_unchanged=True, elapsed=time.time() - start,
                  checkpoint=str(checkpoint), metrics=clip_metrics(**ep))
    (Path(args.output) / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['root', 'arm', 'output']:
        parser.add_argument('--' + name, required=True)
    for name in ['model-seed', 'planner-seed', 'clip']:
        parser.add_argument('--' + name, required=True, type=int)
    parser.add_argument('--checkpoint')
    main(parser.parse_args())
