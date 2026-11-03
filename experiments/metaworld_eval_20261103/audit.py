"""Actual native checkpoint restore and read-only held-out backward check."""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.ideas_validate_20261031.preflight import checkpoint_report, finite, gradient_report
from experiments.ideas_validate_20261031.audit import expected_gradients, file_hash, move_batch


def equal(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def main(run, step, output):
    from as_jepa_lewm.train import TrainModule
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    from vforge.ops import OpsConfig, initialize_ops
    root = Path(run); cfg = json.loads((root / 'config.json').read_text())
    torch.set_num_threads(2)
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False),
                   device=torch.device('cuda'), parallel_state=None)
    checkpoint = torch.load(root / 'last.ckpt', map_location='cpu', weights_only=False)
    portable_path = root / f'model-step{step:07d}' / 'weights.pt'
    portable = torch.load(portable_path, map_location='cpu', weights_only=True)
    report = dict(run=str(root), step=step, config_sha256=file_hash(root / 'config.json'),
                  checkpoint_sha256=file_hash(root / 'last.ckpt'),
                  portable_sha256=file_hash(portable_path),
                  checkpoint=checkpoint_report(checkpoint, portable, cfg, step))
    module = TrainModule(cfg).cpu()
    module.load_state_dict(checkpoint['state_dict'], strict=True)
    configured = module.configure_optimizers()
    optimizer = configured['optimizer']; scheduler = configured['lr_scheduler']['scheduler']
    optimizer.load_state_dict(checkpoint['optimizer_states'][0])
    scheduler.load_state_dict(checkpoint['lr_schedulers'][0])
    report['actual_cpu_restore'] = dict(
        model=equal(module.state_dict(), checkpoint['state_dict']),
        optimizer=equal(optimizer.state_dict(), checkpoint['optimizer_states'][0]),
        scheduler=equal(scheduler.state_dict(), checkpoint['lr_schedulers'][0]))
    initial = torch.load(root / 'initial-model/weights.pt', map_location='cpu', weights_only=True)
    report['updated_tensors'] = sum(not torch.equal(value, portable[key]) for key,value in initial.items())
    del initial, portable, checkpoint, optimizer, scheduler, configured
    model = module.model.cuda().eval()
    dataset = BenchmarkWindows(cfg['data'], window=model.window, seed=cfg['seed'], split='test')
    train_ids = set(np.load(Path(cfg['data']) / 'train_episodes.npy').tolist())
    test_ids = set(np.load(Path(cfg['data']) / 'test_episodes.npy').tolist())
    assert not {e // 5 for e in train_ids} & {e // 5 for e in test_ids}
    rng = np.random.default_rng(20261103 + cfg['seed']); selected=[]; originals=set()
    while len(selected) < 2:
        index = int(rng.integers(dataset.natural_length)); ep,start = dataset.locate(index)
        original = int(dataset.episode_ids[ep]) // 5
        if original in originals:continue
        originals.add(original); selected.append(index)
    batch = next(iter(DataLoader(Subset(dataset, selected), batch_size=2, num_workers=0)))
    report['heldout_windows'] = dict(split='test', indices=selected, original_episodes=sorted(originals),
                                     episode=batch['episode'].tolist(), rows=batch['rows'].tolist())
    buffers = {key:value.detach().cpu().clone() for key,value in model.named_buffers()}
    torch.manual_seed(20261103 + cfg['seed']); model.zero_grad(set_to_none=True)
    loss, metrics = model.loss(move_batch(batch, 'cuda'), step=step, ramp_steps=0)
    loss.backward()
    report.update(loss=float(loss.detach()), loss_finite=finite(loss),
                  metrics={k:float(v.detach()) if torch.is_tensor(v) else v for k,v in metrics.items()},
                  gradients=gradient_report(model, expected_gradients(model, cfg)),
                  all_parameter_gradients_finite=all(finite(p.grad) for p in model.parameters() if p.grad is not None),
                  eval_buffers_unchanged=all(torch.equal(buffers[k], v.detach().cpu()) for k,v in model.named_buffers()))
    report['passed'] = (all(report['checkpoint']['checks'].values())
                        and all(report['actual_cpu_restore'].values())
                        and report['updated_tensors'] > 0 and report['loss_finite']
                        and finite(report['metrics']) and report['all_parameter_gradients_finite']
                        and report['eval_buffers_unchanged']
                        and all(row['passed'] for row in report['gradients'].values()))
    report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['scope'] = 'Native full model/AdamW/scheduler CPU restores; actual held-out FP32 backward without optimizer step. No physical/optimal-action labels. Positive gradients are interface evidence,not task performance.'
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=report['passed'], step=step, loss=report['loss'], restore=report['actual_cpu_restore'], gradients=report['gradients'])), flush=True)
    if not report['passed']:raise RuntimeError('Native checkpoint/backward audit failed; see ' + str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', required=True)
    parser.add_argument('--step', type=int, required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.run, args.step, args.output)
