"""Configure the existing, pinned native trainer for a bounded mechanism arm."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import types

from common import OLD, arm_root, bootstrap


def run(args):
    root = arm_root(args.root, args.arm)
    dump = bootstrap(root)
    # The old module has the same basename; import it by its absolute filename.
    import importlib.util
    spec = importlib.util.spec_from_file_location('droid_original_trainer', OLD / 'helper-v8/train.py')
    original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    import torch
    import numpy as np
    import lightning as pl
    import main_hjepa as main
    from lightning.pytorch.callbacks import Callback
    from as_jepa_lewm.droid_mechanisms import build
    from droid_data import DROIDClipReader

    # Pair the previously entropy-seeded window stream across arms. Frozen eval
    # clips and the public bounded retry policy retain their original semantics.
    original_rng = DROIDClipReader._rng

    def paired_rng(reader, idx):
        if reader.deterministic_getitem or reader.frozen is not None:
            return original_rng(reader, idx)
        if not hasattr(reader, '_mechanism_rng'):
            worker = torch.utils.data.get_worker_info()
            reader._mechanism_rng = np.random.default_rng(np.random.SeedSequence(
                [args.seed, int(__import__('os').environ.get('RANK', 0)), 0 if worker is None else worker.id, 20261105]))
        return reader._mechanism_rng

    DROIDClipReader._rng = paired_rng
    original.build_candidate = lambda arm, seed: build(args.arm, seed)
    original_dump = dump
    output = root / 'training' / f'temporal-forecast-seed{args.seed}'
    if args.stage == 'preflight':
        output = root / args.tag / args.host / f'temporal-forecast-seed{args.seed}'

    def metadata(path, value):
        path = Path(path)
        if path.name == 'entry.json':
            value.update(mechanism=args.arm, sample_level=True, factual_action_idm=args.arm.endswith('idm'),
                         candidate_origin='random initialization within this round', paired_video_sampler=True)
            if path.exists():
                value.update(random_initialization=False, restored_state_sha256=value['initial_state_sha256'])
                value['initial_state_sha256'] = json.loads(path.read_text())['initial_state_sha256']
                path = output / f'entry-{args.stage}-resume.json'
        try:
            original_dump(path, value)
        except OSError:
            if not path.name.startswith('health-rank'):
                raise
            fallback = Path('/tmp/droid-mechanisms-health') / args.arm / str(args.seed) / path.name
            original_dump(fallback, value)
    original.dump = metadata

    class MechanismAudit(Callback):
        def on_fit_start(self, trainer, module):
            torch.manual_seed(args.seed + trainer.global_rank * 100003)
            torch.cuda.manual_seed_all(args.seed + trainer.global_rank * 100003)

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            step = int(trainer.global_step)
            if step not in [0, 1, 2, 600, 601]:
                return
            hashes = {}
            for key in ['pixels_level1', 'action_level1']:
                hashes[key] = hashlib.sha256(batch[key].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            dump(output / f'batch-rank{trainer.global_rank}-step{step}.json', dict(step=step, hashes=hashes))

        def on_before_optimizer_step(self, trainer, module, optimizer):
            step = int(trainer.global_step)
            if args.arm.endswith('idm') and (step < 3 or (step+1) % 100 == 0):
                norm = sum(float(p.grad.float().square().sum()) for p in module.model.idm.parameters() if p.grad is not None)
                assert math.isfinite(norm) and norm > 0, ('IDM gradient', step, norm)

        def on_train_end(self, trainer, module):
            if not trainer.is_global_zero:
                return
            path = output / 'droid_lewm_object.ckpt'
            assert path.is_file()
            dump(output / f'{args.stage}-ready.json', dict(passed=True, stage=args.stage, arm=args.arm,
                 seed=args.seed, step=int(trainer.global_step), checkpoint=str(path),
                 model_state_sha256=original.digest_state(module.model)))

    native_trainer = pl.Trainer

    def bounded_trainer(*a, **kwargs):
        kwargs['max_steps'] = dict(preflight=3, health=600, screen=5840, long=29200)[args.stage]
        kwargs['callbacks'].append(MechanismAudit())
        return native_trainer(*a, **kwargs)

    main.pl = types.SimpleNamespace(Trainer=bounded_trainer, pytorch=pl.pytorch)
    original.run(argparse.Namespace(root=str(root), host=args.host, arm='temporal-forecast', seed=args.seed,
                                    preflight=args.stage == 'preflight', preflight_tag=args.tag))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['root', 'host', 'arm', 'stage']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--tag', default='preflight-v1')
    run(p.parse_args())
