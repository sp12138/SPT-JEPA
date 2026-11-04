"""Invoke the public trainer, replacing only model/loss/optimizer for our arms."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types

from bridge import install_paths, build_candidate, candidate_batch, candidate_config


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def digest_state(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def run(args):
    root = Path(args.root)
    install_paths(root)
    import torch
    import lightning as pl
    import main_hjepa as main
    from omegaconf import OmegaConf
    from lightning.pytorch.callbacks import Callback

    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(4)
    arm = args.arm
    output = root / 'training' / f'{arm}-seed{args.seed}'
    if args.preflight:
        output = root / args.preflight_tag / args.host / f'{arm}-seed{args.seed}'
    cfgname = 'droid_hjepa_l2' if arm == 'h-jepa' else 'droid_lewm'

    class Audit(Callback):
        def on_fit_start(self, trainer, module):
            self.initial_time = time.time()
            self.latest = {}
            if arm in ['temporal-forecast', 'innovation']:
                from vforge.ops import OpsConfig, initialize_ops
                initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False),
                               device=module.device, parallel_state=None)
            if trainer.is_global_zero:
                path = output / 'initial-state.pt'
                if not path.exists():
                    torch.save(module.model.state_dict(), path)
                dump(output / 'entry.json', dict(arm=arm, seed=args.seed,
                     random_initialization=True, external_pretrained=False,
                     parameters=sum(p.numel() for p in module.model.parameters()),
                     initial_state_sha256=digest_state(module.model),
                     global_batch=256 if not args.preflight else 4,
                     local_batch=128 if not args.preflight else 2,
                     devices=trainer.world_size, gpu=torch.cuda.get_device_name(),
                     public_config=cfgname, candidate_config=candidate_config(args.seed)
                     if arm in ['temporal-forecast', 'innovation'] else None,
                     precision=str(trainer.precision), preflight=args.preflight))

        def on_before_optimizer_step(self, trainer, module, optimizer):
            step = int(trainer.global_step)
            if step not in [0, 1, 2] and (step + 1) % 100:
                return
            prefixes = (['encoder', 'pool', 'predictor', 'action_encoder', 'renderer']
                        if arm in ['temporal-forecast', 'innovation'] else
                        [f'jepas.{i}.{part}' for i in range(2 if arm == 'h-jepa' else 1)
                         for part in ['encoder', 'predictor', 'idm_loss_fn']])
            gradients = {}
            for prefix in prefixes:
                values = [p.grad.detach().float().square().sum()
                          for n, p in module.model.named_parameters()
                          if n.startswith(prefix + '.') and p.grad is not None]
                norm = float(torch.stack(values).sum().sqrt()) if values else 0.
                assert math.isfinite(norm), (step, prefix, norm)
                if step >= 2:
                    assert norm > 0, (step, prefix, 'missing gradient')
                gradients[prefix] = norm
            self.latest = dict(gradient_step=step, gradients=gradients)

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            step = int(trainer.global_step)
            if step <= 3 or step % 100 == 0:
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs
                if torch.is_tensor(loss):
                    loss = float(loss.detach())
                    assert math.isfinite(loss)
                dump(output / f'health-rank{trainer.global_rank}.json', dict(
                    step=step, epoch=int(trainer.current_epoch), loss=loss,
                    **self.latest, world_size=trainer.world_size,
                    utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    elapsed=time.time() - self.initial_time,
                    lr=[g['lr'] for g in trainer.optimizers[0].param_groups],
                    peak_gpu_bytes=torch.cuda.max_memory_allocated()))

        def on_train_end(self, trainer, module):
            if trainer.is_global_zero:
                dump(output / 'training-complete.json', dict(
                    passed=True, step=int(trainer.global_step), epochs=int(trainer.current_epoch),
                    model_state_sha256=digest_state(module.model), preflight=args.preflight))

        def on_train_epoch_end(self, trainer, module):
            epoch = int(trainer.current_epoch) + 1
            if not trainer.is_global_zero or epoch not in [10, 100] or args.preflight:
                return
            # Audit is appended after the public object writer: evaluators never
            # open an object checkpoint until its writer has returned.
            checkpoint = output / f'{module._output_model_name}_epoch_{epoch}_object.ckpt'
            assert checkpoint.is_file()
            dump(output / f'epoch{epoch}-ready.json', dict(passed=True, epoch=epoch,
                 global_step=int(trainer.global_step), checkpoint=str(checkpoint),
                 bytes=checkpoint.stat().st_size, model_state_sha256=digest_state(module.model)))

    original_trainer = main.pl.Trainer

    def audited_trainer(*a, **kwargs):
        kwargs['callbacks'].append(Audit())
        return original_trainer(*a, **kwargs)

    main.pl = types.SimpleNamespace(Trainer=audited_trainer, pytorch=pl.pytorch)
    if arm in ['temporal-forecast', 'innovation']:
        main.create_world_model = lambda cfg: (build_candidate(arm, args.seed), {})

        def forward(module, batch, stage, cfg):
            loss, metrics = module.model.loss(candidate_batch(batch), int(module.global_step), 0)
            assert torch.isfinite(loss), (stage, int(module.global_step))
            module.log_dict({f'{stage}/{k}': v.detach() for k, v in metrics.items()},
                            on_step=True, sync_dist=True, batch_size=batch['pixels_level1'].shape[0])
            return dict(loss=loss)

        main._hjepa_training_forward = forward
        # This placeholder is constructed by main; CandidateModule owns the real optimizers.
        main._build_hjepa_optimizer_factory = lambda model, cfg: lambda p: torch.optim.AdamW(p)
        base_module = main.GradClipModule

        class CandidateModule(base_module):
            def configure_optimizers(self):
                render, other = [], []
                for name, parameter in self.model.named_parameters():
                    if parameter.requires_grad:
                        (render if name.startswith('renderer.') else other).append(parameter)
                optimizer = torch.optim.AdamW([
                    dict(params=other, lr=5e-5, weight_decay=1e-3, betas=(.9, .999)),
                    dict(params=render, lr=1e-4, weight_decay=0., betas=(.9, .95))], foreach=True)
                # The existing temporal-forecast schedule, fixed at the full public budget.
                total = 29200
                warmup = 292

                def schedule(step):
                    if step < warmup:
                        return max(1, step) / warmup
                    return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / (total - warmup))))

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
                return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='step'))

        main.GradClipModule = CandidateModule

    output.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], '--config-name', cfgname, f'seed={args.seed}', f'subdir={output}',
                'trainer.devices=2', 'trainer.max_epochs=100',
                '+trainer.strategy=ddp_find_unused_parameters_true',
                '+trainer.enable_progress_bar=false', '+trainer.log_every_n_steps=20',
                'loader.num_workers=6', 'save_every_n_epochs=10', 'resume_every_n_steps=100']
    if args.preflight:
        sys.argv += ['+trainer.max_steps=3', 'trainer.limit_train_batches=3',
                     'loader.batch_size=2', 'loader.num_workers=0', 'loader.persistent_workers=false',
                     '~loader.prefetch_factor', 'resume_every_n_steps=1', '+trainer.limit_val_batches=1']
    from hydra import initialize_config_dir, compose
    with initialize_config_dir(config_dir=str(root / 'public/h_jepa/config/train'), version_base=None):
        cfg = compose(config_name=cfgname, overrides=sys.argv[3:])
    main.run.__wrapped__(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['root', 'host', 'arm']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--preflight-tag', default='preflight-v7')
    run(parser.parse_args())
