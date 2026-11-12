"""Four-host, fixed-wall-time comparison with fresh world-model heads.

Frozen RGB features are identical across methods. RoboJEPA uses its public
300M predictor, latent L1 and detached K2 objective; SPT reuses global16.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.robocasa_compare import train as common
from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_refine import encoder_ablation as spt


class Native(nn.Module):
    def __init__(self, root, cfg):
        super().__init__()
        from experiments.robocasa_compare.evaluate import install_public
        install_public(root)
        from app.robo_jepa.world_model import WorldModel
        config = json.loads((root / "sources/robo_jepa/configs/released/wm-300m/config.json").read_text())
        config.pop("encoder_version", None)  # Packaging metadata, not a predictor argument.
        config.update(grid_height=14, grid_width=14, attn_impl="sdpa",
                      use_activation_checkpointing=True)
        torch.manual_seed(61001 + cfg["seed"])
        self.wm = WorldModel(config, train_config=dict(rollout_steps=2,
            predictor_window_steps=1, eval_rollout_steps=2,
            train_rollout_prefixes="random", rollout_stop_gradient=True))
        self.predictor = self.wm.transition_model

    def predict(self, features, actions):
        b, t = features.shape[:2]
        states = features.new_zeros(b, t, 64)
        return self.wm.forward(features, F.pad(actions, (0, 50)), states=states,
            view_map=torch.tensor([[0, 2]], device=features.device).expand(b, -1),
            embodiment_ids=torch.full((b, 1), 2, device=features.device, dtype=torch.long))[0]

    def loss(self, batch, step=0, prefix=None):
        z = batch["features"].permute(0, 2, 1, 3, 4).contiguous()
        b, t, v, p, d = z.shape
        z = z.reshape(b, t, v, 14, 14, d)
        a = batch["action"]
        predicted = self.predict(z, a)
        mask = z.new_ones(b, t)
        tf = self.wm.compute_loss(predicted, z, None, None, mask, shift=1)["loss"]
        # Public K2: a random true prefix followed by one detached TF prediction,
        # then one further prediction. The public rollout divides its weight
        # by (R+1), giving TF/2 + AR/4 for K2, not an invented equal-weight loss.
        if prefix is None:
            prefix = int(torch.randint(t - 2, (1,)).item()) + 1
        assert 1 <= prefix <= t - 2
        history = torch.cat((z[:, :prefix], predicted[:, prefix-1:prefix].detach()), 1)
        further = self.predict(history, a[:, :prefix+1])[:, -1:]
        ar = self.wm.compute_loss(further, z[:, prefix+1:prefix+2], None, None,
                                   mask[:, :1])["loss"]
        loss = tf * .5 + ar * .25
        return loss, dict(loss=loss, tf=tf, ar=ar, prefix=loss.new_tensor(prefix))


def build(root, cfg):
    common.install_runtime(root / "code")
    if cfg["method"] == "robojepa":
        return Native(root, cfg)
    return spt.build(root, cfg)


def prepare(raw, cfg, device):
    if cfg["method"] == "spt":
        return spt.prepare(raw, device)
    return {k: raw[k].to(device, non_blocking=True) for k in ("features", "action")}


def optimizer(model, cfg):
    if cfg["method"] == "spt":
        opt, _ = common.make_optimizer(model, 100)
    else:
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            (no_decay if "bias" in n or p.ndim == 1 else decay).append(p)
        opt = torch.optim.AdamW([dict(params=decay, weight_decay=.1),
                                dict(params=no_decay, weight_decay=0.)],
                               lr=5e-5, betas=(.9, .95), eps=1e-8, foreach=True)
    for g in opt.param_groups:
        g["peak_lr"] = 1e-4 if cfg["method"] == "spt" and g is opt.param_groups[-1] else 5e-5
    return opt


def set_rates(opt, elapsed, total):
    if total is None:
        # The user authorized an open-ended run. No arbitrary terminal cosine
        # anneal: both methods get the same fixed15-minute warmup, then flat LR.
        factor = min(1., max(1e-5, elapsed / 900.))
        for group in opt.param_groups:
            group["lr"] = group["peak_lr"] * factor
        return factor
    fraction = min(1., elapsed / total)
    factor = max(1e-5, fraction / .01) if fraction < .01 else .5 * (
        1 + math.cos(math.pi * (fraction - .01) / .99))
    for group in opt.param_groups:
        group["lr"] = group["peak_lr"] * factor
    return factor


def prefixes(cfg):
    return spt.gradient_prefixes(cfg) if cfg["method"] == "spt" else (
        "predictor.predictor_blocks.", "predictor.predictor_embed.", "predictor.predictor_proj.")


def checkpoint(root, run, cfg, model, opt, step, elapsed, rank, world, label=None):
    import torch.distributed as dist
    states = [None] * world
    dist.all_gather_object(states, common.rng_state())
    if rank == 0:
        state = dict(model=common.cpu(model.state_dict()), optimizer=common.cpu(opt.state_dict()),
            scheduler=dict(kind="wall-warmup-flat" if cfg["budget_seconds"] is None else "wall-cosine",
                           elapsed=elapsed, total=cfg["budget_seconds"], warmup_seconds=900.),
            rank_rng=states, step=step, samples=step*cfg["global_batch"],
            active_seconds=elapsed, world_size=world,
            parameter_associations=common.associations(model, opt),
            config_sha256=sha(run/"config.json"), nn_sha256=common.state_hash(model.state_dict()),
            cache_manifest_sha256=cfg["cache_manifest_sha256"])
        common.save_full_state(run, state)
        if label:
            target = run/label
            target.mkdir(exist_ok=False)
            torch.save(state["model"], target/"weights.pt")
            save(target/"ready.json", dict(passed=True, label=label, step=step,
                active_seconds=elapsed, nn_sha256=state["nn_sha256"],
                weights_sha256=sha(target/"weights.pt")))
    dist.barrier()


def train(root, run, local_data, target_seconds, entry=False, resume=False, fp32_retry=False):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local); device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    torch.set_num_threads(2); torch.set_float32_matmul_precision("high")
    cfg = json.loads((run/"config.json").read_text())
    assert world == 8 and (cfg["budget_seconds"] is None or target_seconds <= cfg["budget_seconds"])
    ready = json.loads((root/f"STAGE_READY-{cfg['host']}.json").read_text())
    assert ready["passed"] and ready["manifest_sha256"] == cfg["cache_manifest_sha256"]
    model = build(root, cfg).to(device)
    from vforge.ops import OpsConfig, initialize_ops
    initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False), device=device, parallel_state=None)
    opt = optimizer(model, cfg)
    step = 0; elapsed = 0.; restored = None
    initial = common.state_hash(model.state_dict())
    hashes = [None] * world; dist.all_gather_object(hashes, initial)
    assert len(set(hashes)) == 1
    if resume:
        full = torch.load(run/"last.ckpt", map_location="cpu", weights_only=False)
        assert full["config_sha256"] == sha(run/"config.json") and full["world_size"] == world
        assert full["parameter_associations"] == common.associations(model, opt)
        model.load_state_dict(full["model"], strict=True)
        opt.load_state_dict(full["optimizer"])
        assert common.checkpoint_equal(full["optimizer"], opt.state_dict())
        assert common.state_hash(model.state_dict()) == full["nn_sha256"]
        step = full["step"]; elapsed = full["active_seconds"]
        assert full["samples"] == step*cfg["global_batch"]
        assert full["scheduler"]["elapsed"] == elapsed
        restored = full["rank_rng"][rank]
        if rank == 0:
            save(run/f"restore-{step:08d}.json", dict(passed=True, step=step,
                active_seconds=elapsed, model_optimizer_scheduler_rankRNG_cursor_exact=True,
                world_size=world, replayed_saved_updates=0))
        del full
    elif rank == 0:
        save(run/"initial.json", dict(passed=True, fresh_heads=True, seed=cfg["seed"],
            parameters=sum(p.numel() for p in model.parameters()), nn_sha256=initial,
            all_rank_initial_hashes_identical=True, external_dynamics_weights=False))
    wrapped = DDP(common.LossModel(model), device_ids=[local], broadcast_buffers=False,
                  find_unused_parameters=True)
    data = spt.FeatureWindows(local_data, cfg)
    loader = DataLoader(data, batch_sampler=common.CounterBatches(step, 100000000,
        cfg["global_batch"], rank, world), num_workers=2, pin_memory=True, prefetch_factor=2,
        generator=torch.Generator().manual_seed(91000+cfg["seed"]+rank))
    iterator = iter(loader)
    if restored:
        common.restore_rng(restored)
        assert common.checkpoint_equal(common.rng_state(), restored)
    else:
        seed = 98000+cfg["seed"]+rank
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if rank == 0:
        save(run/"ACTIVE.json", dict(pid=os.getpid(), world_size=world, restored=resume,
            target_seconds=target_seconds, entry=entry, started_unix=time.time()))
    model.train(); began = time.time(); segment_updates = 0; checkpoint_elapsed = elapsed
    while elapsed < target_seconds:
        assert not (root/"STOP").exists()
        tick = time.perf_counter()
        raw = next(iterator); batch = prepare(raw, cfg, device)
        opt.zero_grad(set_to_none=True)
        set_rates(opt, elapsed, cfg["budget_seconds"])
        iteration_rng = common.rng_state() if fp32_retry else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = wrapped(batch, step+1)
        finite_loss = bool(torch.isfinite(loss))
        if fp32_retry:
            loss_flag = torch.tensor(int(finite_loss), device=device)
            dist.all_reduce(loss_flag, op=dist.ReduceOp.MIN)
            finite_loss = bool(loss_flag)
        if finite_loss: loss.backward()
        if fp32_retry:
            finite = finite_loss and all(bool(torch.isfinite(p.grad).all())
                for p in model.parameters() if p.grad is not None)
            flag = torch.tensor(int(finite), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if not bool(flag):
                # Same physical batch and sample cursor; no optimizer update,
                # skipped example or LR change. Preserve the failed AMP attempt.
                opt.zero_grad(set_to_none=True)
                common.restore_rng(iteration_rng)
                with torch.autocast("cuda", enabled=False):
                    loss, metrics = wrapped(batch, step+1)
                assert torch.isfinite(loss), ("FP32 retry loss", step+1, metrics)
                loss.backward()
                finite = all(bool(torch.isfinite(p.grad).all())
                    for p in model.parameters() if p.grad is not None)
                flag.fill_(int(finite)); dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                if rank == 0:
                    save(run/f"numeric-retry-{step+1:08d}.json", dict(step=step+1,
                        same_batch_and_cursor=True, optimizer_updates_before_retry=0,
                        outer_RNG_restored=True, precision="FP32", finite=bool(flag),
                        algorithm_loss_optimizer_LR_unchanged=True))
                assert bool(flag), ("FP32 retry gradients", step+1)
        else:
            assert finite_loss, metrics
        check = step < 8 or step % 100 == 0 or segment_updates < 2
        norms = {}
        if check:
            for n, p in model.named_parameters():
                if p.grad is not None:
                    value = float(p.grad.detach().float().norm())
                    assert math.isfinite(value), n
                    for pre in prefixes(cfg):
                        if n.startswith(pre): norms[pre] = norms.get(pre, 0.) + value*value
            for pre in prefixes(cfg):
                if pre == "action_encoder." and step < 2: continue
                assert norms.get(pre, 0.) > 0, (pre, norms)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step(); torch.cuda.synchronize()
        duration = torch.tensor(time.perf_counter()-tick, device=device, dtype=torch.float64)
        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        elapsed += float(duration); step += 1; segment_updates += 1
        if rank == 0 and (check or step % 20 == 0):
            row = dict(step=step, samples=step*cfg["global_batch"], active_seconds=elapsed,
                allocated_segment_seconds=time.time()-began,
                loss=float(loss.detach()), metrics={k:float(v.detach()) for k,v in metrics.items()},
                gradients={k:math.sqrt(v) for k,v in norms.items()},
                lr=[g["lr"] for g in opt.param_groups], host=cfg["host"], method=cfg["method"])
            save(run/"progress.json", row)
            if step % 100 == 0 or step <= 8:
                with (run/"curve.jsonl").open("a") as stream: stream.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
        final = (entry and segment_updates == 8) or elapsed >= target_seconds
        if final or elapsed-checkpoint_elapsed >= 900:
            label = "entry" if entry else f"hour-{round(target_seconds/3600):02d}" if final else None
            checkpoint(root, run, cfg, model, opt, step, elapsed, rank, world, label)
            checkpoint_elapsed = elapsed
        if final: break
    if rank == 0:
        save(run/f"TRAINED-{'entry' if entry else round(target_seconds/3600)}.json", dict(
            passed=True, step=step, active_seconds=elapsed,
            allocated_segment_seconds=time.time()-began, new_updates=segment_updates,
            full_state_saved=True, completed=cfg["budget_seconds"] is not None and elapsed >= cfg["budget_seconds"]))
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True); p.add_argument("--run", type=Path, required=True)
    p.add_argument("--local-data", type=Path, required=True)
    p.add_argument("--target-seconds", type=float, required=True)
    p.add_argument("--entry", action="store_true"); p.add_argument("--resume", action="store_true")
    p.add_argument("--fp32-retry", action="store_true")
    a = p.parse_args()
    train(a.root, a.run, a.local_data, a.target_seconds, a.entry, a.resume, a.fp32_retry)
