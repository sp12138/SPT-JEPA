"""Paired sample-level consequence learners on an expanding factual corpus."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiments.robocasa_compare import train as common
from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_refine import encoder_ablation as old
from experiments.robocasa_time_compare.train import optimizer


def feature_delta_loss(self, batch, step=0, ramp_steps=0):
    dense = batch["features"].float()
    b, t, patches, width = dense.shape
    assert patches == 196 and width == 1664
    means = self.pool(self.feature_adapter(dense.flatten(0, 1))).reshape(b, t, 64)
    actions = self.action_encoder(batch["action"])
    # Fixed visual features, with the actual first observation as an anchor.
    # There is no learned skip that can bypass the bottleneck with future RGB.
    target = F.adaptive_avg_pool2d(dense.reshape(b*t, 14, 14, width)
        .permute(0, 3, 1, 2), (4, 4)).flatten(1).reshape(b, t, -1).detach()
    prediction = self.predict(means[:, :-1], actions[:, :-1])
    decoded = self.future_decoder(prediction-means[:, :1]).float()
    delta = target[:, 1:]-target[:, :1]
    tf = (decoded-delta).abs().mean()
    prefix = int(torch.randint(t-2, (1,), device=dense.device))+1
    history = torch.cat((means[:, :prefix], prediction[:, prefix-1:prefix].detach()), 1)
    future = self.predict(history, actions[:, :prefix+1])[:, -1:]
    ar = (self.future_decoder(future-means[:, :1]).float()-(target[:, prefix+1:prefix+2]-target[:, :1])).abs().mean()
    loss = tf*.5+ar*.25
    copy = (target[:, 1:]-target[:, :-1]).abs().mean()
    return loss, dict(loss=loss, fixed_feature_TF=tf, fixed_feature_AR=ar,
        fixed_feature_copy_L1=copy, feature_std=means.float().std((0, 1)).mean())


def build(root, cfg):
    base = old.build(root, cfg)
    if cfg["variant"] == "feature-delta":
        base.renderer.requires_grad_(False)
        with torch.random.fork_rng():
            torch.manual_seed(90120261114)
            base.future_decoder = nn.Linear(64, 1664*16, bias=False)
        base.loss = types.MethodType(feature_delta_loss, base)
    else:
        assert cfg["variant"] == "spt-rate"
    return base


class Windows:
    def __init__(self, root, manifest, cfg):
        self.records = json.loads(manifest.read_text())["records"]
        assert self.records and all(r["split"] == "train" for r in self.records)
        self.cfg = cfg; self.features = {}; self.observations = {}
        # Normalize the actual command by the declared action bounds, rather
        # than reusing statistics derived from a different action semantic.
        self.mean = np.zeros(14, np.float32); self.std = np.ones(14, np.float32)
        self.std[7:10] = .3; self.std[13] = .04

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([781021, self.cfg["seed"], int(index)]))
        row = self.records[int(rng.integers(len(self.records)))]; ident = row["id"]
        start = int(rng.integers(10))
        if ident not in self.features:
            self.features[ident] = np.load(row["local_features"], mmap_mode="r")
            with np.load(row["local_observations"]) as stream:
                assert set(stream.files) == {"render", "commands"}
                self.observations[ident] = (stream["render"].copy(), stream["commands"].copy())
        render, command = self.observations[ident]
        raw = np.zeros((8, 14), np.float32)
        raw[:-1, 7:10] = command[start:start+7, :3]; raw[:-1, 13] = -.04*command[start:start+7, 6]
        action = np.zeros_like(raw); action[:-1] = (raw[:-1]-self.mean)/self.std
        return dict(features=torch.from_numpy(self.features[ident][start:start+8].transpose(1, 0, 2, 3).copy()),
            render=torch.from_numpy(render[start:start+8].transpose(1, 0, 2, 3, 4).copy()),
            action=torch.from_numpy(action))


def prepare(raw):
    f = raw["features"].cuda(non_blocking=True); b, v, w = f.shape[:3]
    return dict(features=f.flatten(0, 1), render=raw["render"].cuda(non_blocking=True).flatten(0, 1),
        action=raw["action"].cuda(non_blocking=True)[:, None].expand(b, v, w, 14).reshape(b*v, w, 14))


def rates(opt, elapsed):
    warmup = 900.
    factor = max(elapsed/warmup, .001) if elapsed < warmup else math.sqrt(warmup/elapsed)
    for group in opt.param_groups: group["lr"] = group["base_lr"]*factor


def checkpoint(root, run, cfg, model, opt, step, elapsed, manifest, rank, world, label=None):
    import torch.distributed as dist
    rng = [None]*world; dist.all_gather_object(rng, common.rng_state())
    if rank == 0:
        state = dict(model=common.cpu(model.state_dict()), optimizer=common.cpu(opt.state_dict()),
            scheduler=dict(kind="warmup900-inverse-sqrt-active-wall", elapsed=elapsed), rank_rng=rng,
            step=step, samples=step*32, active_seconds=elapsed, world_size=world,
            parameter_associations=common.associations(model, opt), config_sha256=sha(run/"config.json"),
            nn_sha256=common.state_hash(model.state_dict()), data_cursor=dict(next_step=step,
                manifest=str(manifest), manifest_sha256=sha(manifest), deterministic_counter_batches=True))
        common.save_full_state(run, state)
        if label:
            folder = run/label; folder.mkdir(exist_ok=False)
            torch.save(state["model"], folder/"weights.pt")
            save(folder/"ready.json", dict(passed=True, label=label, step=step, active_seconds=elapsed,
                nn_sha256=state["nn_sha256"], weights_sha256=sha(folder/"weights.pt"),
                data_manifest_sha256=sha(manifest)))
    dist.barrier()


def train(root, run, manifest, target, label, max_updates=None):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local); torch.set_num_threads(2); torch.set_float32_matmul_precision("high")
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    cfg = json.loads((run/"config.json").read_text()); assert world == cfg["world_size"]
    model = build(root, cfg).cuda()
    from vforge.ops import OpsConfig, initialize_ops
    initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False), device=torch.device("cuda"), parallel_state=None)
    opt = optimizer(model, cfg); step = 0; elapsed = 0.; restored = None
    for group in opt.param_groups: group["base_lr"] = group["peak_lr"]
    if (run/"last.ckpt").exists():
        full = torch.load(run/"last.ckpt", map_location="cpu", weights_only=False)
        assert full["config_sha256"] == sha(run/"config.json") and full["world_size"] == world
        assert full["parameter_associations"] == common.associations(model, opt)
        model.load_state_dict(full["model"], strict=True); opt.load_state_dict(full["optimizer"])
        assert common.checkpoint_equal(full["optimizer"], opt.state_dict())
        assert common.state_hash(model.state_dict()) == full["nn_sha256"]
        assert sha(Path(full["data_cursor"]["manifest"])) == full["data_cursor"]["manifest_sha256"]
        assert full["data_cursor"]["next_step"] == full["step"]
        step = full["step"]; elapsed = full["active_seconds"]
        assert full["scheduler"]["elapsed"] == elapsed
        restored = full["rank_rng"][rank]
        if rank == 0:
            save(run/f"restore-{step:08d}.json", dict(passed=True, step=step,
                model_optimizer_scheduler_rank_RNG_cursor_exact=True, world_size=world,
                previous_manifest=full["data_cursor"]["manifest"], next_manifest=str(manifest),
                data_version_transition_is_intentional=sha(manifest)!=full["data_cursor"]["manifest_sha256"]))
        del full
    elif rank == 0:
        shared = {k:v for k,v in model.state_dict().items() if k.startswith(("feature_adapter.", "pool.", "predictor.", "action_encoder."))}
        save(run/"initial.json", dict(passed=True, fresh_random_model=True, seed=cfg["seed"],
            nn_sha256=common.state_hash(model.state_dict()), shared_encoder_predictor_sha256=common.state_hash(shared),
            parameters=sum(p.numel() for p in model.parameters()), trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    wrapped = DistributedDataParallel(common.LossModel(model), device_ids=[local],
        broadcast_buffers=False, find_unused_parameters=True)
    data = Windows(root, manifest, cfg)
    loader = DataLoader(data, batch_sampler=common.CounterBatches(step, 100000000, 32, rank, world),
        num_workers=2, pin_memory=True, prefetch_factor=2,
        generator=torch.Generator().manual_seed(91000+cfg["seed"]+rank))
    iterator = iter(loader)
    if restored is None:
        seed = 98000+cfg["seed"]+rank
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    else:
        common.restore_rng(restored); assert common.checkpoint_equal(common.rng_state(), restored)
    segment = 0; saved_at = elapsed; started = time.time()
    while elapsed < target:
        # A graceful stop still publishes a full state below.
        if (root/"STOP").exists(): break
        began = time.perf_counter(); batch = prepare(next(iterator)); iteration_rng = common.rng_state()
        opt.zero_grad(set_to_none=True); rates(opt, elapsed)
        with torch.autocast("cuda", dtype=torch.bfloat16): loss, metrics = wrapped(batch, step+1)
        finite = torch.tensor(int(bool(torch.isfinite(loss))), device="cuda"); dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if bool(finite): loss.backward()
        valid = bool(finite) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        finite.fill_(int(valid)); dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            opt.zero_grad(set_to_none=True); common.restore_rng(iteration_rng)
            with torch.autocast("cuda", enabled=False): loss, metrics = wrapped(batch, step+1)
            assert torch.isfinite(loss); loss.backward()
            valid = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            finite.fill_(int(valid)); dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if rank == 0: save(run/f"numeric-retry-{step+1:08d}.json", dict(passed=bool(finite), same_batch_RNG_cursor=True, precision="FP32"))
            assert bool(finite), ("non-finite FP32 retry", step+1)
        norms = {}
        if segment < 8 or step % 100 == 0:
            prefixes = ["feature_adapter.", "pool.", "predictor.", "action_encoder.",
                "future_decoder." if cfg["variant"] == "feature-delta" else "renderer."]
            for prefix in prefixes:
                norms[prefix] = sum(float(p.grad.detach().float().norm())**2
                    for name, p in model.named_parameters() if name.startswith(prefix) and p.grad is not None)**.5
                if step >= 2: assert norms[prefix] > 0 and math.isfinite(norms[prefix]), (prefix, norms)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step(); torch.cuda.synchronize()
        duration = torch.tensor(time.perf_counter()-began, device="cuda", dtype=torch.float64)
        dist.all_reduce(duration, op=dist.ReduceOp.MAX); elapsed += float(duration); step += 1; segment += 1
        if rank == 0 and (segment <= 8 or step % 100 == 0):
            save(run/"progress.json", dict(step=step, samples=step*32, active_seconds=elapsed,
                segment_seconds=time.time()-started, loss=float(loss.detach()),
                metrics={k:float(v.detach()) for k,v in metrics.items()}, lr=[g["lr"] for g in opt.param_groups],
                gradients=norms,
                manifest_sha256=sha(manifest), variant=cfg["variant"], time=time.time()))
        final = elapsed >= target or (max_updates is not None and segment >= max_updates)
        if final or elapsed-saved_at >= 900:
            checkpoint(root, run, cfg, model, opt, step, elapsed, manifest, rank, world, label if final else None)
            saved_at = elapsed
        if final: break
    if not (run/label/"ready.json").exists(): checkpoint(root, run, cfg, model, opt, step, elapsed, manifest, rank, world, label)
    if rank == 0:
        save(run/("TRAINED-"+label+".json"), dict(passed=True, step=step, active_seconds=elapsed,
            segment_updates=segment, model_optimizer_scheduler_rank_RNG_cursor_saved=True, world_size=world))
    dist.destroy_process_group()


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True);p.add_argument("--run",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--target-seconds",type=float,required=True)
    p.add_argument("--label",required=True);p.add_argument("--max-updates",type=int)
    a=p.parse_args();train(a.root,a.run,a.manifest,a.target_seconds,a.label,a.max_updates)
