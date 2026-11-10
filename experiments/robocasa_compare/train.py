"""Finite SPT-JEPA comparison using unchanged neural losses and full DDP resumes."""
import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import types

import numpy as np
import torch
from torch import nn

from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_compare.data import Windows


def install_runtime(code):
    """Import model primitives, avoiding unrelated runtime training frontends."""
    code = Path(code)
    sys.path.insert(0, str(code / "runtime"))
    for name in ["vforge", "vforge.models", "vforge.distributed"]:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(code / "runtime" / name.replace(".", "/"))]
            sys.modules[name] = module


def model_config(arm, seed):
    result = dict(mode="L2", beta=.25, sigma0=.1, window=8, anchor_gap=0,
                  renderer_width=384, renderer_depth=12, renderer_heads=6,
                  action_dim=14, rho=.8, forecast_fraction=.5,
                  forecast_seed=20261029 + seed * 100003)
    if arm == "spatial":
        result.update(state_slots=16, state_width=16, spatial_pool=True)
    elif arm == "long-rollout":
        result.update(window=16, rate_prior="forecast")
    else:
        assert arm == "temporal-forecast"
    return result


def build(config, code):
    install_runtime(code)
    config = dict(config)
    analytic=config.pop("analytic_prior",False)
    local=config.pop("local_encoder",False)
    if analytic:
        assert not local and not config.get("spatial_pool",False)
        from experiments.robocasa_refine.analytic_model import AnalyticResidualSPT
        return AnalyticResidualSPT(**config)
    if local:
        from experiments.robocasa_refine.local_model import LocalSpatialForecast
        assert config.pop("spatial_pool")
        return LocalSpatialForecast(**config)
    if "clean_calibration" in config:
        from experiments.robocasa_refine.model import RefinedSPT
        return RefinedSPT(**config)
    if config.pop("spatial_pool", False):
        from experiments.robocasa_compare.spatial_model import SpatialForecast
        return SpatialForecast(**config)
    from as_jepa_lewm.robust import RobustLeWM
    return RobustLeWM(**config)


def make_dataset(config,root,split="train"):
    if config.get("input_semantics"):
        from experiments.robocasa_refine.intervene import Interventions
        return Interventions(root,config["model"]["window"],config["seed"],split,config["input_semantics"])
    return Windows(root,config["model"]["window"],config["seed"],split)


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def make_optimizer(model, total, phase_start=0):
    renderer, other = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (renderer if name.startswith("renderer.") else other).append(p)
    groups = [dict(params=other, lr=5e-5, weight_decay=1e-3, betas=(.9, .999)),
              dict(params=renderer, lr=1e-4, weight_decay=0., betas=(.9, .95))]
    optimizer = torch.optim.AdamW(groups, foreach=True)
    horizon = total-phase_start
    assert horizon > 0
    warmup = max(1, int(horizon * .01))

    def schedule(step):
        step = max(0, step-phase_start)
        if step < warmup:
            return max(1, step) / warmup
        return .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, horizon - warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k:cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(v) for v in value)
    return value


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def prepare_batch(batch, device):
    from torch.nn import functional as F
    pixels = batch["pixels"].to(device, non_blocking=True)
    action = batch["action"].to(device, non_blocking=True)
    assert pixels.ndim == 6 and pixels.shape[1] == 2
    b, views, window = pixels.shape[:3]
    pixels = pixels.flatten(0, 1)
    action = action[:, None].expand(b, views, window, 14).reshape(b * views, window, 14)
    render = F.interpolate(pixels.flatten(0, 1).float(), size=(64, 64), mode="area").round()
    render = render.reshape(b * views, window, 3, 64, 64)
    return dict(pixels=pixels, action=action, render=render, anchor=render[:, 0], anchor224=pixels[:, 0])


class LossModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.core = model

    def forward(self, batch, step):
        return self.core.loss(batch, step=step)


class CounterBatches:
    def __init__(self, start, stop, global_batch, rank, world):
        assert global_batch % world == 0
        self.start, self.stop, self.batch = start, stop, global_batch
        self.rank, self.local = rank, global_batch // world

    def __iter__(self):
        for step in range(self.start, self.stop):
            begin = step * self.batch + self.rank * self.local
            yield list(range(begin, begin + self.local))

    def __len__(self):
        return self.stop - self.start


def gradients(model):
    result = {}
    for prefix in ["encoder", "pool", "predictor", "action_encoder", "renderer"]:
        values = [p.grad.detach().double().square().sum() for n, p in model.named_parameters()
                  if n.startswith(prefix + ".") and p.grad is not None]
        result[prefix] = float(torch.stack(values).sum().sqrt()) if values else 0.
    return result


def associations(model, optimizer):
    names = {id(p):n for n, p in model.named_parameters()}
    return [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]


def checkpoint_equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(checkpoint_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return isinstance(b,type(a)) and len(a)==len(b) and all(checkpoint_equal(x,y) for x,y in zip(a,b))
    return a==b


def save_full_state(output,state):
    # A shared-filesystem rename can report ENOENT after the destination was
    # already committed. Accept that outcome only after exact full-state readback.
    tmp=output/f"checkpoint-{state['step']}-pid{os.getpid()}.tmp"
    with tmp.open("wb") as stream:
        torch.save(state,stream)
        stream.flush();os.fsync(stream.fileno())
    try:
        tmp.replace(output/"last.ckpt")
    except OSError as error:
        try:committed=torch.load(output/"last.ckpt",map_location="cpu",weights_only=False)
        except Exception:raise error
        if not checkpoint_equal(state,committed):raise error
        save(output/f"checkpoint-rename-readback-{state['step']}.json",
             dict(passed=True,original_error=repr(error),full_model_optimizer_scheduler_rng_cursor_exact=True,
                  step=state["step"],sgd_replayed=0))


def checkpoint(output, model, optimizer, scheduler, config, step, world, rank, publish=True):
    import torch.distributed as dist
    all_rng = [None] * world
    dist.all_gather_object(all_rng, rng_state())
    failure=[None]
    if rank == 0:
      try:
        state = dict(model=cpu(model.state_dict()), optimizer=cpu(optimizer.state_dict()),
                     scheduler=scheduler.state_dict(), rank_rng=all_rng, step=step,
                     samples=step * config["global_batch"], world_size=world,
                     parameter_associations=associations(model, optimizer),
                     config_sha256=sha(output / "config.json"),
                     nn_sha256=state_hash(model.state_dict()), bf16_grad_scaler=None)
        save_full_state(output,state)
        if publish:
            endpoint = output / f"step-{step:06d}"
            endpoint.mkdir(exist_ok=False)
            torch.save(state["model"], endpoint / "weights.pt")
            save(endpoint / "ready.json", dict(passed=True, step=step, model_sha256=state["nn_sha256"],
                                              weights_sha256=sha(endpoint / "weights.pt")))
      except Exception as error:
        failure[0]=repr(error)
    dist.broadcast_object_list(failure,src=0)
    if failure[0]:raise RuntimeError("Checkpoint failed on rank0: "+failure[0])


def train(args):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    config = json.loads((args.run / "config.json").read_text())
    assert not (args.root / "STOP").exists() and not (args.run / "STOP").exists()
    assert 0 < args.stop_step <= config["max_steps"]
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = build(config["model"], args.root / "code").to(device)
    initial = torch.load(args.root / f"random-parent-seed{seed}.pt", map_location="cpu", weights_only=True)
    shared = {k:v for k, v in initial.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(shared, strict=False)
    optimizer, scheduler = make_optimizer(model, config["max_steps"], config.get("phase_start",0))
    start = 0
    resumed_rng = None
    if args.resume:
        state = torch.load(args.run / "last.ckpt", map_location="cpu", weights_only=False)
        assert state["config_sha256"] == sha(args.run / "config.json") and state["world_size"] == world
        assert state["parameter_associations"] == associations(model, optimizer)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = state["step"]
        assert state["samples"] == start * config["global_batch"] and start < args.stop_step
        resumed_rng = state["rank_rng"][rank]
        assert state_hash(model.state_dict()) == state["nn_sha256"]
        del state
    from vforge.ops import OpsConfig, initialize_ops
    initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False), device=device, parallel_state=None)
    wrapped = DistributedDataParallel(LossModel(model), device_ids=[local],
                                      broadcast_buffers=False, find_unused_parameters=True)
    if not args.resume and rank == 0:
        save(args.run / "initial.json", dict(passed=True, all_weights_random=True, seed=seed,
             common_random_parameters=len(shared), initial_sha256=state_hash(model.state_dict()),
             common_encoder_sha256=state_hash(model.encoder.state_dict()),
             parameters=sum(p.numel() for p in model.parameters()), world_size=world,
             global_batch=config["global_batch"], pretrained_or_policy_ancestry=False,
             sources={str(p):sha(p) for p in [Path(__file__), Path(inspect.getfile(type(model))),
                                             Path(inspect.getfile(importlib.import_module("as_jepa_lewm.model")))]}))
    data_root = args.data_root or Path(config["data"])
    assert sha(data_root / "dataset.json") == sha(Path(config["data"]) / "dataset.json")
    dataset = make_dataset(config,data_root)
    loader = DataLoader(dataset, batch_sampler=CounterBatches(start, args.stop_step, config["global_batch"], rank, world),
                        num_workers=2, pin_memory=True, persistent_workers=False,
                        generator=torch.Generator().manual_seed(seed + 9000 + rank), prefetch_factor=2)
    iterator = iter(loader)
    if resumed_rng:
        restore_rng(resumed_rng)
    else:
        torch.manual_seed(seed + 50000 + rank)
        random.seed(seed + 50000 + rank); np.random.seed(seed + 50000 + rank)
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    began = time.time()
    model.train()
    step = start
    try:
        for batch in iterator:
            optimizer.zero_grad(set_to_none=True)
            batch = prepare_batch(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = wrapped(batch, step)
            assert torch.isfinite(loss), (step, float(loss))
            loss.backward()
            if step < 3 or (step + 1) % 100 == 0:
                norms = gradients(model)
                assert all(math.isfinite(v) for v in norms.values())
                # The residual output and conditional gates initialize at zero.
                # Actuator gradients appear on the third backward, after two updates.
                if step >= 2:
                    assert all(v > 0 for v in norms.values()), (step, norms)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); scheduler.step(); step += 1
            if rank == 0 and (step <= 3 or step % 100 == 0):
                save(args.run / "progress.json", dict(step=step, initial_step=start, loss=float(loss.detach()),
                    gradients=norms, gradient_update=step, elapsed_seconds=time.time()-began,
                    seconds_per_update=(time.time()-began)/max(1, step-start),
                    metrics={k:float(v.detach()) for k, v in metrics.items()},
                    learning_rates=[g["lr"] for g in optimizer.param_groups],
                    updated_unix=time.time(), host=os.uname().nodename,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(), world_size=world))
            stop = torch.tensor(int(stopping or (args.root / "STOP").exists() or (args.run / "STOP").exists()), device=device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if step == args.stop_step or bool(stop):
                checkpoint(args.run, model, optimizer, scheduler, config, step, world, rank)
                if rank == 0:
                    save(args.run / f"stage-{step:06d}.json", dict(passed=not bool(stop), step=step,
                         previous_step=start, stopped=bool(stop), full_state_saved=True,
                         unsaved_updates_replayed=json.loads((args.run / "REPLAY.json").read_text())["updates"]
                         if (args.run / "REPLAY.json").exists() else 0))
                break
            if step % 250 == 0:
                checkpoint(args.run, model, optimizer, scheduler, config, step, world, rank, publish=False)
        assert step == args.stop_step or bool(stop)
    except BaseException as error:
        if rank == 0:
            save(args.run / f"training-error-from-{start}.json", dict(error=repr(error), attempted_step=step))
        raise
    finally:
        dist.destroy_process_group()


def initialize(root):
    root = Path(root)
    assert (root / "packed/dataset.json").exists()
    for seed in [1, 2]:
        torch.manual_seed(seed)
        model = build(model_config("temporal-forecast", seed), root / "code")
        path = root / f"random-parent-seed{seed}.pt"
        assert not path.exists()
        torch.save(model.state_dict(), path)
        for arm, host in [("temporal-forecast", "dev13"), ("spatial", "dev15"), ("long-rollout", "dev6")]:
            run = root / "training" / f"{arm}-seed{seed}"
            run.mkdir(parents=True, exist_ok=False)
            cfg = dict(seed=seed, arm=arm, host=host, model=model_config(arm, seed),
                       data=str(root / "packed"), max_steps=5000,
                       global_batch=64 if arm == "long-rollout" else 128,
                       stages=[2, 500, 5000], random_parent=str(path),
                       supervision="Future RGB and own latent prediction only; no action target or expert path",
                       cameras="Independent shared-weight side/wrist models, averaged costs")
            save(run / "config.json", cfg)
        del model
    save(root / "training-initialized.json", dict(passed=True, model_updates=0,
          pretrained_or_policy_ancestry=False, shared_parents_are_random=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-directory", dest="run", type=Path)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    if args.initialize:
        initialize(args.root)
    else:
        train(args)
