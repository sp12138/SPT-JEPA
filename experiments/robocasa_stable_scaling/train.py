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


class ConditionalFeatureDecoder(nn.Module):
    """A scene-conditioned observation map with exactly zero change at zero."""
    def __init__(self):
        super().__init__()
        self.context = nn.Linear(1664, 192)
        self.motion = nn.Linear(64, 192, bias=False)
        self.output = nn.Linear(192, 1664, bias=False)

    def forward(self, delta, anchor):
        context = self.context(anchor.reshape(anchor.shape[0], 1, 1664, 16).transpose(-1, -2))
        motion = self.motion(delta)[:, :, None]
        # Preserve small changes while using mixed precision for the linears.
        value = self.output(F.silu(context.float()+motion.float())-F.silu(context.float()))
        return value.transpose(-1, -2).flatten(-2)


def additive_predict(self, means, actions):
    return means.float()+actions.float()


class ControlAffineTransition(nn.Module):
    """Learn autonomous drift and a state-dependent action response."""
    def __init__(self):
        super().__init__()
        self.field = nn.Sequential(nn.Linear(64, 192), nn.GELU(), nn.Linear(192, 64*65))
        nn.init.zeros_(self.field[-1].weight)
        nn.init.zeros_(self.field[-1].bias)

    def forward(self, means, actions):
        with torch.autocast("cuda", enabled=False):
            coefficients = self.field(means.float()).reshape(*means.shape[:-1], 64, 65)
            drift = coefficients[..., 0]
            effect = (coefficients[..., 1:] @ actions.float().unsqueeze(-1)).squeeze(-1)
            return means.float()+drift+effect


def control_affine_predict(self, means, actions):
    return self.predictor(means, actions)


def fp32_cached_encode(self, info):
    with torch.autocast("cuda", enabled=False):
        return self.original_cached_encode(info)


def conditional_initial_rate(self, mean):
    # q(z0|x0) is also the conditional prior p(z0|x0): its KL is zero.
    # x0 is already observed by the forward learner; future RGB is not an input.
    return mean.new_zeros(mean.shape[0], dtype=torch.float32)


def observation_codes(model, dense):
    if getattr(model,"fp32_observation",False):
        with torch.autocast("cuda",enabled=False):
            return _observation_codes(model,dense.float())
    return _observation_codes(model,dense)


def _observation_codes(model, dense):
    b,t,p,_ = dense.shape
    tokens = model.feature_adapter(dense.flatten(0, 1)).reshape(b,t,p,192)
    if getattr(model,"joint_views",False):
        assert b % 2 == 0
        tokens = tokens.reshape(b//2,2,t,p,192).permute(0,2,1,3,4)
        tokens = tokens+model.joint_view_embedding[None,None,:,None]
        return model.pool(tokens.reshape(b//2*t,2*p,192)).reshape(b//2,t,64)
    return model.pool(tokens.flatten(0,1)).reshape(b,t,64)


def decode_feature_change(model, delta, anchor):
    if getattr(model,"joint_views",False):
        delta = delta[:,None].expand(-1,2,-1,-1).reshape(anchor.shape[0],delta.shape[1],64)
    return (model.future_decoder(delta,anchor) if model.conditional_feature_decoder
        else model.future_decoder(delta)).float()


def feature_delta_loss(self, batch, step=0, ramp_steps=0):
    dense = batch["features"].float()
    b, t, patches, width = dense.shape
    assert patches == 196 and width == 1664
    means = observation_codes(self,dense)
    raw_action = batch["action"]
    if self.joint_views: raw_action = raw_action.reshape(b//2,2,t,14)[:,0]
    if self.fp32_observation:
        with torch.autocast("cuda",enabled=False): actions=self.action_encoder(raw_action.float())
    else: actions = self.action_encoder(raw_action)
    # Fixed visual features, with the actual first observation as an anchor.
    # There is no learned skip that can bypass the bottleneck with future RGB.
    target = F.adaptive_avg_pool2d(dense.reshape(b*t, 14, 14, width)
        .permute(0, 3, 1, 2), (4, 4)).flatten(1).reshape(b, t, -1).detach()
    prediction = self.predict(means[:, :-1], actions[:, :-1])
    def decode(value):
        return decode_feature_change(self,value,target[:, :1])
    decoded = decode(prediction-means[:, :1])
    delta = target[:, 1:]-target[:, :1]
    tf = (decoded-delta).abs().mean()
    prefix = int(torch.randint(t-2, (1,), device=dense.device))+1
    history = torch.cat((means[:, :prefix], prediction[:, prefix-1:prefix].detach()), 1)
    future = self.predict(history, actions[:, :prefix+1])[:, -1:]
    ar = (decode(future-means[:, :1])-(target[:, prefix+1:prefix+2]-target[:, :1])).abs().mean()
    loss = tf*.5+ar*.25
    copy = (target[:, 1:]-target[:, :-1]).abs().mean()
    return loss, dict(loss=loss, fixed_feature_TF=tf, fixed_feature_AR=ar,
        fixed_feature_copy_L1=copy, feature_std=means.float().std((0, 1)).mean())


def feature_codec_loss(self, batch, step=0, ramp_steps=0):
    """SPT's analytic conditional code; replace pixel flow distortion only.

    Fixed future visual features are natural observations, not action labels.
    Keep the existing sigma/rho/beta and RGB-normalized code rate. This is a
    rate/distortion replacement, not a claim of a new joint feature ELBO.
    """
    from experiments.robocasa_refine.analytic_model import innovation_error
    from vforge.models.as_jepa.codes import entry_nats, surprise_nats
    from vforge.models.as_jepa.constants import FRAME_ELEMS
    dense=batch["features"].float(); bv,t,_,width=dense.shape
    means=observation_codes(self,dense).float(); b=means.shape[0]
    raw=batch["action"]
    if self.joint_views: raw=raw.reshape(b,2,t,14)[:,0]
    with torch.autocast("cuda",enabled=False): actions=self.action_encoder(raw.float())
    mu=means.reshape(b,t,self.state_slots,self.state_width)
    noise=torch.randn_like(mu)
    for i in range(1,t): noise[:,i]=self.rho*noise[:,i-1]+math.sqrt(1-self.rho**2)*noise[:,i]
    noisy=mu+self.sigma0*noise
    forecast_means=self.mean_trajectory(means[:,0],actions[:,:-1]).reshape_as(mu)
    powers=mu.new_tensor([self.rho**i for i in range(1,t)])[None,:,None,None]
    forecast=forecast_means[:,1:]+powers*(noisy[:,:1]-mu[:,:1])
    choose=torch.rand(b,t-1,1,1,device=mu.device)<self.forecast_fraction
    condition=torch.where(choose,forecast,noisy[:,1:]).flatten(2)
    target=F.adaptive_avg_pool2d(dense.reshape(bv*t,14,14,width).permute(0,3,1,2),(4,4)).flatten(1).reshape(bv,t,-1).detach()
    predicted=decode_feature_change(self,condition-means[:,:1],target[:,:1])
    distortion=(predicted-(target[:,1:]-target[:,:1])).abs().mean()
    error=innovation_error(mu,forecast_means,self.rho)
    entry=entry_nats(mu[:,0],self.sigma0)
    surprise=surprise_nats(error,torch.zeros_like(error),self.sigma0)
    surprise=surprise+self.state_dim*.5*(-self.rho**2-math.log1p(-self.rho**2))
    rate=(entry+surprise.sum(1)).mean()/((t-1)*FRAME_ELEMS*(2 if self.joint_views else 1))
    loss=distortion+self.beta*rate
    return loss,dict(loss=loss,feature_distortion=distortion,rate=rate,entry=entry.mean(),
        surprise=surprise.mean(),feature_std=means.std((0,1)).mean(),
        clean_rollout_mse=(mu[:,1:]-forecast_means[:,1:]).square().mean(),
        rate_share=self.beta*rate/distortion.detach().clamp_min(1e-8))


def pixel_prediction_distortion(self, batch, render, anchor, condition):
    """Compare conditional change density with conditional future-image mean.

    Both use the original code/rate and natural future RGB, without action
    targets. The residual density is an invertible target translation given
    the observed anchor. The mean objective deliberately abandons density
    modeling; a constant renderer input cannot leak observed future pixels.
    """
    from vforge.models.as_jepa.pixels import pack
    from vforge.models.as_jepa.flow import flow_loss, draw_timesteps
    w = render.shape[1]
    y = pack(render[:, 1:].flatten(0, 1))
    anchors = pack(anchor).repeat_interleave(w-1, dim=0)
    if self.pixel_objective == "residual-flow":
        return flow_loss(self.renderer, y-anchors, condition.flatten(0, 1), anchors,
                         self.training_shift)
    assert self.pixel_objective == "future-mean"
    # Consume the baseline draws to preserve subsequent RNG coupling. Neither
    # random value, nor a partially observed future frame, enters the renderer.
    draw_timesteps(y.shape[0], y.device, self.training_shift)
    torch.randn_like(y)
    t = torch.ones(y.shape[0], 1, device=y.device)
    with self.renderer.anchored(anchors):
        predicted = self.renderer(torch.zeros_like(y), t, condition=condition.flatten(0, 1))
    return (predicted-y).square().mean()


def build(root, cfg):
    base = old.build(root, cfg)
    if cfg["variant"] in ("feature-delta", "feature-conditional", "action-transport", "codec-transport", "codec-conditional"):
        base.renderer.requires_grad_(False)
        base.conditional_feature_decoder = cfg["variant"] != "feature-delta"
        base.joint_views = cfg.get("joint_views",False)
        base.fp32_observation = cfg["variant"] in ("action-transport","codec-transport","codec-conditional")
        if base.joint_views:
            assert cfg["variant"] in ("action-transport","codec-transport","codec-conditional")
            base.joint_view_embedding = nn.Parameter(torch.zeros(2,192))
        with torch.random.fork_rng():
            torch.manual_seed(90120261114)
            base.future_decoder = (nn.Linear(64, 1664*16, bias=False) if cfg["variant"] == "feature-delta"
                else ConditionalFeatureDecoder())
        if cfg["variant"] in ("action-transport","codec-transport"):
            base.predictor.requires_grad_(False)
            with torch.random.fork_rng():
                torch.manual_seed(91220261114)
                base.action_encoder = nn.Linear(14, 64, bias=False)
            base.predict = types.MethodType(additive_predict, base)
        base.loss = types.MethodType(feature_codec_loss if cfg["variant"].startswith("codec-") else feature_delta_loss, base)
    else:
        assert cfg["variant"] in ("spt-rate", "spt-residual-flow", "spt-future-mean",
                                  "spt-control-affine", "spt-deep-observation", "spt-conditional-rate")
        if cfg["variant"] in ("spt-residual-flow", "spt-future-mean"):
            base.pixel_objective = cfg["variant"].removeprefix("spt-")
            base.distortion = types.MethodType(pixel_prediction_distortion, base)
        if cfg["variant"] == "spt-control-affine":
            with torch.random.fork_rng():
                torch.manual_seed(92320261114+cfg["seed"])
                base.action_encoder = nn.Linear(14, 64, bias=False)
                base.predictor = ControlAffineTransition()
            base.predict = types.MethodType(control_affine_predict, base)
        if cfg["variant"] == "spt-deep-observation":
            from experiments.robocasa_action_probe.probe import deepen_observation_adapter
            assert cfg["observation_depth"] == 2
            with torch.random.fork_rng():
                torch.manual_seed(93420261114+cfg["seed"])
                base.feature_adapter = deepen_observation_adapter(base.feature_adapter, 2)
        if cfg["variant"] == "spt-conditional-rate":
            base.initial_rate = types.MethodType(conditional_initial_rate, base)
        if cfg["variant"] in ("spt-deep-observation", "spt-control-affine", "spt-conditional-rate"):
            base.original_cached_encode = base.encode
            base.encode = types.MethodType(fp32_cached_encode, base)
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
                manifest=str(manifest), manifest_sha256=sha(manifest), deterministic_counter_batches=True),
            training_source_sha256=sha(Path(__file__)))
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
    if cfg["variant"] in ("action-transport","codec-transport","codec-conditional",
                          "spt-control-affine","spt-deep-observation","spt-conditional-rate"):
        torch.set_float32_matmul_precision("highest")
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
        encoder = {k:v for k,v in shared.items() if k.startswith(("feature_adapter.","pool."))}
        full_encoder = dict(encoder)
        if cfg.get("joint_views"): full_encoder["joint_view_embedding"] = model.joint_view_embedding
        save(run/"initial.json", dict(passed=True, fresh_random_model=True, seed=cfg["seed"],
            nn_sha256=common.state_hash(model.state_dict()), shared_encoder_predictor_sha256=common.state_hash(shared),
            common_encoder_parameter_sha256=common.state_hash(encoder), visual_encoder_sha256=common.state_hash(full_encoder),
            initial_projection_and_pool_sha256=common.state_hash({k:v for k,v in encoder.items()
                if k.startswith(("feature_adapter.0.","feature_adapter.1.","pool."))}),
            joint_views=cfg.get("joint_views",False),
            observation_decoder_sha256=common.state_hash(model.future_decoder.state_dict()) if hasattr(model,"future_decoder") else None,
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
        stopping = torch.tensor(int((root/"STOP").exists() or (run/"STOP").exists()),device="cuda")
        dist.all_reduce(stopping,op=dist.ReduceOp.MAX)
        if bool(stopping): break
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
        if segment < 8 or (step+1) % 100 == 0:
            prefixes = ["feature_adapter.", "pool.", "action_encoder."]
            if cfg.get("joint_views"): prefixes.append("joint_view_embedding")
            if cfg["variant"] not in ("action-transport","codec-transport"): prefixes.append("predictor.")
            prefixes.append("future_decoder." if hasattr(model,"future_decoder") else "renderer.")
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
    complete = elapsed >= target or (max_updates is not None and segment >= max_updates)
    if not (run/label/"ready.json").exists():
        checkpoint(root, run, cfg, model, opt, step, elapsed, manifest, rank, world, label if complete else None)
    if rank == 0:
        save(run/("TRAINED-"+label+".json"), dict(passed=complete, interrupted=not complete, step=step, active_seconds=elapsed,
            segment_updates=segment, model_optimizer_scheduler_rank_RNG_cursor_saved=True, world_size=world))
    dist.destroy_process_group()


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True);p.add_argument("--run",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--target-seconds",type=float,required=True)
    p.add_argument("--label",required=True);p.add_argument("--max-updates",type=int)
    a=p.parse_args();train(a.root,a.run,a.manifest,a.target_seconds,a.label,a.max_updates)
