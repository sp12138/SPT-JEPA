"""Detached factual-action probes, including a shared nonlinear potential.

The residual is a difference of the same function, so identical observations
always imply zero action. Only the actually issued action of this transition
is a label. No goal, expert path or world-model gradient enters this fit.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.robocasa_action_probe import probe
from experiments.robocasa_action_probe.scaling import summarize
from experiments.robocasa_compare import train as common
from experiments.robocasa_compare.acquire import save, sha

MODULE = "experiments.robocasa_stable_scaling.readout"


class Potential(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone())
        self.net = nn.Sequential(nn.LayerNorm(weight.shape[1], elementwise_affine=False),
            nn.Linear(weight.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 3))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, current, following):
        current, following = current.detach(), following.detach()
        return F.linear(following-current, self.weight)+self.net(following)-self.net(current)


def load_pairs(directory):
    rows, codes, actions, splits = [], [], [], []
    for shard in range(8):
        path = directory/"cache"/f"part-{shard}.npz"
        proof = json.loads(path.with_suffix(".json").read_text())
        assert proof["passed"] and sha(path) == proof["sha256"]
        assert proof["all_features_detached"] and proof["readout_frozen_unchanged"]
        with np.load(path) as data:
            rows.extend(data["indices"]); codes.extend(data["codes"])
            actions.extend(data["actions"]); splits.extend(data["splits"])
    assert sorted(rows) == list(range(1216))
    order = np.argsort(rows); z = np.stack(codes)[order]; a = np.stack(actions)[order]
    train = np.asarray(splits)[order] == "train"
    assert sum(train) == 1120 and sum(~train) == 96
    result = []
    for mask in (train, ~train):
        result.append(tuple(torch.from_numpy(x[mask].reshape(-1, x.shape[-1])).float().cuda()
            for x in (z[:, :-1], z[:, 1:], a)))
    return result


def fit(root, arm="spt64-linear", steps=5000):
    torch.set_num_threads(2); torch.set_float32_matmul_precision("high")
    torch.manual_seed(20261114)
    directory = root/"heads"/arm
    source = directory/"linear-head.pt"; ready = json.loads((directory/"linear-ready.json").read_text())
    assert sha(source) == ready["head_sha256"]
    weight = torch.load(source, map_location="cpu", weights_only=True)["weight"]
    model = Potential(weight).cuda(); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    (x0, x1, target), (v0, v1, vy) = load_pairs(directory)
    generator = torch.Generator(device="cuda").manual_seed(20261114)
    with torch.no_grad(): baseline = float((model(v0, v1)-vy).square().mean())
    # The archived ridge score was evaluated in FP64, whereas this neural head
    # uses FP32. Check identity in matching arithmetic, then report FP32 too.
    with torch.no_grad():
        reference = float((F.linear((v1-v0).double(), weight.cuda().double())-vy.double()).square().mean())
    assert abs(reference**.5-ready["validation_RMSE"]) < 1e-6
    best = (baseline, common.cpu(model.state_dict()), 0); history = []
    started = time.perf_counter()
    for step in range(1, steps+1):
        indices = torch.randint(len(x0), (512,), device="cuda", generator=generator)
        opt.zero_grad(set_to_none=True)
        loss = (model(x0[indices], x1[indices])-target[indices]).square().mean()
        assert torch.isfinite(loss); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); opt.step()
        if step % 100 == 0 or step == steps:
            with torch.no_grad(): mse = float((model(v0, v1)-vy).square().mean())
            assert np.isfinite(mse)
            history.append(dict(step=step, train_RMSE=float(loss.detach())**.5, validation_RMSE=mse**.5))
            if mse < best[0]: best = (mse, common.cpu(model.state_dict()), step)
    common.save_full_state(directory, dict(model=common.cpu(model.state_dict()), optimizer=common.cpu(opt.state_dict()),
        scheduler=dict(kind="constant", lr=1e-3), step=steps, rng=common.rng_state(),
        data_cursor=dict(kind="CUDA random sampler", generator=generator.get_state().cpu(), batch=512),
        source_linear_sha256=ready["head_sha256"], samples=steps*512))
    model.load_state_dict(best[1]); model.zero_grad(set_to_none=True)
    leaf = v0[:32].detach().requires_grad_()
    test = (model(leaf, v1[:32])-vy[:32]).square().mean(); test.backward()
    assert leaf.grad is None and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    assert any(p.grad is not None and float(p.grad.norm()) > 0 for p in model.parameters())
    with torch.no_grad(): assert torch.equal(model(v0[:32], v0[:32]), torch.zeros_like(vy[:32]))
    model.zero_grad(set_to_none=True)
    torch.save(dict(weight=weight, potential=best[1], head_type="potential", selected_step=best[2]), directory/"head.pt")
    save(directory/"ready.json", dict(ready, head_sha256=sha(directory/"head.pt"),
        head_type="shared nonlinear potential difference", selected_step=best[2], steps_trained=steps,
        validation_RMSE=best[0]**.5, linear_validation_RMSE=baseline**.5,
        linear_FP64_validation_RMSE=reference**.5,
        validation_history=history, actual_stopgradient_backward_pass=True,
        exact_zero_for_identical_codes=True, no_action_loss_to_backbone=True,
        elapsed_seconds=time.perf_counter()-started, target_case_feedback_used_for_selection=False))
    print(json.dumps(dict(root=str(root), best_step=best[2], baseline_RMSE=baseline**.5,
        validation_RMSE=best[0]**.5)), flush=True)


class PotentialPolicy(probe.Policy):
    def __init__(self, root, arm):
        super().__init__(root, arm)
        saved = torch.load(root/"heads"/arm/"head.pt", map_location="cpu", weights_only=True)
        self.potential = Potential(saved["weight"]).cuda().eval().requires_grad_(False)
        self.potential.load_state_dict(saved["potential"], strict=True)
        self.head_hash = common.state_hash(self.potential.state_dict())

    @torch.no_grad()
    def command(self, observation, goal):
        current = self.encode(observation)
        value = self.potential(current, goal)
        assert torch.isfinite(value).all()
        value *= torch.clamp(.3/value.norm().clamp_min(1e-12), max=1.)
        action = np.zeros(7, np.float32); action[:3] = value.cpu().numpy()
        return action

    def unchanged(self):
        return super().unchanged() and common.state_hash(self.potential.state_dict()) == self.head_hash


def control(root, arm="spt64-linear", shard=0, shards=1):
    torch.set_num_threads(2); torch.set_float32_matmul_precision("high")
    policy = PotentialPolicy(root, arm)
    plan = json.loads((root/"plan.json").read_text())
    for index in range(len(plan["cases"])):
        if index % shards == shard: probe.evaluate(root, arm, index, policy=policy)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", choices=("fit", "control"), required=True); p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    if a.mode == "fit": fit(a.root, steps=a.steps)
    else: control(a.root, shard=a.shard, shards=a.shards)
