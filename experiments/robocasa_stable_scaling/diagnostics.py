"""Frozen factual validation of the predictive backbone, separately from control."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_compare import train as common
from experiments.robocasa_time_compare.train import build


@torch.no_grad()
def inspect(root, label):
    torch.set_num_threads(2); torch.set_float32_matmul_precision("high")
    plan = json.loads((root/"plan.json").read_text())
    source = Path(plan["source"])/"training/spt64-seed1"
    cfg = json.loads((source/"config.json").read_text())
    common.install_runtime(root/"code")
    from vforge.ops import OpsConfig, initialize_ops
    initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False), device=torch.device("cuda"), parallel_state=None)
    ready = json.loads((source/label/"ready.json").read_text()); path = source/label/"weights.pt"
    assert sha(path) == ready["weights_sha256"]
    model = build(root, cfg).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    before = common.state_hash(model.state_dict()); assert before == ready["nn_sha256"]
    folder = root/"readout-pilot"/label/"heads/spt64-linear/cache"
    rows, codes, splits = [], [], []
    for shard in range(8):
        with np.load(folder/f"part-{shard}.npz") as data:
            rows.extend(data["indices"]); codes.extend(data["codes"]); splits.extend(data["splits"])
    order = np.argsort(rows); z = np.stack(codes)[order]; split = np.asarray(splits)[order]
    card = json.loads((root/"data.json").read_text()); commands = []
    for record in card["records"]:
        if record["split"] == "validation":
            with np.load(Path("/tmp/spt-time-compare-20261112/observations")/(record["id"]+".npz")) as data:
                commands.append(data["commands"])
    z = z[split == "validation"].reshape(96, 17, 2, 64)
    commands = np.stack(commands)
    states = []; actions = []
    mean = np.asarray(card["mean"], np.float32); std = np.asarray(card["std"], np.float32)
    for start in (0, 9):
        states.append(z[:, start:start+8].transpose(0, 2, 1, 3).reshape(-1, 8, 64))
        raw = np.zeros((96, 7, 14), np.float32)
        raw[:, :, 7:10] = commands[:, start:start+7, :3]
        raw[:, :, 13] = -.04*commands[:, start:start+7, 6]
        actions.append(np.repeat(((raw-mean)/std)[:, None], 2, axis=1).reshape(-1, 7, 14))
    states = torch.from_numpy(np.concatenate(states)).cuda(); actions = torch.from_numpy(np.concatenate(actions)).cuda()
    zero = torch.from_numpy(-mean/std).cuda(); sums = dict(actual=0., zero=0., copy=0., response=0.)
    for offset in range(0, len(states), 32):
        latent = states[offset:offset+32]; action = actions[offset:offset+32]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model.predict(latent[:, :-1], model.action_encoder(action)).float()
            alternate = model.predict(latent[:, :-1], model.action_encoder(zero.expand_as(action))).float()
        target = latent[:, 1:]
        sums["actual"] += float((predicted-target).square().sum())
        sums["zero"] += float((alternate-target).square().sum())
        sums["copy"] += float((latent[:, :-1]-target).square().sum())
        sums["response"] += float((predicted-alternate).square().sum())
    assert common.state_hash(model.state_dict()) == before
    save(root/"predictive-diagnostic"/(label+".json"), dict(passed=True, label=label,
        source_nn_sha256=before, validation_episodes=96, windows_per_episode=2,
        actual_MSE_over_copy=sums["actual"]/sums["copy"], zero_MSE_over_copy=sums["zero"]/sums["copy"],
        action_response_energy_over_copy=sums["response"]/sums["copy"],
        sums=sums, new_SGD=0, model_unchanged=True, causal_counterfactual_not_observed=True,
        latent_units_change_across_checkpoints=True, not_a_control_result=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--label", required=True)
    a = p.parse_args(); inspect(a.root, a.label)
