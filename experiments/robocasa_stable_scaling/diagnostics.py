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
def inspect_decoder_span(root, run, label):
    """Oracle linear-range diagnostic; future targets never enter a learner."""
    from torch.nn import functional as F
    assert json.loads((run/"config.json").read_text())["variant"] == "feature-delta"
    torch.set_num_threads(2); torch.set_float32_matmul_precision("highest")
    ready = json.loads((run/label/"ready.json").read_text()); weights = run/label/"weights.pt"
    assert sha(weights) == ready["weights_sha256"]
    matrix = torch.load(weights, map_location="cpu", weights_only=True)["future_decoder.weight"].float().cuda()
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    orthogonality_error = float((q.T@q-torch.eye(q.shape[1], device=q.device)).abs().max())
    assert orthogonality_error < 1e-5
    singular = torch.linalg.svdvals(matrix)
    assert float(singular[-1]) > float(singular[0])*1e-6
    sums = dict(oracle_L1=0., oracle_L2=0., copy_L1=0., copy_L2=0., target_L2=0.)
    records = [r for r in json.loads((root/"data.json").read_text())["records"] if r["split"] == "validation"]
    assert len(records) == 96
    for row in records:
        features = np.load(Path("/tmp/spt-time-compare-20261112/features/vjepa")/(row["id"]+".npy"), mmap_mode="r")
        dense = torch.from_numpy(np.stack([features[s:s+8].transpose(1, 0, 2, 3) for s in (0, 9)]).copy()).cuda().float()
        phi = F.adaptive_avg_pool2d(dense.reshape(-1, 14, 14, 1664).permute(0, 3, 1, 2), (4, 4)).flatten(1).reshape(4, 8, -1)
        target = phi[:, 1:]-phi[:, :1]
        residual = target-(target@q)@q.T
        copy_error = phi[:, 1:]-phi[:, :-1]
        sums["oracle_L1"] += float(residual.abs().sum()); sums["oracle_L2"] += float(residual.square().sum())
        sums["copy_L1"] += float(copy_error.abs().sum()); sums["copy_L2"] += float(copy_error.square().sum())
        sums["target_L2"] += float(target.square().sum())
    save(root/"decoder-span-diagnostic"/run.name/(label+".json"), dict(passed=True, new_SGD=0,
        source_weights_sha256=ready["weights_sha256"], rank=q.shape[1], orthogonality_error=orthogonality_error,
        oracle_minimum_L2_over_copy=sums["oracle_L2"]/sums["copy_L2"],
        oracle_L1_at_L2_projection_over_copy=sums["oracle_L1"]/sums["copy_L1"],
        explained_anchored_feature_L2=1-sums["oracle_L2"]/sums["target_L2"],
        oracle_next_RGB_features_are_diagnostic_only=True, no_proposal_or_teacher_created=True,
        L1_value_is_not_a_lower_bound=True, not_a_control_result=True, sums=sums))


@torch.no_grad()
def inspect_current(root, run, label):
    """Separate sensing/forecast coordinates, with no new learner updates."""
    from torch.nn import functional as F
    from experiments.robocasa_stable_scaling.train import build as current_build, observation_codes, decode_feature_change
    cfg = json.loads((run/"config.json").read_text())
    torch.set_num_threads(2); torch.set_float32_matmul_precision("high")
    model = current_build(root, cfg).cuda().eval().requires_grad_(False)
    from vforge.ops import OpsConfig, initialize_ops
    initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False), device=torch.device("cuda"), parallel_state=None)
    ready = json.loads((run/label/"ready.json").read_text()); weights = run/label/"weights.pt"
    assert sha(weights) == ready["weights_sha256"]
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    before = common.state_hash(model.state_dict()); assert before == ready["nn_sha256"]
    records = [r for r in json.loads((root/"data.json").read_text())["records"] if r["split"] == "validation"]
    assert len(records) == 96
    sums = dict(latent_actual=0., latent_zero=0., latent_copy=0.,
        feature_actual=0., feature_zero=0., feature_observed=0., feature_copy=0.)
    for row in records:
        base = Path("/tmp/spt-time-compare-20261112"); ident = row["id"]
        features = np.load(base/"features/vjepa"/(ident+".npy"), mmap_mode="r")
        with np.load(base/"observations"/(ident+".npz")) as stream: commands = stream["commands"].copy()
        dense = torch.from_numpy(np.stack([features[s:s+8].transpose(1, 0, 2, 3) for s in (0, 9)]).copy()).cuda().flatten(0, 1).float()
        actions = np.zeros((2, 7, 14), np.float32)
        for i, start in enumerate((0, 9)):
            actions[i, :, 7:10] = commands[start:start+7, :3]/.3
            actions[i, :, 13] = -commands[start:start+7, 6]
        actions = torch.from_numpy(np.repeat(actions[:, None], 2, axis=1).reshape(4, 7, 14)).cuda()
        if cfg.get("joint_views"): actions=actions.reshape(2,2,7,14)[:,0]
        phi = F.adaptive_avg_pool2d(dense.reshape(-1, 14, 14, 1664).permute(0, 3, 1, 2), (4, 4)).flatten(1).reshape(4, 8, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = observation_codes(model,dense).float()
            predicted = model.predict(z[:, :-1], model.action_encoder(actions)).float()
            zero = model.predict(z[:, :-1], model.action_encoder(torch.zeros_like(actions))).float()
            if hasattr(model,"future_decoder"):
                def decode(value):
                    return decode_feature_change(model,value,phi[:, :1])
                actual_phi = decode(predicted-z[:, :1])
                zero_phi = decode(zero-z[:, :1])
                observed_phi = decode(z[:, 1:]-z[:, :1])
        sums["latent_actual"] += float((predicted-z[:, 1:]).square().sum())
        sums["latent_zero"] += float((zero-z[:, 1:]).square().sum())
        sums["latent_copy"] += float((z[:, :-1]-z[:, 1:]).square().sum())
        if hasattr(model,"future_decoder"):
            target = phi[:, 1:]-phi[:, :1]
            for key, value in [("feature_actual", actual_phi), ("feature_zero", zero_phi), ("feature_observed", observed_phi)]:
                sums[key] += float((value-target).abs().sum())
            sums["feature_copy"] += float((phi[:, 1:]-phi[:, :-1]).abs().sum())
    result = dict(passed=True, variant=cfg["variant"], label=label, source_nn_sha256=before,
        validation_episodes=96, windows_per_episode=2, new_SGD=0, model_unchanged=True,
        action_input_is_actual_command=True, zero_action_is_not_observed_physics=True,
        latent_units_change_across_checkpoints=True, not_a_control_result=True,
        actual_latent_MSE_over_copy=sums["latent_actual"]/sums["latent_copy"],
        zero_latent_MSE_over_copy=sums["latent_zero"]/sums["latent_copy"], sums=sums)
    if hasattr(model,"future_decoder"):
        result.update({key+"_L1_over_copy": sums[key]/sums["feature_copy"]
            for key in ("feature_actual", "feature_zero", "feature_observed")})
        result["feature_units_fixed_across_checkpoints"] = True
    assert common.state_hash(model.state_dict()) == before
    save(root/"current-diagnostics"/run.name/(label+".json"), result)


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
    p.add_argument("--run", type=Path); p.add_argument("--decoder-span", action="store_true")
    a = p.parse_args()
    if a.decoder_span:
        assert a.run is not None
        inspect_decoder_span(a.root, a.run, a.label)
    elif a.run is None: inspect(a.root, a.label)
    else: inspect_current(a.root, a.run, a.label)
