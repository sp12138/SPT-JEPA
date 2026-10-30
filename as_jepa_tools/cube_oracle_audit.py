"""Frozen Cube audit: physical coverage, encoded-goal metric, rollout ranking.

The recorded-action family and physical costs are diagnostic oracles only.
This does not train a policy, alter a checkpoint, or claim benchmark success.
All large artifacts belong on EFS. Run collection shards before scoring models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def setup(a):
    import hdf5plugin  # noqa: F401 -- register dataset compression
    import h5py
    from clear_lewm.datasets import metadata_fingerprint
    from clear_lewm.protocols import protocol_from_dict

    data = Path(a.data)
    card = json.loads((data / "dataset_card.json").read_text())
    manifest = json.loads(Path(a.manifest).read_text())
    assert manifest["task"] == "cube"
    assert metadata_fingerprint(card["h5_path"]) == manifest["dataset"]["fingerprint"]["value"]
    allowed = set(np.load(data / (manifest["campaign_split"] + "_episodes.npy")).tolist())
    pairs = manifest["pairs"][:a.limit] if a.limit else manifest["pairs"]
    assert all(p["episode_id"] in allowed for p in pairs)
    pairs = [(i, p) for i, p in enumerate(pairs) if i % a.shards == a.shard]
    assert 0 <= a.shard < a.shards
    Path(a.output).mkdir(parents=True, exist_ok=True)
    return card, manifest, protocol_from_dict(manifest["protocol"]), pairs, h5py.File(card["h5_path"], "r")


def candidate_actions(recorded, mean, std, count, seed):
    """Fixed raw-action library, independent of evaluated model.

    Gaussian candidates use the training-normalized CEM initial distribution.
    There is no clipping, matching the existing SWM CEM->environment interface.
    The last candidate exactly repeats recorded actions to audit state restore.
    """
    assert count >= 4
    rng = np.random.default_rng(seed)
    values = [recorded.copy(), np.zeros_like(recorded)]
    labels = ["recorded_oracle", "zero"]
    n_random = (count - 3) // 2
    for _ in range(n_random):
        values.append(rng.normal(size=recorded.shape) * std + mean)
        labels.append("gaussian")
    for i in range(count - 3 - n_random):
        scale = 0.25 if i % 2 == 0 else 1.0
        values.append(recorded + scale * std * rng.normal(size=recorded.shape))
        labels.append("recorded_perturb_" + str(scale))
    values.append(recorded.copy())
    labels.append("recorded_repeat")
    return np.asarray(values, dtype=np.float32), labels


def collect(a):
    import mujoco
    import stable_worldmodel as swm
    from clear_lewm.runner import _install_task_success
    from clear_lewm.tasks import cube_symmetry_angle_deg

    card, manifest, spec, pairs, h5 = setup(a)
    horizon = int(spec.goal_offset)
    stride = int(card["frameskip"])
    assert horizon % stride == 0 and horizon == 25
    world = swm.World("swm/OGBCube-v0", num_envs=1, max_episode_steps=2 * spec.eval_budget,
                      image_shape=(224, 224), env_type="single", ob_type="states",
                      multiview=False, width=224, height=224, visualize_info=False,
                      terminate_at_goal=False)
    _install_task_success(world, "cube", spec)
    env = world.envs.envs[0].unwrapped
    summary = []
    try:
        for case, pair in pairs:
            started = time.time()
            row, end = int(pair["start_row"]), int(pair["goal_row"])
            assert end - row == horizon
            start_pixel, goal_pixel = h5["pixels"][row], h5["pixels"][end]
            goal_pos = h5["privileged_block_0_pos"][end]
            goal_quat = h5["privileged_block_0_quat"][end]
            recorded = h5["action"][row:end]
            actions, labels = candidate_actions(recorded, np.asarray(card["action_mean"]),
                                                np.asarray(card["action_std"]), a.candidates,
                                                a.seed + 1009 * case)
            for directory in a.proposals:
                for path in sorted(Path(directory).glob(f"*-case{case:04d}.npz")):
                    with np.load(path) as proposal:
                        assert str(proposal["manifest_sha256"].item()) == sha(a.manifest)
                        actions = np.concatenate([actions, proposal["actions"][None]], axis=0)
                        labels.append("cem_" + str(proposal["label"].item()))
            world.reset(seed=[a.seed + case])
            env.set_state(h5["qpos"][row], h5["qvel"][row])
            env.set_target_pos(0, goal_pos, goal_quat)
            state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
            state = np.empty(mujoco.mj_stateSize(env._model, state_spec))
            mujoco.mj_getState(env._model, env._data, state, state_spec)
            pixels, positions, quaternions, effectors, holds, successes, qposes = [], [], [], [], [], [], []
            for sequence in actions:
                mujoco.mj_setState(env._model, env._data, state, state_spec)
                mujoco.mj_forward(env._model, env._data)
                env._reset_next_step = False
                env._clear_lewm_hold_count = 0
                env._success = False
                cpix, cpos, cquat, ceff, chold, csucc, cqpos = [], [], [], [], [], [], []
                for t in range(horizon + 1):
                    if t:
                        env.step(sequence[t - 1])
                    info = env.compute_ob_info()
                    cpos.append(info["privileged/block_0_pos"])
                    cquat.append(info["privileged/block_0_quat"])
                    ceff.append(info["proprio/effector_pos"])
                    chold.append(int(env._clear_lewm_hold_count))
                    csucc.append(bool(env._success))
                    cqpos.append(env._data.qpos.copy())
                    if t % stride == 0:
                        cpix.append(np.asarray(env.render()).copy())
                pixels.append(cpix); positions.append(cpos); quaternions.append(cquat)
                effectors.append(ceff); holds.append(chold); successes.append(csucc); qposes.append(cqpos)
            positions = np.asarray(positions)
            quaternions = np.asarray(quaternions)
            pixels = np.asarray(pixels, dtype=np.uint8)
            qposes = np.asarray(qposes)
            pos_err = np.linalg.norm(positions - goal_pos, axis=-1)
            ori_err = cube_symmetry_angle_deg(quaternions.reshape(-1, 4),
                                              np.broadcast_to(goal_quat, quaternions.shape).reshape(-1, 4)).reshape(pos_err.shape)
            physical_cost = np.maximum(pos_err / spec.cube_position_threshold_m,
                                       ori_err / spec.cube_orientation_threshold_deg)
            repeat_idx = labels.index("recorded_repeat")
            restore_error = float(np.max(np.abs(qposes[0] - qposes[repeat_idx])))
            assert restore_error < 1e-7, f"State restore not deterministic: {restore_error}"
            initial_rgb_mse = float(np.mean((pixels[0, 0].astype(float) - start_pixel) ** 2))
            replay_pos_error = float(np.linalg.norm(positions[0, -1] - goal_pos))
            replay_qpos_error = float(np.max(np.abs(qposes[0, -1] - h5["qpos"][end])))
            target = Path(a.output) / f"case{case:04d}.npz"
            assert not target.exists(), f"Refusing to overwrite {target}"
            np.savez_compressed(target, actions=actions, family=np.asarray(labels), pixels=pixels,
                                start_pixels=start_pixel, goal_pixels=goal_pixel,
                                physical_cost=physical_cost, position_error_m=pos_err,
                                orientation_error_deg=ori_err, positions=positions, quaternions=quaternions,
                                effector_positions=np.asarray(effectors), hold_count=np.asarray(holds),
                                success=np.asarray(successes), qpos=qposes, case_index=case,
                                manifest_sha256=sha(a.manifest), pair_json=json.dumps(pair),
                                block_steps=np.arange(0, horizon + 1, stride),
                                recorded_rgb=h5["pixels"][row:end + 1:stride])
            record = dict(case=case, candidates=len(actions), seconds=time.time()-started,
                          initial_rgb_mse=initial_rgb_mse, recorded_endpoint_pos_error_m=replay_pos_error,
                          recorded_endpoint_qpos_max_error=replay_qpos_error,
                          duplicate_restore_max_qpos_error=restore_error,
                          recorded_success=bool(np.asarray(successes)[0].any()),
                          any_candidate_success=bool(np.asarray(successes).any()), file=str(target))
            summary.append(record)
            print(json.dumps(record), flush=True)
            dump(Path(a.output) / f"collection-shard{a.shard:02d}.json", dict(
                manifest_sha256=sha(a.manifest), script_sha256=sha(__file__), records=summary,
                args=vars(a), scope="Frozen 25-step open-loop diagnostic, not 50-step replanning benchmark."))
    finally:
        h5.close()
        world.close()


def load_model(a, card):
    import torch
    import stable_worldmodel as swm
    from vforge.ops import OpsConfig, initialize_ops
    initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False), device=torch.device("cuda"), parallel_state=None)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    model = swm.wm.utils.load_pretrained(a.checkpoint).cuda().eval()
    model.requires_grad_(False)
    own = json.loads((Path(a.checkpoint) / "config.json").read_text()).get("_target_", "").startswith("as_jepa_lewm.")
    if own:
        mean, std = np.asarray(card["action_mean"]), np.asarray(card["action_std"])
    else:
        # Reproduce the released-reference full-data StandardScaler exactly.
        from sklearn.preprocessing import StandardScaler
        ds = swm.data.load_dataset(card["h5_path"], keys_to_cache=["action"])
        vals = ds.get_col_data("action")
        scaler = StandardScaler().fit(vals[np.isfinite(vals).all(axis=1)])
        mean, std = scaler.mean_, scaler.scale_
    return model, mean, std


def transform_pixels(pixels):
    import torch
    from clear_lewm.runner import _image_transform
    shape = pixels.shape
    tensor = torch.as_tensor(np.array(pixels, copy=True)).reshape(-1, *shape[-3:]).permute(0, 3, 1, 2)
    return _image_transform(224)(tensor).cuda().reshape(*shape[:-3], 3, 224, 224)


def propose(a):
    """Native CEM first-plan proposals, optionally added to the common library."""
    import gymnasium as gym
    import torch
    import stable_worldmodel as swm
    card, manifest, spec, pairs, h5 = setup(a)
    model, mean, std = load_model(a, card)
    from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
    from stable_worldmodel.planning.solver import CEMSolver
    cost = ShootingCostEvaluator(model, GoalMSE())
    try:
        for case, pair in pairs:
            pixels = transform_pixels(h5["pixels"][int(pair["start_row"]):int(pair["start_row"])+1])[None]
            goal = transform_pixels(h5["pixels"][int(pair["goal_row"]):int(pair["goal_row"])+1])[None]
            solver = CEMSolver(cost, num_samples=300, n_steps=30, topk=30, device="cuda", seed=a.seed + 1009*case)
            solver.configure(action_space=gym.spaces.Box(-1, 1, (1, 5)), n_envs=1,
                             config=SimpleNamespace(action_block=5, horizon=5))
            with torch.inference_mode():
                # The shared goal encoder removes the current-action key. The
                # rollout replaces this placeholder with the first candidate.
                result = solver.solve({"pixels": pixels, "goal": goal,
                                       "action": torch.zeros(1, 1, 25, device="cuda")})
            chosen = result["actions"]
            assert tuple(chosen.shape) == (1, 5, 25), chosen.shape
            assert bool(torch.isfinite(chosen).all()), "Nonfinite CEM actions"
            # Mirror StandardScaler.inverse_transform on float32 policy actions:
            # separate in-place multiply/add preserve its rounding and dtype.
            raw = chosen.detach().cpu().numpy()[0].reshape(25, 5).copy()
            raw *= std
            raw += mean
            target = Path(a.output) / f"{a.label}-case{case:04d}.npz"
            assert not target.exists(), target
            np.savez_compressed(target, actions=raw.astype(np.float32), label=a.label,
                                manifest_sha256=sha(a.manifest), checkpoint=a.checkpoint,
                                first_plan_only=True, solver_seed=a.seed + 1009*case,
                                solver_samples=300, solver_iterations=30, solver_elites=30,
                                unclipped_action_fraction=float(np.mean(np.abs(raw)>1)),
                                script_sha256=sha(__file__))
            print(json.dumps(dict(case=case, proposal=str(target))), flush=True)
    finally:
        h5.close()


def corr(x, y):
    from scipy.stats import spearmanr
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else None


def score(a):
    import torch
    card, manifest, spec, pairs, h5 = setup(a)
    h5.close()
    model, mean, std = load_model(a, card)
    weights_sha256 = sha(Path(a.checkpoint) / "weights.pt")
    records = []
    with torch.inference_mode():
        for case, pair in pairs:
            path = Path(a.pool) / f"case{case:04d}.npz"
            with np.load(path) as data:
                assert str(data["manifest_sha256"].item()) == sha(a.manifest)
                actions = data["actions"]
                n, raw_t, ad = actions.shape
                norm = torch.as_tensor(((actions-mean)/std).reshape(n, raw_t//5, ad*5), dtype=torch.float32, device="cuda")
                start = transform_pixels(data["start_pixels"][None])[None, None].expand(1, n, 1, -1, -1, -1)
                goal_z = model.encode({"pixels": transform_pixels(data["goal_pixels"][None])[None]})["emb"][0, 0]
                pred = model.rollout({"pixels": start}, norm[None])["predicted_emb"][0]
                true_parts = []
                for lo in range(0, n, a.encode_batch):
                    true_parts.append(model.encode({"pixels": transform_pixels(data["pixels"][lo:lo+a.encode_batch])})["emb"])
                true = torch.cat(true_parts, 0)
                predicted_cost = ((pred-goal_z)**2).sum(-1).cpu().numpy()
                true_cost = ((true-goal_z)**2).sum(-1).cpu().numpy()
                model_error = ((pred-true)**2).mean(-1).cpu().numpy()
                true_numpy = true.cpu().numpy()
                family = data["family"]
                physical = data["physical_cost"]
                pos = data["position_error_m"]
                ori = data["orientation_error_deg"]
                success = data["success"].any(1)
                # Compare action effects after removing each model's shared drift.
                delta_pred = pred[:, -1]-pred[0, -1]
                delta_true = true[:, -1]-true[0, -1]
                effect_cos = torch.nn.functional.cosine_similarity(delta_pred, delta_true, dim=-1).cpu().numpy()
                groups = {"all": np.arange(n), "nonoracle": np.flatnonzero(np.isin(family, ["zero", "gaussian"]) | np.char.startswith(family, "cem_")),
                          "oracle_neighborhood": np.flatnonzero(np.char.startswith(family, "recorded"))}
                for name, ids in groups.items():
                    if not len(ids):
                        continue
                    phys_idx = int(ids[np.argmin(physical[ids, -1])])
                    enc_idx = int(ids[np.argmin(true_cost[ids, -1])])
                    pred_idx = int(ids[np.argmin(predicted_cost[ids, -1])])
                    selections = {}
                    for key, idx in [("physical_oracle", phys_idx), ("encoded_oracle", enc_idx), ("model", pred_idx)]:
                        selections[key] = dict(index=idx, family=str(family[idx]), position_error_m=float(pos[idx, -1]),
                                               orientation_error_deg=float(ori[idx, -1]), normalized_pose_cost=float(physical[idx, -1]),
                                               sustained_success=bool(success[idx]))
                    records.append(dict(case=case, group=name, count=len(ids),
                        candidate_any_success=bool(success[ids].any()), selections=selections,
                        physical_encoded_rank=corr(physical[ids, -1], true_cost[ids, -1]),
                        encoded_predicted_rank=corr(true_cost[ids, -1], predicted_cost[ids, -1]),
                        physical_predicted_rank=corr(physical[ids, -1], predicted_cost[ids, -1]),
                        position_encoded_rank=corr(pos[ids, -1], true_cost[ids, -1]),
                        orientation_encoded_rank=corr(ori[ids, -1], true_cost[ids, -1]),
                        rollout_mse_by_block=model_error[ids].mean(0).tolist(),
                        candidate_latent_variance_by_block=np.var(true_numpy[ids], axis=0).mean(-1).tolist(),
                        centered_action_effect_mse=float(np.mean((
                            (pred[ids, -1].cpu().numpy()-pred[ids, -1].cpu().numpy().mean(0))
                            -(true_numpy[ids, -1]-true_numpy[ids, -1].mean(0)))**2)),
                        action_effect_cosine=float(np.mean(effect_cos[ids[ids != 0]])) if np.any(ids != 0) else None,
                        metric_regret=float(physical[enc_idx, -1]-physical[phys_idx, -1]),
                        signed_dynamics_regret=float(physical[pred_idx, -1]-physical[enc_idx, -1])))
                target = Path(a.output) / f"scores-case{case:04d}.npz"
                assert not target.exists(), target
                np.savez_compressed(target, predicted_cost=predicted_cost, encoded_cost=true_cost,
                                    rollout_mse=model_error, action_effect_cosine=effect_cos,
                                    true_embeddings=true_numpy, predicted_embeddings=pred.cpu().numpy(),
                                    family=family, physical_cost=physical, position_error_m=pos, orientation_error_deg=ori)
            dump(Path(a.output) / f"scores-shard{a.shard:02d}.json", dict(
                checkpoint=a.checkpoint, checkpoint_weights_sha256=weights_sha256,
                manifest_sha256=sha(a.manifest), script_sha256=sha(__file__), records=records,
                args=vars(a), scope="Fixed 25-step candidate library. Oracle families use held-out recorded actions only for diagnosis. First-plan ranking is not closed-loop benchmark success."))
            print(json.dumps(dict(case=case, scored=n, checkpoint=a.checkpoint)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["collect", "score", "propose"])
    p.add_argument("--manifest", required=True); p.add_argument("--data", required=True)
    p.add_argument("--output", required=True); p.add_argument("--checkpoint")
    p.add_argument("--pool"); p.add_argument("--label", default="model")
    p.add_argument("--shard", type=int, default=0); p.add_argument("--shards", type=int, default=1)
    p.add_argument("--limit", type=int); p.add_argument("--candidates", type=int, default=32)
    p.add_argument("--seed", type=int, default=20261030); p.add_argument("--encode-batch", type=int, default=8)
    p.add_argument("--proposals", action="append", default=[])
    a = p.parse_args()
    if a.mode in {"score", "propose"} and not a.checkpoint:
        p.error("score/propose requires --checkpoint")
    if a.mode == "score" and not a.pool:
        p.error("score requires --pool")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    {"collect": collect, "score": score, "propose": propose}[a.mode](a)


if __name__ == "__main__":
    main()
