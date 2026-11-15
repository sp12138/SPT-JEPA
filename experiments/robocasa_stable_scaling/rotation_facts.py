"""Ordinary TRAIN-only rotational interventions, with no goal or policy targets.

Native Reach ignores commanded rotations and can return before advancing the
simulator for zero XYZ. This collection-only controller keeps its workspace,
base lock and OSC hold, but gives that hold the actually requested rotation
and waits for both position and orientation. Evaluation remains unmodified.
Kinematics is private controller feedback; only RGB and current commands are
saved as experience. Frozen readout predictions are diagnostics, never actions
or training labels. This deliberately changes the collection interface.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from experiments.robocasa_action_probe.probe import Policy
from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_compare.evaluate import install_public, reset
from experiments.robocasa_refine.intervene import pixels
from experiments.robocasa_stable_scaling.stream import make_env


def step_rotation(env, command):
    """Blocking relative base-frame SE(3) input; no task goal is consulted."""
    if not np.any(command[3:6]):
        observation, _, _, _ = env.step(command)
        return observation
    from robots.robocasa.robot_env import (
        _GOAL_OSC_OUTPUT_MAX, _REACH_WORKSPACE_R, _layoutA_to_robosuite12,
        ACTION_PER_STEP_DIM,
    )
    inner = env._env
    _, rotation_world_base = inner._get_base_world_pose()
    target = inner._get_eef_pos()+rotation_world_base@command[:3]
    home = np.asarray(inner._reach_home_eef_pos)
    offset = target-home; distance = np.linalg.norm(offset)
    if distance > _REACH_WORKSPACE_R:
        target = home+offset*(_REACH_WORKSPACE_R/distance)
    current = Rotation.from_quat(inner._get_eef_quat_world())
    desired = Rotation.from_rotvec(rotation_world_base@command[3:6])*current
    inner._reach_home_eef_quat = desired.as_quat()
    for _ in range(env._max_steps):
        error = target-inner._get_eef_pos()
        angular_error = (desired*Rotation.from_quat(inner._get_eef_quat_world()).inv()).magnitude()
        # Declared controller accuracy, not a learned loss or success threshold.
        if np.linalg.norm(error) < env._pos_tol and angular_error < .001:
            break
        sub = np.zeros(ACTION_PER_STEP_DIM, np.float64)
        sub[5:8] = np.clip(rotation_world_base.T@error/_GOAL_OSC_OUTPUT_MAX[:3], -1., 1.)
        sub[11] = command[6]
        sub = inner._reach_apply_orientation_hold(sub)
        inner._env.step(_layoutA_to_robosuite12(sub))
        if inner._lock_base_in_arm_mode:
            inner._restore_base_joints()
    return inner._build_obs()


def collect(root, lane):
    install_public(root)
    torch.set_num_threads(2); torch.set_float32_matmul_precision("highest")
    policy = Policy(root, "spt64-linear")
    card = json.loads((root/"data.json").read_text())["records"]
    pairs = sorted({tuple(r["kitchen"]) for r in card if r["split"] == "train"})
    assert len(pairs) >= 4 and 0 <= lane < 4
    pair = pairs[lane]
    for replicate in range(2):
        seed = 72000+lane*10+replicate
        rng = np.random.default_rng(seed)
        directions = rng.normal(size=(8, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        # Three declared angular scales, never chosen using control outcomes.
        magnitudes = np.asarray([.005, .05, .5])[np.arange(8) % 3]
        forward = directions*magnitudes[:, None]
        rotation_actions = np.zeros((16, 7), np.float32)
        rotation_actions[:8, 3:6] = forward
        rotation_actions[8:, 3:6] = -forward[::-1]
        initial = None
        for mode in ("no-op", "rotation"):
            name = f"lane-{lane}-rep-{replicate}-{mode}"
            directory = root/"records"/name
            if (directory/"ready.json").exists():
                proof = json.loads((directory/"ready.json").read_text())
                assert proof["passed"] and sha(directory/"experience.npz") == proof["experience_sha256"]
                if initial is None:
                    with np.load(directory/"experience.npz") as stream:
                        initial = stream["pixels"][0].copy()
                continue
            assert not directory.exists(), "Preserve incomplete outputs; repair explicitly"
            actions = rotation_actions if mode == "rotation" else np.zeros_like(rotation_actions)
            env = make_env(pair, seed, "reach"); began = time.time()
            try:
                observation = reset(env, "reach", seed, "init")
                images = [pixels(observation)]
                if initial is None:
                    initial = images[0].copy()
                initial_rgb_difference = float(np.mean((images[0].astype(float)-initial.astype(float))**2))
                def code(obs):
                    return policy.encode({k: obs[k] for k in ("img_0", "img_1")}).detach().cpu().numpy()
                codes = [code(observation)]
                for command in actions:
                    observation = step_rotation(env, command)
                    images.append(pixels(observation)); codes.append(code(observation))
            finally:
                env.close()
            assert policy.unchanged()
            images = np.stack(images); codes = np.stack(codes)
            assert images.shape == (17, 2, 3, 224, 224) and codes.shape == (17, 128)
            assert np.all(actions[:, :3] == 0) and np.isfinite(codes).all()
            directory.mkdir(parents=True)
            raw = directory/"experience.npz"
            np.savez_compressed(raw, pixels=images, commands=actions)
            # Own frozen visual codes are a cache. Predicted actions stay separate.
            np.save(directory/"frozen_codes.npy", codes)
            weight = policy.weight.detach().cpu().numpy()
            predictions = (codes[1:]-codes[:-1])@weight.T
            np.save(directory/"diagnostic_predictions_NEVER_TARGETS.npy", predictions)
            save(directory/"ready.json", dict(passed=True, lane=lane, replicate=replicate,
                mode=mode, kitchen=pair, seed=seed, split="train", records=16,
                experience_sha256=sha(raw), code_sha256=sha(directory/"frozen_codes.npy"),
                parent=policy.readout.parent, all_learning_weights_unchanged=True,
                current_XYZ_inputs_are_zero=True, no_reward_state_goal_or_expert_in_experience=True,
                no_learned_policy_controls_collection=True, no_new_training=True,
                separate_physical_episodes_not_assumed_paired=True,
                initial_rgb_mse_vs_noop=initial_rgb_difference,
                controller_angular_tolerance_rad=.001, angular_scales_rad=[.005, .05, .5],
                orientation_controller_differs_from_native_Reach=True,
                mean_diagnostic_false_XYZ_norm_mm=float(np.linalg.norm(predictions, axis=-1).mean()*1000),
                elapsed_seconds=time.time()-began))
            print(json.dumps(dict(name=name, passed=True, false_XYZ_norm_mm=float(
                np.linalg.norm(predictions, axis=-1).mean()*1000))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lane", type=int, required=True)
    args = parser.parse_args()
    collect(args.root, args.lane)
