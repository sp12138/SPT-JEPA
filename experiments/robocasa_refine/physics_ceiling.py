"""Protected state-assisted contact diagnostic, never a learner or baseline.

Uses the public controller's command bounds and episode length. A successful
trajectory establishes attainability for that case; failure proves no ceiling.
Neither traces nor its state-dependent controls may enter training or planning.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.robocasa_compare import evaluate as base
from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_refine import validation


def diagnose(root, kitchen, seed, height, output, policy="waypoint"):
    base.install_public(root)
    validation.ROOT = root
    env = validation.make_env("push", kitchen, seed)
    try:
        goal_obs = base.reset(env, "push", seed, "goal")
        obs = base.reset(env, "push", seed, "init")
        case = root / "cases" / (validation.name("push", kitchen, seed) + ".npz")
        cached = np.load(case)
        replay = base.verify_physics(env, cached, case.name)
        rgb_error = float(np.mean((np.stack([obs["img_0"], obs["img_1"]]).astype(float)
                                   - cached["initial"].astype(float)) ** 2))
        assert rgb_error < 5.
        base.goal_proof("push", goal_obs, obs)
        target = env._object_reach_target_world.copy()
        actual = env._env._env
        body = int(actual.obj_body_id["obj"])
        model = actual.sim.model
        object_bodies = {body}
        for _ in range(int(model.nbody)):
            object_bodies.update(i for i, parent in enumerate(model.body_parentid)
                                 if int(parent) in object_bodies)
        object_geoms = {i for i, b in enumerate(model.geom_bodyid) if int(b) in object_bodies}
        tick_contacts = []
        native_step = actual.step

        def step_with_contact_capture(action):
            result = native_step(action)
            contacts = []
            for contact in actual.sim.data.contact[:actual.sim.data.ncon]:
                pair = [int(contact.geom1), int(contact.geom2)]
                if any(g in object_geoms for g in pair):
                    names = [model.geom_id2name(g) or str(g) for g in pair]
                    contacts.append(dict(geoms=names, distance=float(contact.dist)))
            tick_contacts.append(dict(eef=env._env._get_eef_pos().tolist(),
                                      object=env._get_object_world_pos().tolist(), contacts=contacts))
            return result

        actual.step = step_with_contact_capture
        z_contact = float(env._push_step_z_min) + .003 + height
        commands, rows, frames = [], [], [[obs["img_0"], obs["img_1"]]]
        # Four-command cycles: approach behind, descend, establish contact,
        # push toward the target. All changes depend on protected state.
        waypoints = []
        for index in range(18):
            obj = env._get_object_world_pos().copy()
            if index % 4 == 0:
                direction = target[:2] - obj[:2]
                distance = float(np.linalg.norm(direction))
                direction /= max(distance, 1e-9)
                behind = obj[:2] - .07 * direction
                waypoints = [np.r_[behind, z_contact + .10], np.r_[behind, z_contact],
                             np.r_[obj[:2] + direction * min(.035, distance), z_contact],
                             np.r_[target[:2] + direction * .025, z_contact]]
            waypoint = waypoints[index % 4]
            if policy == "servo" and index >= 2:
                direction = target[:2] - obj[:2]
                distance = float(np.linalg.norm(direction))
                direction /= max(distance, 1e-9)
                waypoint = np.r_[obj[:2] + direction * (min(.025, distance) - .04), z_contact]
                if np.linalg.norm(obj - target) < .02:
                    waypoint = env._env._get_eef_pos().copy()
            delta = env._env._get_base_world_R().T @ (waypoint - env._env._get_eef_pos())
            norm = float(np.linalg.norm(delta))
            action = np.zeros(7, dtype=np.float32)
            action[:3] = delta * min(1., .3 / max(norm, 1e-9))
            assert np.linalg.norm(action[:3]) <= .300001 and not action[3:].any()
            first_tick = len(tick_contacts)
            obs, reward, done, info = env.step(action)
            distance = float(np.linalg.norm(env._get_object_world_pos() - target))
            rows.append(dict(macro_step=index + 1, goal_distance=distance, reward=float(reward),
                             eef=env._env._get_eef_pos().tolist(), object=env._get_object_world_pos().tolist(),
                             waypoint=waypoint.tolist(), ticks=len(tick_contacts) - first_tick,
                             object_contact_ticks=sum(bool(x["contacts"]) for x in tick_contacts[first_tick:]),
                             robot_contact_ticks=sum(any(any("gripper" in g or "robot0" in g for g in c["geoms"])
                                                         for c in x["contacts"]) for x in tick_contacts[first_tick:]),
                             native_substeps=info.get("blocking_n_substeps")))
            commands.append(action)
            frames.append([obs["img_0"], obs["img_1"]])
            if done:
                break
        output.mkdir(parents=True, exist_ok=False)
        trace = output / "protected-trace.npz"
        np.savez_compressed(trace, actions=np.asarray(commands), frames=np.asarray(frames))
        result = dict(passed=True, diagnostic_only=True, state_assisted=True,
                      prohibited_training_or_proposal_memory=True, kitchen=kitchen, seed=seed,
                      diagnostic_policy=policy,
                      contact_height_offset=height, z_contact=z_contact,
                      z_bounds=[env._push_step_z_min, env._push_step_z_max],
                      goal=target.tolist(), physical_replay_errors=replay, initial_rgb_mse=rgb_error,
                      case_sha256=sha(case), max_commands=18, maxnorm=.3, rotation_and_gripper_zero=True,
                      final_goal_distance=rows[-1]["goal_distance"],
                      final_task_complete=rows[-1]["goal_distance"] < .02,
                      any_task_complete=any(x["goal_distance"] < .02 for x in rows),
                      final_normalized_progress=rows[-1]["reward"], rows=rows,
                      trace_sha256=sha(trace),
                      conclusion_scope="Success establishes case attainability; failure is not unreachability proof")
        save(output / "summary.json", result)
        save(output / "protected-contacts.json", tick_contacts)
        print(json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--kitchen", type=int, required=True)
    p.add_argument("--seed", type=int, default=7401)
    p.add_argument("--height", type=float, default=0.)
    p.add_argument("--policy", choices=["waypoint", "servo"], default="waypoint")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    diagnose(a.root, a.kitchen, a.seed, a.height, a.output, a.policy)
