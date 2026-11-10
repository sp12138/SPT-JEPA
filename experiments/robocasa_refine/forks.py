"""Goal-free, repeated-initial-state control experiments in TRAIN kitchens.

Distinct factual outcomes from the same observation rule out an action-ignorant
lookup of each observed state. No model score selects kitchens or trajectories.
Physical state is recorded only in the separate reset-replay proof, never data.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_compare.campaign import command, environment
from experiments.robocasa_compare import evaluate as base
from experiments.robocasa_refine import intervene

HOSTS = ("dev13", "dev15", "dev6")


def initialize(root, source):
    card = json.loads((source / "interventions/dataset.json").read_text())
    pairs = sorted({tuple(x["kitchen"]) for x in card["records"]},
        key=lambda x: hashlib.sha256(b"fork-v1:" + json.dumps(x).encode()).hexdigest())[:64]
    split = {tuple(x["kitchen"]): x["split"] for x in card["records"]}
    assert len(pairs) == 64 and sum(split[x] == "validation" for x in pairs) >= 3
    groups = [dict(id=f"l{p[0]}-s{p[1]}-r{s}", kitchen=p, seed=s, split=split[p])
              for p in pairs for s in (9401, 9402)]
    assert not (root / "fork-plan.json").exists()
    save(root / "fork-plan.json", dict(source=str(source), groups=groups, branches=8, steps=16,
        policy="Eight scrambled Sobol spherical first commands; independent isotropic uniform-radius suffixes",
        maxnorm=.3, current_controls_only=True, no_goal_reward_success_or_model_based_selection=True,
        all_kitchens_parent_TRAIN=True, original_validation_and_target_kitchens_excluded=True,
        physics_only_in_separate_technical_replay_proofs=True, automatic_training=False))


def episode(root, group, branch):
    from scipy.stats import qmc
    base.install_public(root)
    env = intervene.make_env(group["kitchen"], group["seed"])
    label = group["id"] + f"-b{branch}"
    try:
        obs = base.reset(env, "push", group["seed"], "init")
        initial = intervene.pixels(obs)
        physical = base.physics(env)
        canonical = root / "reset-proofs" / (group["id"] + ".npz")
        if branch == 0:
            canonical.parent.mkdir(exist_ok=True)
            assert not canonical.exists()
            np.savez_compressed(canonical, pixels=initial, **physical)
            errors = dict(qpos=0., qvel=0., time=0.)
            rgb_error = 0.
        else:
            first = np.load(canonical)
            errors = {k: float(np.max(np.abs(physical[k]-first[k]))) for k in ("qpos", "qvel", "time")}
            assert max(errors.values()) <= 1e-8, errors
            rgb_error = float(np.mean((initial.astype(float)-first["pixels"].astype(float))**2))
            assert rgb_error < 5., rgb_error
            # One observed root image conditions every factual branch. The
            # branch's own raw render was checked before this canonicalization.
            initial = first["pixels"].copy()
        policy_seed = int.from_bytes(hashlib.sha256(group["id"].encode()).digest()[:4], "little")
        points = qmc.Sobol(d=3, scramble=True, seed=policy_seed).random_base2(3)
        u = points[branch]
        z, theta, radius = 2*u[0]-1, 2*np.pi*u[1], .3*u[2]
        first_action = radius*np.array([np.sqrt(1-z*z)*np.cos(theta), np.sqrt(1-z*z)*np.sin(theta), z])
        rng = np.random.default_rng(np.random.SeedSequence([20261110, *group["kitchen"], group["seed"], branch, 65]))
        images, poses, commands = [initial], [base.native_training_pose(env)], []
        for index in range(16):
            action = np.zeros(7, np.float32)
            if index == 0:
                action[:3] = first_action
            else:
                direction = rng.normal(size=3); direction /= np.linalg.norm(direction)
                action[:3] = direction * rng.uniform(0., .3)
            obs, _, _, _ = env.step(action)
            images.append(intervene.pixels(obs)); poses.append(base.native_training_pose(env)); commands.append(action)
        out = root / "transitions" / (label + ".npz")
        out.parent.mkdir(exist_ok=True); assert not out.exists()
        np.savez_compressed(out, pixels=np.stack(images), commands=np.stack(commands), achieved=np.diff(np.stack(poses), axis=0))
        save(out.with_suffix(".json"), dict(group, passed=True, id=label, fork_id=group["id"], branch=branch,
            path=str(out), sha256=sha(out), frames=17, transitions=16,
            physical_replay_errors=errors, initial_rgb_mse=rgb_error,
            shared_initial_RGB="Observed branch0 after strict physical/RGB replay gate",
            no_absolute_state_goal_reward_success_or_future_action_target=True))
    finally:
        env.close()


def collect(root, host):
    plan = json.loads((root / "fork-plan.json").read_text())
    selected = [g for i, g in enumerate(plan["groups"]) if i % 3 == HOSTS.index(host)]
    (root / "collection-logs").mkdir(exist_ok=True)
    def one(gpu):
        rows = []
        for group in selected[gpu::8]:
            for branch in range(8):
                label = group["id"] + f"-b{branch}"
                output = root / "transitions" / (label + ".json")
                argv = [str(root / "environment/bin/python"), "-u", "-m", "experiments.robocasa_refine.forks",
                    "--root", str(root), "--mode", "episode", "--group-json", json.dumps(group), "--branch", str(branch)]
                try:
                    assert not output.exists()
                    command(argv, root / "collection-logs" / (label + ".log"), environment(root, str(gpu), True), root / "code", 1200)
                    row = json.loads(output.read_text()); assert row["passed"] and sha(row["path"]) == row["sha256"]
                except Exception as error:
                    row = dict(passed=False, fork_id=group["id"], branch=branch, error=repr(error), method_negative=False)
                    save(root / "collection-errors" / (label + ".json"), row)
                rows.append(row)
        return rows
    with ThreadPoolExecutor(max_workers=8) as pool: rows = sum(pool.map(one, range(8)), [])
    save(root / f"COLLECTION_DONE-{host}.json", dict(disposed=True, rows=rows))


def finish(root):
    plan = json.loads((root / "fork-plan.json").read_text())
    records, groups, incomplete = [], [], []
    for group in plan["groups"]:
        paths = [root / "transitions" / (group["id"] + f"-b{b}.json") for b in range(8)]
        if not all(p.exists() for p in paths):
            incomplete.append(group["id"]); continue
        rows = [json.loads(p.read_text()) for p in paths]
        assert all(x["passed"] and sha(x["path"]) == x["sha256"] for x in rows)
        assert all(max(x["physical_replay_errors"].values()) <= 1e-8 and x["initial_rgb_mse"] < 5. for x in rows)
        records.extend(rows); groups.append(dict(**group, record_ids=[x["id"] for x in rows]))
    assert len(groups) >= 64 and len({tuple(x["kitchen"]) for x in groups}) >= 32, (len(groups), len(incomplete))
    source = Path(plan["source"])
    old = json.loads((source / "interventions/dataset.json").read_text())
    card = dict(passed=True, records=records, groups=groups, branches=8, window=16, fixed_start=0,
        incomplete_groups=incomplete, transitions=len(records)*16,
        action_mean=old["action_mean"], action_std=old["action_std"],
        normalizer_scope="Original TRAIN parent convention; no new state-derived features",
        no_goal_reward_success_or_future_action_targets=True,
        partial_groups_excluded_only_for_predeclared_shared_root_contract=True)
    save(root / "interventions/dataset.json", card)
    print(json.dumps(dict(complete_groups=len(groups), episodes=len(records), transitions=card["transitions"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", choices=["initialize", "collect", "episode", "finish"], required=True)
    p.add_argument("--source", type=Path); p.add_argument("--host", choices=HOSTS)
    p.add_argument("--group-json"); p.add_argument("--branch", type=int)
    a = p.parse_args()
    if a.mode == "initialize": initialize(a.root, a.source)
    elif a.mode == "collect": collect(a.root, a.host)
    elif a.mode == "episode": episode(a.root, json.loads(a.group_json), a.branch)
    else: finish(a.root)
