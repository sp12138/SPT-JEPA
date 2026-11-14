"""Goal-free multiscale factual logs with an explicit physical kitchen split."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_compare.evaluate import install_public, reset
from experiments.robocasa_refine.intervene import make_env, pixels
from experiments.robocasa_refine.encoder_ablation import load_encoder, encode


def atomic_array(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".partial")
    with temp.open("wb") as stream:
        np.save(stream, value); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def episode(root, encoder, index, lane):
    import cv2
    card = json.loads((root/"data.json").read_text())
    split = json.loads((root/"plan.json").read_text()).get("collection_split", "train")
    assert split in ("train", "validation")
    kitchens = sorted({tuple(r["kitchen"]) for r in card["records"] if r["split"] == split})
    assert kitchens
    pair = kitchens[(index//8+lane*7) % len(kitchens)]; seed = 51000+index
    ident = f"stream-{index:09d}"; directory = root/"stream"
    proof_path = directory/"records"/(ident+".json")
    if proof_path.exists(): return json.loads(proof_path.read_text())
    rng = np.random.default_rng(np.random.SeedSequence([20261114, index]))
    directions = rng.normal(size=(8, 3)); directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = np.exp(rng.uniform(np.log(.003), np.log(.3), size=(8, 1)))
    forward = directions*radii
    actions = np.zeros((16, 7), np.float32)
    actions[:8, :3] = forward; actions[8:, :3] = -forward[::-1]
    started = time.time(); env = make_env(pair, seed)
    try:
        observation = reset(env, "push", seed, "init")
        images = [pixels(observation)]
        for action in actions:
            # Only RGB is consumed. Reward, success, goal and kinematic state
            # returned by the simulator are neither read nor logged.
            observation, _, _, _ = env.step(action)
            images.append(pixels(observation))
        images = np.stack(images)
    finally:
        env.close()
    assert images.shape == (17, 2, 3, 224, 224) and images.dtype == np.uint8
    dense = encode(encoder, torch.from_numpy(images).flatten(0, 1).cuda(), "vjepa")
    features = dense.reshape(17, 2, 196, 1664).cpu().numpy()
    assert np.isfinite(features).all() and features.dtype == np.float16
    render = np.stack([cv2.resize(x.transpose(1, 2, 0), (64, 64), interpolation=cv2.INTER_AREA)
        .transpose(2, 0, 1) for x in images.reshape(-1, 3, 224, 224)]).reshape(17, 2, 3, 64, 64)
    feature_path = directory/"features/vjepa"/(ident+".npy")
    obs_path = directory/"observations"/(ident+".npz")
    rgb_path = directory/"rgb"/(ident+".npz")
    atomic_array(feature_path, features)
    obs_path.parent.mkdir(parents=True, exist_ok=True); rgb_path.parent.mkdir(parents=True, exist_ok=True)
    for path, values in [(obs_path, dict(render=render, commands=actions)), (rgb_path, dict(pixels=images))]:
        temp = path.with_suffix(path.suffix+".partial")
        with temp.open("wb") as stream: np.savez_compressed(stream, **values)
        temp.replace(path)
    row = dict(passed=True, id=ident, index=index, lane=lane, kitchen=list(pair), seed=seed,
        split=split, frames=17, transitions=16, features=str(feature_path), observations=str(obs_path),
        rgb=str(rgb_path), features_sha256=sha(feature_path), observations_sha256=sha(obs_path),
        rgb_sha256=sha(rgb_path), input_fields=["RGB", "issued_current_commands"],
        no_goal_reward_state_or_expert=True, no_future_or_optimal_action_labels=True,
        policy="isotropic log-uniform .003..0.3 translation; eight random increments then their reversed negatives",
        maximum_action_norm=.3, loop_is_not_assumed_physically_closed=True,
        started_unix=started, completed_unix=time.time())
    save(proof_path, row)
    return row


def collect(root, lane, lanes=8, index_base=0, cohort="base", max_episodes=None):
    assert 0 <= lane < lanes and index_base >= 0
    torch.set_num_threads(2); torch.set_float32_matmul_precision("high")
    install_public(root); encoder = load_encoder(root, "vjepa")
    versions = [p._version for p in encoder.parameters()]
    index = index_base+lane; completed = 0; failures = 0
    progress = root/"stream"/(f"PROGRESS-{lane}.json" if cohort == "base" else f"PROGRESS-{cohort}-{lane}.json")
    while not (root/"STOP").exists() and not (root/"stream"/("STOP-"+cohort)).exists():
        if max_episodes is not None and completed >= max_episodes: break
        try:
            row = episode(root, encoder, index, lane)
            completed += 1; failures = 0
            assert versions == [p._version for p in encoder.parameters()]
            assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
            save(progress, dict(completed=completed, cohort=cohort, lanes=lanes, index_base=index_base,
                latest_id=row["id"], latest_index=index, time=time.time(), encoder_frozen=True))
            print(json.dumps(dict(id=row["id"], seconds=row["completed_unix"]-row["started_unix"])), flush=True)
        except Exception as error:
            failures += 1
            save(root/"stream/errors"/f"index-{index:09d}.json", dict(error=repr(error),
                traceback=traceback.format_exc(), lane=lane, index=index, method_negative=False))
            if failures >= 3: raise
        index += lanes
    save(progress.with_name("DONE-"+progress.name.removeprefix("PROGRESS-")),
        dict(passed=True, cohort=cohort, lane=lane, committed_episodes=completed, time=time.time()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--lane", type=int, required=True); p.add_argument("--lanes", type=int, default=8)
    p.add_argument("--index-base", type=int, default=0); p.add_argument("--cohort", default="base")
    p.add_argument("--max-episodes", type=int)
    a = p.parse_args(); collect(a.root, a.lane, a.lanes, a.index_base, a.cohort, a.max_episodes)
