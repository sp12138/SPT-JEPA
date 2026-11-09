"""Kitchen-grouped factual RGB transitions; state is used only to form controls.

The 14-D displacement convention is RoboJEPA's franka_omron_state_delta:
base xyz/quaternion, relative EE xyz/quaternion xyz, first gripper finger.
No state, task, reward, success, goal, or action-prediction target is returned.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.robocasa_compare.acquire import TASKS, save, sha

CAMERAS = ["robot0_agentview_left", "robot0_eye_in_hand"]
STRIDE = 4
MAX_WINDOW = 16
TARGET_KITCHENS = {(k, k) for k in range(1, 11)}


def split_for(kitchen):
    assert tuple(kitchen) not in TARGET_KITCHENS, ("Test kitchen in training source", kitchen)
    assert all(11 <= int(value) <= 60 for value in kitchen), ("Not an official pretrain layout/style", kitchen)
    value = int(hashlib.sha256(json.dumps(list(kitchen)).encode()).hexdigest()[:8], 16)
    return "validation" if value % 10 == 0 else "train"


def inventory(root):
    records = []
    for task in TASKS:
        paths = list((root / "data" / task).glob("**/meta/info.json"))
        assert len(paths) == 1, (task, paths)
        dataset = paths[0].parent.parent
        info = json.loads(paths[0].read_text())
        assert info["fps"] == 20
        for line in (dataset / "meta/episodes.jsonl").read_text().splitlines():
            episode = json.loads(line)
            index = episode["episode_index"]
            extra = dataset / "extras" / f"episode_{index:06d}/ep_meta.json"
            meta = json.loads(extra.read_text())
            kitchen = [int(meta["layout_id"]), int(meta["style_id"])]
            split = split_for(kitchen)
            # All arms use the same current-window support, including the long arm.
            starts = max(0, episode["length"] - STRIDE * (MAX_WINDOW - 1))
            records.append(dict(task=task, episode=index, dataset=str(dataset), kitchen=kitchen,
                                split=split, native_length=episode["length"], starts=starts,
                                metadata_sha256=sha(extra), eligible=starts > 0))
    train = {tuple(r["kitchen"]) for r in records if r["split"] == "train"}
    valid = {tuple(r["kitchen"]) for r in records if r["split"] == "validation"}
    assert train and valid and not train & valid
    assert not (train | valid) & TARGET_KITCHENS
    result = dict(passed=True, records=records, cameras=CAMERAS, stride=STRIDE,
                  max_window=MAX_WINDOW, target_kitchens=sorted(TARGET_KITCHENS),
                  train_kitchens=len(train), validation_kitchens=len(valid),
                  episodes=len(records), eligible_episodes=sum(r["eligible"] for r in records),
                  no_target_kitchen_in_training_or_validation=True,
                  split_rule="SHA256(layout,style)%10==0 validation; all other pretrain kitchens train",
                  control_convention="RoboJEPA14D actual state displacement; no proprio input")
    save(root / "inventory.json", result)
    return result


def read_video(path, length):
    import av
    import cv2
    cv2.setNumThreads(1)
    images = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            image = frame.to_ndarray(format="rgb24")
            images.append(cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA).transpose(2, 0, 1))
    assert len(images) == length, (str(path), len(images), length)
    return np.stack(images)


def pack_task(root, task):
    import pyarrow.parquet as pq
    manifest = json.loads((root / "inventory.json").read_text())
    records = [dict(r) for r in manifest["records"] if r["task"] == task]
    out = root / "packed" / task
    out.mkdir(parents=True, exist_ok=False)
    total = sum(r["native_length"] for r in records)
    pixels = np.lib.format.open_memmap(out / "pixels.npy", mode="w+", dtype=np.uint8,
                                      shape=(total, 2, 3, 224, 224))
    poses = np.lib.format.open_memmap(out / "poses.npy", mode="w+", dtype=np.float32, shape=(total, 14))
    offset = 0
    for r in records:
        dataset = Path(r["dataset"])
        index = r["episode"]
        length = r["native_length"]
        path = dataset / f"data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet"
        table = pq.read_table(path, columns=["observation.state", "frame_index", "timestamp"])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        assert state.shape == (length, 16) and np.isfinite(state).all()
        assert np.array_equal(np.asarray(table["frame_index"]), np.arange(length))
        assert np.allclose(np.asarray(table["timestamp"]), np.arange(length) / 20., atol=1e-4)
        # This preserves the published state-delta convention, including its
        # quaternion-sign limitation; no orientation correction is silently added.
        pose = np.concatenate([state[:, :13], state[:, 14:15]], axis=-1).astype(np.float32)
        videos = [dataset / f"videos/chunk-{index // 1000:03d}/observation.images.{cam}/episode_{index:06d}.mp4"
                  for cam in CAMERAS]
        for view, video in enumerate(videos):
            pixels[offset:offset + length, view] = read_video(video, length)
        poses[offset:offset + length] = pose
        r.update(offset=offset, parquet_sha256=sha(path), video_sha256=[sha(p) for p in videos])
        offset += length
    assert offset == total
    pixels.flush(); poses.flush()
    del pixels, poses
    save(out / "manifest.json", dict(passed=True, records=records, native_frames=total,
         cameras=CAMERAS, pixels_sha256=sha(out / "pixels.npy"), poses_sha256=sha(out / "poses.npy")))
    print(json.dumps(dict(task=task, packed=True, episodes=len(records), frames=total)), flush=True)


def finish(root):
    manifests = [json.loads((root / "packed" / t / "manifest.json").read_text()) for t in TASKS]
    values = []
    records = []
    for task, manifest in zip(TASKS, manifests):
        assert manifest["passed"]
        poses = np.load(root / "packed" / task / "poses.npy", mmap_mode="r")
        for r in manifest["records"]:
            records.append(dict(r, shard=task))
            if r["split"] == "train" and r["eligible"]:
                state = poses[r["offset"]:r["offset"] + r["native_length"]].astype(np.float64)
                values.append(state[STRIDE:] - state[:-STRIDE])
    controls = np.concatenate(values)
    mean = controls.mean(0)
    std = np.maximum(controls.std(0), 1e-5)
    card = dict(passed=True, records=records, action_mean=mean.tolist(), action_std=std.tolist(),
                action_dim=14, action_statistics_scope="eligible TRAIN episodes only; every valid four-native-step displacement",
                cameras=CAMERAS, stride=STRIDE, common_max_window=MAX_WINDOW,
                train_windows=sum(r["starts"] for r in records if r["split"] == "train"),
                validation_windows=sum(r["starts"] for r in records if r["split"] == "validation"),
                source_inventory_sha256=sha(root / "inventory.json"),
                prohibited_learner_fields=["proprioception", "state", "reward", "success", "task", "language", "goal", "policy_target"])
    save(root / "packed/dataset.json", card)
    print(json.dumps({k:card[k] for k in ["passed", "train_windows", "validation_windows", "action_dim"]}), flush=True)


class Windows:
    def __init__(self, root, window, seed, split="train"):
        self.root = Path(root)
        self.card = json.loads((self.root / "dataset.json").read_text())
        assert self.card["passed"] and window in [8, 16]
        self.window, self.seed = window, seed
        self.records = [r for r in self.card["records"] if r["split"] == split and r["eligible"]]
        self.ends = np.cumsum([r["starts"] for r in self.records])
        self.n = int(self.ends[-1])
        self.shards = {}
        self.mean = np.asarray(self.card["action_mean"], np.float32)
        self.std = np.asarray(self.card["action_std"], np.float32)

    def __len__(self):
        return self.n

    def __getitem__(self, sample_index):
        import torch
        # Counter-based stream: restart does not repeat completed data draws.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(sample_index)]))
        row = int(rng.integers(self.n))
        episode = int(np.searchsorted(self.ends, row, side="right"))
        record = self.records[episode]
        start = row - (int(self.ends[episode-1]) if episode else 0)
        task = record["shard"]
        if task not in self.shards:
            self.shards[task] = (np.load(self.root / task / "pixels.npy", mmap_mode="r"),
                                 np.load(self.root / task / "poses.npy", mmap_mode="r"))
        images, poses = self.shards[task]
        rows = record["offset"] + start + np.arange(self.window) * STRIDE
        assert rows[-1] < record["offset"] + record["native_length"]
        pixels = np.asarray(images[rows]).transpose(1, 0, 2, 3, 4).copy()
        action = np.zeros((self.window, 14), np.float32)
        action[:-1] = (np.diff(np.asarray(poses[rows]), axis=0) - self.mean) / self.std
        # The unknown action following the final image is never read or labelled.
        return dict(pixels=torch.from_numpy(pixels), action=torch.from_numpy(action))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["inventory", "pack", "finish"], required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    args = parser.parse_args()
    if args.mode == "inventory":
        inventory(args.root)
    elif args.mode == "pack":
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda t: pack_task(args.root, t), args.tasks))
    else:
        finish(args.root)
