"""CPU-only diagnostic: can recorded future actions reveal the initial scene?

Physical labels are held-out audit targets, never model-training supervision.
Uses BenchmarkWindows' exact episode/window indexing without loading images.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from as_jepa_tools.benchmark_data import BenchmarkWindows


def episode_indices(dataset, count, seed):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(dataset.episodes), min(count, len(dataset.episodes)), replace=False)
    return [int(dataset.ends[i - 1] if i else 0) + int(rng.integers(dataset.counts[i])) for i in chosen]


def collect(root, split, count, seed):
    ds = BenchmarkWindows(root, window=8, split=split)
    ds.open()
    keys = [k for k in ("pos_agent", "pos_target", "privileged_block_0_pos", "proprio_effector_pos") if k in ds.file]
    features, initial, terminal, episodes, starts = [], [], [], [], []
    for index in episode_indices(ds, count, seed):
        ep, start = ds.locate(index)
        first = int(ds.offsets[ep]) + start
        last = first + 7 * ds.stride
        actions = (np.asarray(ds.actions[first:last], dtype=np.float64) - ds.mean) / ds.std
        features.append(actions)
        initial.append(np.concatenate([np.asarray(ds.file[k][first]).reshape(-1) for k in keys]))
        terminal.append(np.concatenate([np.asarray(ds.file[k][last]).reshape(-1) for k in keys]))
        episodes.append(int(ds.episode_ids[ep]))
        starts.append(start)
    return dict(actions=np.stack(features), initial=np.stack(initial), terminal=np.stack(terminal),
                episodes=episodes, starts=starts, keys=keys, frameskip=ds.stride)


def score(prediction, y, train_y):
    error = np.mean((prediction - y) ** 2, axis=0)
    denominator = np.mean((y - train_y.mean(0)) ** 2, axis=0)
    r2 = 1 - error / np.maximum(denominator, 1e-20)
    return dict(r2_against_train_mean=[float(r) if d > 1e-12 else None for r, d in zip(r2, denominator)],
                mse=error.tolist(), denominator=denominator.tolist())


def fit_predict(x, y, tx, kernel):
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-8)
    x, tx = (x - mean) / std, (tx - mean) / std
    ym = y.mean(0)
    if kernel == "linear":
        w = np.linalg.solve(x.T @ x / len(x) + .001 * np.eye(x.shape[1]), x.T @ (y - ym) / len(x))
        return tx @ w + ym
    d2 = np.maximum((x*x).sum(1)[:, None] + (x*x).sum(1)[None, :] - 2*x@x.T, 0)
    t2 = np.maximum((tx*tx).sum(1)[:, None] + (x*x).sum(1)[None, :] - 2*tx@x.T, 0)
    bandwidth = max(float(np.median(d2[np.triu_indices(len(x), 1)])), 1e-8)
    gram, test_gram = np.exp(-d2 / (2 * bandwidth)), np.exp(-t2 / (2 * bandwidth))
    w = np.linalg.solve(gram + .001 * len(x) * np.eye(len(x)), y - ym)
    return test_gram @ w + ym


def audit(root, count, seed):
    train, val = collect(root, "train", count, seed), collect(root, "validation", count, seed + 1)
    assert train["keys"] == val["keys"]
    assert not set(train["episodes"]).intersection(val["episodes"])
    out = dict(label_keys=train["keys"], train_episodes=train["episodes"], validation_episodes=val["episodes"],
               train_starts=train["starts"], validation_starts=val["starts"],
               frameskip=train["frameskip"], action_shape=list(train["actions"].shape[1:]), results={})
    variants = {"first_action": lambda a: a[:, 0], "last_action": lambda a: a[:, -1],
                "mean_action": lambda a: a.mean(1), "whole_future_action_sequence": lambda a: a.reshape(len(a), -1)}
    for name, transform in variants.items():
        x, tx = transform(train["actions"]), transform(val["actions"])
        for kernel in ("linear", "rbf"):
            row = {}
            for target in ("initial", "terminal", "displacement"):
                y = train[target] if target != "displacement" else train["terminal"] - train["initial"]
                ty = val[target] if target != "displacement" else val["terminal"] - val["initial"]
                row[target] = score(fit_predict(x, y, tx, kernel), ty, y)
            out["results"][f"{name}/{kernel}"] = row
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/efs/prescottsun/SPT-JEPA/data/hardtasks-v1")
    p.add_argument("--count", type=int, default=256)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = {"scope": "Diagnostic physical labels only; CPU; no training or GPUs.",
              "count_requested_each_split": args.count, "seed": 20261030,
              "tasks": {task: audit(Path(args.root) / f"{task}-metadata", args.count, 20261030)
                        for task in ("tworoom", "cube")}}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    for task, row in result["tasks"].items():
        print(task, row["label_keys"], len(row["train_episodes"]), len(row["validation_episodes"]))
        for name in ("first_action/linear", "whole_future_action_sequence/linear", "whole_future_action_sequence/rbf"):
            print(name, {k: [round(r, 4) if r is not None else None for r in v["r2_against_train_mean"]]
                         for k, v in row["results"][name].items()})


if __name__ == "__main__":
    main()
