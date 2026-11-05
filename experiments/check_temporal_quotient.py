"""CPU algebra + exact-image revisit audit for a temporal-target control.

This checks a concrete target change, including its unresolved cross-clip
inconsistency. No approximation threshold, augmentation or semantic labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from as_jepa_tools.benchmark_data import BenchmarkWindows


def assignment(labels):
    unique = list(dict.fromkeys(labels))
    return np.array([[float(label == key) for key in unique] for label in labels])


def quotient_kernel(labels, rho):
    s = assignment(labels)
    n = len(labels)
    temporal = rho**abs(np.arange(n)[:, None]-np.arange(n)[None, :])
    compressed = s.T @ temporal @ s
    inv_scale = 1/np.sqrt(np.diag(compressed))
    quotient = (compressed*inv_scale[:, None])*inv_scale[None, :]
    return s @ quotient @ s.T, temporal, s


def alias_lower_bound(temporal, s):
    # Columns corresponding to one identical encoded observation must coincide.
    projected = (temporal @ s / s.sum(0)[None, :]) @ s.T
    return float(np.mean((temporal-projected)**2))


def controls():
    labels = ["A", "B", "A", "C"]
    k, temporal, s = quotient_kernel(labels, .5)
    assert np.linalg.eigvalsh(k).min() > -1e-12
    assert np.allclose(np.diag(k), 1) and np.allclose(k[0], k[2])
    assert alias_lower_bound(temporal, s) > 0
    raw, _, _ = quotient_kernel(["A", "B", "C"], .5)
    assert np.allclose(raw, .5**abs(np.arange(3)[:, None]-np.arange(3)[None, :]))
    fast, _, _ = quotient_kernel(["A", "B"], .5)
    slow, _, _ = quotient_kernel(["A", "C", "D", "E", "B"], .5)
    assert fast[0, -1] != slow[0, -1]
    return dict(local_alias_lower_bound=alias_lower_bound(temporal, s),
                quotient_eigenvalues=np.linalg.eigvalsh(k).tolist(),
                different_path_same_observation_pair_targets=[float(fast[0, -1]), float(slow[0, -1])],
                global_pair_inconsistency_squared_error_floor=float((fast[0, -1]-slow[0, -1])**2/2))


def audit(root, count):
    result = {}
    for split, seed in [("train", 20261030), ("validation", 20261031)]:
        ds = BenchmarkWindows(root, window=8, split=split); ds.open()
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(ds.episodes), min(count, len(ds.episodes)), replace=False)
        rows = []
        for i in chosen:
            start = int(rng.integers(ds.counts[i]))
            ep = int(ds.episodes[i]); first = int(ds.offsets[ep])+start
            frames = np.asarray(ds.file["pixels"][first:first+7*ds.stride+1])[::ds.stride]
            labels = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
            n = len(labels); s = assignment(labels)
            row = dict(episode=int(ds.episode_ids[ep]), start=start, unique_images=s.shape[1],
                       repeated_pairs=sum(labels[a] == labels[b] for a in range(n) for b in range(a+1, n)),
                       nonadjacent_repeated_pairs=sum(labels[a] == labels[b] for a in range(n) for b in range(a+2, n)))
            row["alias_lower_bound_by_illustrative_rho"] = {}
            for rho in (.5, .9):
                _, temporal, _ = quotient_kernel(labels, rho)
                row["alias_lower_bound_by_illustrative_rho"][str(rho)] = alias_lower_bound(temporal, s)
            rows.append(row)
        result[split] = dict(count=len(rows), clips_with_repeated_images=sum(r["repeated_pairs"] > 0 for r in rows),
                             mean_unique_images=float(np.mean([r["unique_images"] for r in rows])),
                             total_repeated_pairs=sum(r["repeated_pairs"] for r in rows),
                             rows=rows)
    assert not {r["episode"] for r in result["train"]["rows"]}.intersection(r["episode"] for r in result["validation"]["rows"])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/efs/prescottsun/SPT-JEPA/data/hardtasks-v1")
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = dict(scope="Algebra and exact-image duplicates only; no model training or physical labels",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), controls=controls(), tasks={})
    for task in ("tworoom", "cube"):
        result["tasks"][task] = audit(Path(args.root)/f"{task}-metadata", args.count)
        print(task, {k: {a: b for a, b in v.items() if a != "rows"} for k, v in result["tasks"][task].items()}, flush=True)
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
