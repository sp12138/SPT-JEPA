"""Compare position readout before/after QueryPool in a frozen visual encoder.

Only explicit portable weights are loaded. Physical states train diagnostic
linear probes; neither model weights nor a policy are updated. Random projection
matrices are fixed across checkpoints, tasks and data splits.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from as_jepa_lewm.model import ASJepaLeWM
from as_jepa_lewm.robust import RobustLeWM
from as_jepa_tools.benchmark_data import BenchmarkWindows
from vforge.ops import OpsConfig, initialize_ops


PROJECTION_SEED = 20261030
PROJECTION_DIM = 64
RIDGE = .001


def episode_indices(dataset, count, seed):
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(dataset.episodes), min(count, len(dataset.episodes)), replace=False)
    return [(int(dataset.ends[i - 1]) if i else 0) + int(rng.integers(dataset.counts[i])) for i in selected]


class FixedProjection:
    def __init__(self):
        self.matrices = {}

    def matrix(self, dim):
        if dim not in self.matrices:
            # Input dimension determines a separate fixed stream. CLS and mean
            # have identical width and therefore share the exact same matrix.
            rng = np.random.default_rng(np.random.SeedSequence([PROJECTION_SEED, dim]))
            self.matrices[dim] = (rng.standard_normal((dim, PROJECTION_DIM)) / math.sqrt(PROJECTION_DIM)).astype(np.float32)
        return self.matrices[dim]

    def metadata(self):
        return {str(dim): dict(shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest())
                for dim, value in sorted(self.matrices.items())}


@torch.no_grad()
def collect(model, cfg, split, episodes, batch_size, seed, projection):
    dataset = BenchmarkWindows(cfg["data"], window=model.window, split=split)
    ids = episode_indices(dataset, episodes, seed)
    loader = DataLoader(Subset(dataset, ids), batch_size=batch_size, num_workers=0)
    features, labels, episode_ids, starts = {}, [], [], []
    label_keys = None
    for batch in loader:
        dataset.open()
        if label_keys is None:
            label_keys = [key for key in ["pos_agent", "pos_target", "privileged_block_0_pos", "proprio_effector_pos"] if key in dataset.file]
            if not label_keys:
                raise ValueError("No supported physical labels for diagnostic probes")
        rows = batch["rows"].numpy()
        labels.append(np.concatenate([np.stack([np.asarray(dataset.file[key][row]) for row in rows]) for key in label_keys], axis=-1))
        pixels = model.normalize(batch["pixels"].cuda())
        b, w = pixels.shape[:2]
        tokens = model.encoder(pixels.flatten(0, 1), interpolate_pos_encoding=True).last_hidden_state
        patches = tokens[:, 1:]
        side = math.isqrt(patches.shape[1])
        if side * side != patches.shape[1]:
            raise ValueError("Expected a square patch grid after removing one CLS token")
        spatial = F.adaptive_avg_pool2d(patches.transpose(1, 2).reshape(b * w, patches.shape[-1], side, side), (4, 4))
        # Position-major ordering retains the 4x4 grid; no global averaging.
        spatial = spatial.permute(0, 2, 3, 1).flatten(1)
        query = model.state_bn(model.pool(tokens).reshape(b * w, model.state_dim))
        batch_features = dict(query_pool=query, cls=tokens[:, 0], patch_mean=patches.mean(1), spatial4x4=spatial)
        for name, value in list(batch_features.items()):
            if name != "query_pool":
                matrix = torch.from_numpy(projection.matrix(value.shape[-1])).to(value.device)
                batch_features[name + "_rp64"] = value @ matrix
        for name, value in batch_features.items():
            features.setdefault(name, []).append(value.reshape(b, w, -1).float().cpu().numpy())
        episode_ids.extend(batch["episode"].tolist())
        starts.extend(batch["start"].tolist())
    return {k: np.concatenate(v) for k, v in features.items()}, np.concatenate(labels), dict(episodes=episode_ids, starts=starts, label_keys=label_keys)


def score_predictions(prediction, truth, train_mean):
    error = ((prediction - truth) ** 2).mean(0)
    reference = ((truth - train_mean) ** 2).mean(0)
    valid = reference > 1e-12
    r2 = 1 - error / np.maximum(reference, 1e-20)
    return dict(mse=error.tolist(), r2_against_train_mean=[float(v) if ok else None for v, ok in zip(r2, valid)],
                mean_r2=float(r2[valid].mean()) if valid.any() else None,
                informative_target_dimensions=int(valid.sum()))


def ridge_probe(train_x, train_y, test_x, test_y):
    x = train_x.reshape(-1, train_x.shape[-1]).astype(np.float64)
    tx = test_x.reshape(-1, test_x.shape[-1]).astype(np.float64)
    y = train_y.reshape(-1, train_y.shape[-1]).astype(np.float64)
    ty = test_y.reshape(-1, test_y.shape[-1]).astype(np.float64)
    mean, std = x.mean(0), x.std(0)
    std = np.maximum(std, 1e-12)
    x, tx = (x - mean) / std, (tx - mean) / std
    ym = y.mean(0)
    n, dim = x.shape
    if dim > n:
        kernel = x @ x.T / n
        dual = np.linalg.solve(kernel + RIDGE * np.eye(n), y - ym)
        test_prediction = (tx @ x.T / n) @ dual + ym
        train_prediction = kernel @ dual + ym
        solver = "dual"
    else:
        weights = np.linalg.solve(x.T @ x / n + RIDGE * np.eye(dim), x.T @ (y - ym) / n)
        test_prediction, train_prediction = tx @ weights + ym, x @ weights + ym
        solver = "primal"
    return dict(validation=score_predictions(test_prediction, ty, ym), train=score_predictions(train_prediction, y, ym),
                feature_dim=dim, fitted_parameters=(dim + 1) * y.shape[1], solver=solver, ridge=RIDGE,
                train_samples=n, validation_samples=len(tx), train_std_mean=float(std.mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Explicit portable weights file or directory containing weights.pt")
    parser.add_argument("--step", type=int, required=True, help="Explicit source global step, including for initial-model exports")
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe-episodes", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1042)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    started = time.time()
    config_path, checkpoint, out = Path(args.config), Path(args.checkpoint), Path(args.output)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "weights.pt"
    if checkpoint.name == "last.ckpt":
        raise ValueError("Use portable weights; full optimizer checkpoints are intentionally unsupported")
    cfg = json.loads(config_path.read_text())
    if cfg["model"].get("domain", "standard") != "standard":
        raise ValueError("This audit is defined for the standard benchmark pixel observations")
    initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False), device=torch.device("cuda"), parallel_state=None)
    model = (RobustLeWM if cfg.get("robust_variant") else ASJepaLeWM)(**cfg["model"]).cuda().eval()
    if model.pool is None:
        raise ValueError("This comparison requires the learned QueryPool model")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    projection = FixedProjection()
    train_x, train_y, train_info = collect(model, cfg, "train", args.probe_episodes, args.batch_size, args.seed, projection)
    test_x, test_y, test_info = collect(model, cfg, "validation", args.probe_episodes, args.batch_size, args.seed + 1, projection)
    assert train_info["label_keys"] == test_info["label_keys"]
    assert not set(train_info["episodes"]).intersection(test_info["episodes"])
    del model
    torch.cuda.empty_cache()
    results = {}
    for name, x in train_x.items():
        tx = test_x[name]
        results[name] = dict(
            clean=ridge_probe(x, train_y, tx, test_y),
            time_shifted=ridge_probe(np.roll(x, 3, axis=1), train_y, np.roll(tx, 3, axis=1), test_y),
            temporal_displacement=ridge_probe(x[:, 1:] - x[:, :1], train_y[:, 1:] - train_y[:, :1],
                                              tx[:, 1:] - tx[:, :1], test_y[:, 1:] - test_y[:, :1]))
    out.mkdir(parents=True, exist_ok=True)
    report = dict(config=str(config_path), config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  step=args.step, train=train_info, validation=test_info, projection_seed=PROJECTION_SEED,
                  projections=projection.metadata(), features=results, seconds=time.time() - started,
                  scope="Frozen visual encoder, fixed train-standardized linear ridge, physical labels only for diagnostic probes. Full spatial features have greater readout capacity; RP64 controls dimension but is not a learned optimal compression. Low linear R2 does not prove absent nonlinear information.")
    (out / "spatial-probe.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(checkpoint=str(checkpoint), step=args.step, seconds=report["seconds"],
                          clean_r2={k: v["clean"]["validation"]["r2_against_train_mean"] for k, v in results.items()})), flush=True)


if __name__ == "__main__":
    main()
