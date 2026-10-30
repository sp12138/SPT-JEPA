"""Train-only covariance metrics and an objective-only native-planner adapter."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

METRICS = ("identity", "observation", "increment")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def tensor_digest(value):
    array = value.detach().cpu().contiguous().numpy()
    h = hashlib.sha256(str((array.shape, array.dtype)).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def fit_whitener(samples):
    """Fixed Ledoit-Wolf formula; no cross-validation or task-dependent knobs."""
    from sklearn.covariance import LedoitWolf
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("Need finite samples x features")
    estimator = LedoitWolf(assume_centered=False, store_precision=False).fit(values)
    covariance = estimator.covariance_
    eigen, vectors = np.linalg.eigh(covariance)
    if eigen[-1] <= 0:
        raise ValueError("A constant representation cannot define a whitening metric")
    # Only a numerical relative floor, tied to FP64 eigensolver precision.
    floor = np.finfo(np.float64).eps * values.shape[1] * eigen[-1]
    used = np.maximum(eigen, floor)
    whitening = (vectors * used ** -.5) @ vectors.T
    return dict(mean=estimator.location_, covariance=covariance, whitening=whitening,
                eigenvalues=eigen, effective_eigenvalues=used), dict(
        samples=len(values), dimensions=values.shape[1], shrinkage=float(estimator.shrinkage_),
        target_scale=float(np.trace(np.cov(values, rowvar=False, bias=True)) / values.shape[1]),
        eigenvalue_min=float(eigen[0]), eigenvalue_max=float(eigen[-1]),
        effective_condition_number=float(used[-1] / used[0]), eigenvalue_floor=float(floor),
        floored_eigenvalues=int((eigen < floor).sum()),
        formula="C=(1-lambda)*centered_empirical_covariance+lambda*trace(S)/d*I; W=C^(-1/2)")


def fit_metrics(clips):
    clips = np.asarray(clips, dtype=np.float64)
    if clips.ndim != 3 or clips.shape[1] < 2:
        raise ValueError("Expected clips x frames x state dimensions")
    samples = dict(observation=clips.reshape(-1, clips.shape[-1]),
                   increment=np.diff(clips, axis=1).reshape(-1, clips.shape[-1]))
    arrays, reports = {}, {}
    for name, value in samples.items():
        matrices, reports[name] = fit_whitener(value)
        arrays.update({name + "_" + key: matrix for key, matrix in matrices.items()})
    return arrays, reports


def make_manifest(dataset, count=128, seed=1042):
    if count > len(dataset.episodes):
        raise ValueError("Insufficient distinct training episodes")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(dataset.episodes), count, replace=False)
    windows = []
    for i in selected:
        index = (int(dataset.ends[i-1]) if i else 0) + int(rng.integers(dataset.counts[i]))
        ep, start = dataset.locate(index)
        windows.append(dict(index=index, episode=int(dataset.episode_ids[ep]), episode_array_index=ep,
            start=start, rows=(int(dataset.offsets[ep])+start+np.arange(dataset.window)*dataset.stride).tolist()))
    result = dict(split="train", count=count, selection_seed=seed, window=dataset.window,
        frameskip=dataset.stride, dataset_card_sha256=digest(dataset.root/"dataset_card.json"),
        metadata_sha256={name:digest(dataset.root/name) for name in ["offsets.npy", "lengths.npy",
            "episode_ids.npy", "train_episodes.npy", "validation_episodes.npy"]}, windows=windows)
    verify_manifest(dataset, result)
    return result


def verify_manifest(dataset, manifest):
    assert manifest["split"] == "train" and manifest["count"] == 128
    assert manifest["window"] == dataset.window and manifest["frameskip"] == dataset.stride
    assert manifest["dataset_card_sha256"] == digest(dataset.root/"dataset_card.json")
    for name, value in manifest["metadata_sha256"].items():
        assert value == digest(dataset.root/name), name
    train = set(np.load(dataset.root/"train_episodes.npy").tolist())
    val = set(np.load(dataset.root/"validation_episodes.npy").tolist())
    assert not train.intersection(val)
    seen = set()
    for row in manifest["windows"]:
        ep, start = dataset.locate(row["index"])
        assert row["episode_array_index"] == ep and row["episode"] == int(dataset.episode_ids[ep])
        assert row["start"] == start
        assert row["rows"] == (int(dataset.offsets[ep])+start+np.arange(dataset.window)*dataset.stride).tolist()
        assert row["episode"] in train and row["episode"] not in val and row["episode"] not in seen
        seen.add(row["episode"])
    assert len(seen) == manifest["count"]


class MetricRecorder:
    """All three costs on each identical bank; no extra model or physical rollout."""
    def __init__(self, output):
        self.output = Path(output)
        self.costs, self.orders, self.rows = [], [], []
        self.manual_identity_max_error = 0.

    def record(self, info, scores, difference, native_error):
        values = torch.stack([scores[name].detach() for name in METRICS]).cpu().numpy()
        orders = np.argsort(values, axis=-1, kind="stable").astype(np.int32)
        self.costs.append(values)
        self.orders.append(orders)
        row = dict(call=len(self.rows), shape=list(values.shape),
            action_candidates_sha256=tensor_digest(info["action_candidates"]),
            goal_emb_sha256=tensor_digest(info["goal_emb"]),
            endpoint_difference_sha256=tensor_digest(difference),
            selected_indices={name:values[i].argmin(-1).tolist() for i,name in enumerate(METRICS)})
        for key in ["pixels", "action_history"]:
            if key in info and torch.is_tensor(info[key]):
                row[key+"_sha256"] = tensor_digest(info[key][:, :1])
        self.rows.append(row)
        self.manual_identity_max_error = max(self.manual_identity_max_error, float(native_error))

    def finish(self):
        if not self.rows:
            raise AssertionError("No native candidate banks were scored")
        # CEM has fixed sample count; concatenate call batches, keeping offsets.
        counts = [value.shape[1] for value in self.costs]
        np.savez_compressed(self.output/"metric-costs.npz", costs=np.concatenate(self.costs, axis=1),
            stable_orders=np.concatenate(self.orders, axis=1), call_offsets=np.r_[0,np.cumsum(counts)])
        (self.output/"metric-candidate-banks.json").write_text(json.dumps(self.rows,indent=2)+"\n")
        return dict(calls=len(self.rows), candidate_batches=sum(counts), metric_order=list(METRICS),
            manual_identity_max_abs_error=self.manual_identity_max_error,
            costs_sha256=digest(self.output/"metric-costs.npz"),
            candidate_banks_sha256=digest(self.output/"metric-candidate-banks.json"),
            extra_model_rollouts=0, scope="Every row scores the same predicted bank under all three metrics. Across evaluation arms, subsequent CEM populations and physical trajectories may differ.")


class MetricGoal(nn.Module):
    """Only endpoint cost changes; encoder, raw-state rollout and goal encode stay native."""
    def __init__(self, native, matrices, selected="identity", recorder=None):
        super().__init__()
        if selected not in METRICS:
            raise ValueError(selected)
        self.native, self.selected, self.recorder = native, selected, recorder
        for name in METRICS[1:]:
            self.register_buffer(name+"_whitening", torch.as_tensor(matrices[name+"_whitening"],dtype=torch.float32))

    def forward(self, info):
        native_cost = self.native(info)
        pred = info[self.native.pred_key]
        goal = info[self.native.goal_key][:, None, -1:].expand_as(pred)
        delta = (pred[:, :, -1:] - goal[:, :, -1:].detach()).flatten(2)
        scores = {"identity":native_cost}
        with torch.autocast(device_type=delta.device.type, enabled=False):
            for name in METRICS[1:]:
                transformed = delta.float() @ getattr(self, name+"_whitening")
                squared = transformed.square()
                scores[name] = squared.sum(-1) if self.native.reduction == "sum" else squared.mean(-1)
        if not all(torch.isfinite(value).all() for value in scores.values()):
            raise ValueError("Nonfinite metric candidate cost")
        manual = delta.square().sum(-1) if self.native.reduction == "sum" else delta.square().mean(-1)
        if self.recorder:
            self.recorder.record(info, scores, delta, (manual-native_cost).abs().max())
        # No recomputation, casting, scaling or matrix multiplication on identity.
        return scores[self.selected]


@contextmanager
def install_objective(matrices, selected, recorder=None):
    from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
    original = ShootingCostEvaluator.__init__
    records = []
    def initialize(self, model, objective, *args, **kwargs):
        if type(objective) is not GoalMSE:
            raise TypeError("This diagnostic only supports the native GoalMSE objective")
        adapted = MetricGoal(objective, matrices, selected, recorder).to(next(model.parameters()).device)
        records.append(dict(class_name=type(objective).__name__, pred_key=objective.pred_key,
                            goal_key=objective.goal_key, reduction=objective.reduction))
        original(self, model, adapted, *args, **kwargs)
    ShootingCostEvaluator.__init__ = initialize
    try:
        yield records
    finally:
        ShootingCostEvaluator.__init__ = original
