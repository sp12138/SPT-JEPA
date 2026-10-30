"""CPU-only predeclared falsification of low-order atomic image measures.

No learned segmentation, object masks, pixel thresholds, GPUs or task training.
Mixed 2D Hankel pencils are used; marginal roots are never claimed to identify
paired 2D points. Physical coordinates may only be read after numerical gates.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment, nnls

from as_jepa_tools.benchmark_data import BenchmarkWindows


BASES = {
    4: [(0, 0), (1, 0), (0, 1), (1, 1)],
    8: [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)],
}


def episode_indices(ds, count, seed):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(ds.episodes), min(count, len(ds.episodes)), replace=False)
    return [int(ds.ends[i-1] if i else 0)+int(rng.integers(ds.counts[i])) for i in chosen]


def image_measures(image):
    x = np.asarray(image, dtype=np.float64)
    if x.shape[-1] not in (3, 4):
        x = x.transpose(1, 2, 0)
    x = x[..., :3]/255.
    luminance = x @ np.array([.2126, .7152, .0722])
    gy, gx = np.gradient(luminance)
    chromaticity = x/np.maximum(x.sum(-1, keepdims=True), 1e-12)
    return {
        "luminance": luminance,
        "gradient_magnitude": np.sqrt(gx*gx+gy*gy),
        "chromatic_deviation": np.linalg.norm(chromaticity-chromaticity.mean((0, 1), keepdims=True), axis=-1),
    }


def measure_moments(measure):
    mass = float(measure.sum())
    if mass <= 1e-12:
        raise ValueError("zero measure")
    # Positive-exponent Fourier convention, coordinates pixel_index / size.
    spectrum = np.fft.fft2(measure).conj()/mass
    return {(x, y): spectrum[y, x] for x in range(9) for y in range(9)}


def node_distance(x, y):
    delta = np.angle(x[:, None, :]/y[None, :, :])/(2*np.pi)
    distance = np.linalg.norm(delta, axis=-1)
    i, j = linear_sum_assignment(distance)
    return float(np.sqrt(np.mean(distance[i, j]**2)))


def fit_pencil(m, order):
    basis = BASES[order]
    def matrix(shift):
        return np.array([[m[(a[0]+b[0]+shift[0], a[1]+b[1]+shift[1])]
                          for b in basis] for a in basis])
    h0, hx, hy = matrix((0, 0)), matrix((1, 0)), matrix((0, 1))
    condition = float(np.linalg.cond(h0))
    tx, ty = np.linalg.solve(h0, hx), np.linalg.solve(h0, hy)
    qx, vectors = np.linalg.eig(tx)
    transformed_y = np.linalg.solve(vectors, ty @ vectors)
    qy = np.diag(transformed_y)
    nodes = np.stack([qx, qy], -1)
    if not np.isfinite(nodes).all() or np.any(abs(nodes) < 1e-12):
        raise ValueError("nonfinite or zero spectral node")
    unit_nodes = nodes/abs(nodes)
    used = {(a[0]+b[0]+s[0], a[1]+b[1]+s[1])
            for a in basis for b in basis for s in [(0, 0), (1, 0), (0, 1)]}
    calibration = sorted(used)
    heldout = [p for p in sorted(m) if p not in used]
    def design(indices):
        return np.array([unit_nodes[:, 0]**p[0]*unit_nodes[:, 1]**p[1] for p in indices])
    a, target = design(calibration), np.array([m[p] for p in calibration])
    real_a, real_target = np.concatenate([a.real, a.imag]), np.concatenate([target.real, target.imag])
    weight, _ = nnls(real_a, real_target, maxiter=1000)
    predicted = design(heldout) @ weight
    truth = np.array([m[p] for p in heldout])
    commutator = np.linalg.norm(tx @ ty-ty @ tx)/(np.linalg.norm(tx)*np.linalg.norm(ty)+1e-20)
    y_offdiag = transformed_y-np.diag(qy)
    error = float(np.linalg.norm(predicted-truth)/max(np.linalg.norm(truth), 1e-12))
    # A fixed set of mixed moments without weights is an unordered address code.
    address = np.array([(unit_nodes[:, 0]**i*unit_nodes[:, 1]**j).mean()
                        for i in range(4) for j in range(4)])
    return dict(condition=condition, commutator_relative=float(commutator),
                simultaneous_diagonalization_relative=float(np.linalg.norm(y_offdiag)/max(np.linalg.norm(transformed_y), 1e-20)),
                node_radial_error_max=float(np.max(abs(abs(nodes)-1))),
                calibration_relative=float(np.linalg.norm(a @ weight-target)/max(np.linalg.norm(target), 1e-12)),
                heldout_relative=error, weight_sum=float(weight.sum()),
                min_weight=float(weight.min()), max_weight=float(weight.max()),
                zero_weight_count=int(np.sum(weight < 1e-12))), unit_nodes, np.concatenate([address.real, address.imag])


def audit_measure(measure, order):
    result, nodes, address = fit_pencil(measure_moments(measure), order)
    y, x = np.indices(measure.shape, dtype=np.float64)
    x, y = x/measure.shape[1], y/measure.shape[0]
    fields = {"x_cosine": 1+.1*np.cos(2*np.pi*x), "mixed_sine": 1+.1*np.sin(2*np.pi*(x+y))}
    result["amplitude_perturbation"] = {}
    for name, field in fields.items():
        other, perturbed_nodes, perturbed_address = fit_pencil(measure_moments(measure*field), order)
        result["amplitude_perturbation"][name] = dict(
            address_node_rms_fraction_image=node_distance(nodes, perturbed_nodes),
            address_feature_rms=float(np.sqrt(np.mean((address-perturbed_address)**2))),
            perturbed_heldout_relative=other["heldout_relative"])
    # Predeclared permissive numerical screen, never selected using task labels.
    result["wellposed"] = bool(result["condition"] <= 1e8 and result["commutator_relative"] <= .05
                               and result["node_radial_error_max"] <= .05 and result["heldout_relative"] <= .2
                               and max(v["address_node_rms_fraction_image"] for v in result["amplitude_perturbation"].values()) <= .02)
    return result, address


def toy_control():
    rng = np.random.default_rng(20261031)
    out = {}
    for order in (4, 8):
        positions = rng.uniform(.05, .95, (order, 2))
        nodes = np.exp(2j*np.pi*positions)
        weights = np.exp(rng.uniform(-1, 1, order)); weights /= weights.sum()
        m = {(i, j): np.sum(weights*nodes[:, 0]**i*nodes[:, 1]**j) for i in range(9) for j in range(9)}
        metrics, recovered, _ = fit_pencil(m, order)
        recovery_error = node_distance(nodes, recovered)
        assert recovery_error < 1e-7 and metrics["heldout_relative"] < 1e-6
        out[str(order)] = dict(recovery_error=recovery_error, **metrics)
    return out


def finite_summary(values):
    v = np.array([x for x in values if np.isfinite(x)])
    return dict(n=int(len(v)), quantiles={str(q): float(np.quantile(v, q)) for q in [0, .1, .5, .9, 1]}) if len(v) else dict(n=0)


def run_task(root, count):
    rows, features, labels, split_indices = [], {}, {}, {}
    keys = None
    for split, seed in [("train", 20261030), ("validation", 20261031)]:
        ds = BenchmarkWindows(root, window=8, split=split); ds.open()
        keys = [k for k in ("pos_agent", "pos_target", "privileged_block_0_pos", "proprio_effector_pos") if k in ds.file]
        split_indices[split] = []
        for index in episode_indices(ds, count, seed):
            ep, start = ds.locate(index)
            image_row = int(ds.offsets[ep])+start
            episode = int(ds.episode_ids[ep])
            identifier = f"{split}/{episode}/{start}"
            split_indices[split].append(identifier)
            label = np.concatenate([np.asarray(ds.file[k][image_row]).reshape(-1) for k in keys])
            # Labels are stored only for a potential gated diagnostic probe.
            labels[identifier] = label
            for measure_name, measure in image_measures(ds.file["pixels"][image_row]).items():
                for order in (4, 8):
                    row = dict(split=split, episode=episode, start=start, image_row=image_row,
                               measure=measure_name, order=order)
                    try:
                        result, feature = audit_measure(measure, order)
                        row.update(result)
                        features[(identifier, measure_name, order)] = feature
                    except (ValueError, np.linalg.LinAlgError, RuntimeError, FloatingPointError) as error:
                        row.update(wellposed=False, error=type(error).__name__+": "+str(error))
                    rows.append(row)
    train_ids = {r["episode"] for r in rows if r["split"] == "train"}
    val_ids = {r["episode"] for r in rows if r["split"] == "validation"}
    assert not train_ids.intersection(val_ids)
    summary = {}
    for measure in ("luminance", "gradient_magnitude", "chromatic_deviation"):
        for order in (4, 8):
            selected = [r for r in rows if r["measure"] == measure and r["order"] == order]
            key = f"{measure}/K{order}"
            summary[key] = dict(count=len(selected), failures=sum("error" in r for r in selected),
                                wellposed_count=sum(r["wellposed"] for r in selected))
            for metric in ("condition", "commutator_relative", "simultaneous_diagonalization_relative",
                           "node_radial_error_max", "calibration_relative", "heldout_relative", "zero_weight_count"):
                summary[key][metric] = finite_summary([r[metric] for r in selected if metric in r])
            for field in ("x_cosine", "mixed_sine"):
                summary[key][field+"_node_stability"] = finite_summary(
                    [r["amplitude_perturbation"][field]["address_node_rms_fraction_image"]
                     for r in selected if "amplitude_perturbation" in r])
            # No physical probe unless >=90% pass in EACH split, defined a priori.
            passed = {split: [r for r in selected if r["split"] == split and r["wellposed"]]
                      for split in ("train", "validation")}
            if any(len(passed[split]) < .9*len(split_indices[split]) for split in passed):
                summary[key]["physical_probe"] = "not run: numerical wellposed coverage below 90% in at least one split"
                continue
            def dataset(split):
                ids = [f"{split}/{r['episode']}/{r['start']}" for r in passed[split]]
                return np.stack([features[(s, measure, order)] for s in ids]), np.stack([labels[s] for s in ids])
            x, y = dataset("train"); tx, ty = dataset("validation")
            mean, scale = x.mean(0), np.maximum(x.std(0), 1e-8)
            x, tx = (x-mean)/scale, (tx-mean)/scale
            ym = y.mean(0)
            w = np.linalg.solve(x.T @ x/len(x)+.001*np.eye(x.shape[1]), x.T @ (y-ym)/len(x))
            error, ref = ((tx @ w+ym-ty)**2).mean(0), ((ty-ym)**2).mean(0)
            summary[key]["physical_probe"] = dict(
                r2=[float(1-e/r) if r > 1e-12 else None for e, r in zip(error, ref)],
                train_count=len(x), validation_count=len(tx), dimension=x.shape[1])
    return dict(label_keys=keys, train_episodes=sorted(train_ids), validation_episodes=sorted(val_ids),
                summary=summary, rows=rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/efs/prescottsun/SPT-JEPA/data/hardtasks-v1")
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = dict(scope="Fixed natural-image measures, CPU only, no learned segmentation/model/GPU",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  requested_count_per_task_split=args.count, toy_controls=toy_control(), tasks={})
    for task in ("tworoom", "cube"):
        result["tasks"][task] = run_task(Path(args.root)/f"{task}-metadata", args.count)
        print(task, {k: dict(wellposed=v["wellposed_count"], n=v["count"],
                             residual=v["heldout_relative"].get("quantiles", {}).get("0.5"))
                     for k, v in result["tasks"][task]["summary"].items()}, flush=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
