"""Diagnostic intervention on a frozen Cube goal metric using train-only probes.

Privileged xyz labels fit diagnostic linear readouts only. They never update
the encoder, dynamics, planner, or existing candidate library. This is not a
proposed deployable method. All model selection uses the same saved candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def fit_readout(z, y, alpha):
    """Ridge in standardized coordinates, returned in raw latent coordinates."""
    z, y = np.asarray(z, float), np.asarray(y, float)
    center, scale, yc = z.mean(0), z.std(0), y.mean(0)
    scale = np.maximum(scale, 1e-6)
    x = (z-center)/scale
    coef = np.linalg.solve(x.T@x + alpha*np.eye(z.shape[1]), x.T@(y-yc))
    raw = coef/scale[:, None]
    intercept = yc-center@raw
    return raw, intercept


def probe_metrics(pred, target):
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    sse = np.square(pred-target).sum(0)
    sst = np.square(target-target.mean(0)).sum(0)
    per_axis = [float(1-e/v) if v > 1e-15 else None for e, v in zip(sse, sst)]
    return dict(r2_variance_weighted=float(1-sse.sum()/sst.sum()) if sst.sum()>1e-15 else None,
                r2_xyz=per_axis, rmse_m=float(np.sqrt(np.square(pred-target).mean())), samples=len(target))


def select_rows(h5, episodes, count, frames, seed, episode_lookup):
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(episodes, size=count, replace=False)).astype(int)
    rows = []
    for episode in picked:
        offset, length = episode_lookup[int(episode)]
        assert length >= frames
        rel = np.linspace(0, length-1, frames).round().astype(int)
        assert len(set(rel.tolist())) == frames
        rows.extend((offset+rel).tolist())
    rows = np.asarray(rows, dtype=np.int64)
    assert np.all(rows[1:] > rows[:-1]), "Rows must be strictly ordered for HDF5 reads"
    assert np.array_equal(h5["ep_idx"][rows], np.repeat(picked, frames)), "Selected row episode IDs differ from split IDs"
    return picked, rows


def encode_rows(model, h5, rows, batch):
    import torch
    from as_jepa_tools.cube_oracle_audit import transform_pixels
    embeddings = []
    with torch.inference_mode():
        for lo in range(0, len(rows), batch):
            pixels = h5["pixels"][rows[lo:lo+batch]]
            emb = model.encode({"pixels": transform_pixels(pixels)[:, None]})["emb"][:, 0]
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def spearman(x, y):
    from scipy.stats import spearmanr
    if np.ptp(x) < 1e-15 or np.ptp(y) < 1e-15:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else None


def ci(values, boot):
    x = np.array([np.nan if v is None else v for v in values], float)
    valid = np.isfinite(x)
    if not valid.any():
        return dict(mean=None, lo=None, hi=None, n=0)
    y = x[boot]
    n = np.isfinite(y).sum(1)
    estimates = np.nansum(y, axis=1)[n>0]/n[n>0]
    lo, hi = np.quantile(estimates, [.025, .975])
    return dict(mean=float(x[valid].mean()), lo=float(lo), hi=float(hi), n=int(valid.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--ridge-alpha", type=float, default=1.)
    p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--bootstrap", type=int, default=5000)
    a = p.parse_args()
    assert a.ridge_alpha > 0
    import hdf5plugin  # noqa: F401
    import h5py
    import torch
    import stable_worldmodel as swm
    from vforge.ops import OpsConfig, initialize_ops
    from as_jepa_tools.cube_oracle_audit import transform_pixels
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/"results.json").exists(), "Refusing to overwrite a completed diagnostic"
    score_path = Path(a.scores)
    shard_files = sorted(score_path.glob("scores-shard*.json"))
    assert shard_files
    metadata = json.loads(shard_files[0].read_text())
    assert Path(metadata["checkpoint"]).resolve() == Path(a.checkpoint).resolve()
    assert metadata["checkpoint_weights_sha256"] == sha(Path(a.checkpoint)/"weights.pt")
    pool = Path(metadata["args"]["pool"])
    manifest_path = Path(metadata["args"]["manifest"])
    assert sha(manifest_path) == metadata["manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    cases = sorted({int(r["case"]) for file in shard_files for r in json.loads(file.read_text())["records"]})
    assert cases == list(range(50)), "This registered diagnostic requires all original 50 cases"
    card = json.loads((Path(a.data)/"dataset_card.json").read_text())
    train_eps = np.load(Path(a.data)/"train_episodes.npy")
    val_eps = np.load(Path(a.data)/"validation_episodes.npy")
    episode_ids = np.load(Path(a.data)/"episode_ids.npy")
    offsets = np.load(Path(a.data)/"offsets.npy")
    lengths = np.load(Path(a.data)/"lengths.npy")
    assert len(episode_ids) == len(offsets) == len(lengths)
    assert len(set(episode_ids.tolist())) == len(episode_ids)
    episode_lookup = {int(e): (int(o), int(n)) for e, o, n in zip(episode_ids, offsets, lengths)}
    assert not set(train_eps.tolist()) & set(val_eps.tolist())
    initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False), device=torch.device("cuda"), parallel_state=None)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    model = swm.wm.utils.load_pretrained(a.checkpoint).cuda().eval()
    model.requires_grad_(False)
    with h5py.File(card["h5_path"], "r", swmr=True) as h5:
        assert np.array_equal(offsets, h5["ep_offset"][:]) and np.array_equal(lengths, h5["ep_len"][:])
        ep_train, rows_train = select_rows(h5, train_eps, a.episodes, a.frames, a.seed, episode_lookup)
        ep_val, rows_val = select_rows(h5, val_eps, a.episodes, a.frames, a.seed+1, episode_lookup)
        assert not set(rows_train.tolist()) & set(rows_val.tolist())
        ztrain = encode_rows(model, h5, rows_train, a.batch)
        zval = encode_rows(model, h5, rows_val, a.batch)
        targets = {"effector": (h5["proprio_effector_pos"][rows_train], h5["proprio_effector_pos"][rows_val]),
                   "block": (h5["privileged_block_0_pos"][rows_train], h5["privileged_block_0_pos"][rows_val])}
        eff_w, eff_b = fit_readout(ztrain, targets["effector"][0], a.ridge_alpha)
        block_w, block_b = fit_readout(ztrain, targets["block"][0], a.ridge_alpha)
        assert np.linalg.matrix_rank(eff_w) == 3, "Effector readout has fewer than three independent directions"
        qeff, _ = np.linalg.qr(eff_w, mode="reduced")
        qrandom, _ = np.linalg.qr(np.random.default_rng(a.seed).normal(size=(ztrain.shape[1], 3)), mode="reduced")
        projections = {"original": np.eye(ztrain.shape[1]),
                       "remove_effector": np.eye(ztrain.shape[1])-qeff@qeff.T,
                       "remove_random": np.eye(ztrain.shape[1])-qrandom@qrandom.T}
        projection_audit = {name: dict(rank=int(np.linalg.matrix_rank(matrix)),
                                      idempotence_max=float(np.max(np.abs(matrix@matrix-matrix))),
                                      symmetry_max=float(np.max(np.abs(matrix-matrix.T))))
                            for name, matrix in projections.items()}
        probes, refits = {}, {}
        for name, matrix in projections.items():
            probes[name], refits[name] = {}, {}
            for target, (ytrain, yval) in targets.items():
                w, intercept = fit_readout(ztrain@matrix, ytrain, a.ridge_alpha)
                refits[name][target] = (w, intercept)
                probes[name][target] = dict(train=probe_metrics(ztrain@matrix@w+intercept, ytrain),
                                            validation=probe_metrics(zval@matrix@w+intercept, yval))
        np.savez_compressed(out/"probe-fit.npz", train_episodes=ep_train, validation_episodes=ep_val,
                            train_rows=rows_train, validation_rows=rows_val, train_embeddings=ztrain,
                            validation_embeddings=zval, effector_weight=eff_w, effector_intercept=eff_b,
                            block_weight=block_w, block_intercept=block_b, effector_basis=qeff, random_basis=qrandom)
        write(out/"probe-results.json", dict(probes=probes, projection_audit=projection_audit, args=vars(a)))
        records, agreement = [], []
        counterfactual = {group: {source: {name: {target: [[], []] for target in targets}
                                                   for name in projections} for source in ["true", "predicted"]}
                          for group in ["nonoracle", "all"]}
        for case in cases:
            with np.load(pool/f"case{case:04d}.npz") as physical, np.load(score_path/f"scores-case{case:04d}.npz") as scored:
                assert physical["actions"].shape == (39, 25, 5)
                goal_pixel = physical["goal_pixels"]
                with torch.inference_mode():
                    goal_z = model.encode({"pixels": transform_pixels(goal_pixel[None])[None]})["emb"][0, 0].cpu().numpy()
                true_z = scored["true_embeddings"][:, -1]
                predicted_z = scored["predicted_embeddings"][:, -1]
                labels = physical["family"]
                pos, angle = physical["position_error_m"][:, -1], physical["orientation_error_deg"][:, -1]
                pose = (pos <= .03) & (angle <= 15)
                success = physical["success"].any(1)
                target_pos = h5["privileged_block_0_pos"][int(manifest["pairs"][case]["goal_row"])]
                groups = {"all": np.arange(39), "nonoracle": np.flatnonzero(np.isin(labels, ["zero", "gaussian"]) | np.char.startswith(labels, "cem_"))}
                costs = {source: {} for source in ["true", "predicted"]}
                for source, embeddings, saved_key in [("true", true_z, "encoded_cost"), ("predicted", predicted_z, "predicted_cost")]:
                    # Re-encode the actual goal; never substitute a replayed endpoint.
                    delta = embeddings.astype(float)-goal_z.astype(float)
                    for method, matrix in projections.items():
                        costs[source][method] = np.square(delta@matrix).sum(-1)
                    costs[source]["block_readout"] = np.square(delta@block_w).sum(-1)
                    # Auxiliary distinction: remove goal-readout error using actual
                    # privileged goal xyz; not the primary four-way comparison.
                    costs[source]["block_readout_physical_goal"] = np.square(embeddings@block_w+block_b-target_pos).sum(-1)
                    expected = scored[saved_key][:, -1].astype(float)
                    max_abs = float(np.max(np.abs(costs[source]["original"]-expected)))
                    scale = max(float(np.max(np.abs(expected))), 1e-6)
                    assert max_abs <= 1e-4*scale + 1e-7, f"Goal re-encoding/cost mismatch: {case} {source} {max_abs}"
                    agreement.append(dict(case=case, source=source, max_abs_cost_error=max_abs, relative_to_case_max=max_abs/scale,
                                          selected_index_unchanged=bool(np.argmin(costs[source]["original"])==np.argmin(expected))))
                    for group, ids in groups.items():
                        true_targets = {"effector": physical["effector_positions"][ids, -1], "block": physical["positions"][ids, -1]}
                        for projection, matrix in projections.items():
                            for target in targets:
                                w, intercept = refits[projection][target]
                                prediction = embeddings[ids]@matrix@w+intercept
                                acc = counterfactual[group][source][projection][target]
                                acc[0].append(prediction); acc[1].append(true_targets[target])
                        for method, c in costs[source].items():
                            chosen = int(ids[np.argmin(c[ids])])
                            records.append(dict(case=case, source=source, group=group, method=method,
                                chosen=chosen, family=str(labels[chosen]), position_error_m=float(pos[chosen]),
                                orientation_error_deg=float(angle[chosen]), physical_cost=float(physical["physical_cost"][chosen, -1]),
                                position_success=float(pos[chosen]<=.03), endpoint_pose_success=float(pose[chosen]),
                                strict_success=float(success[chosen]),
                                physical_rank=spearman(c[ids], physical["physical_cost"][ids, -1]),
                                position_rank=spearman(c[ids], pos[ids]), orientation_rank=spearman(c[ids], angle[ids])))
                np.savez_compressed(out/f"intervention-case{case:04d}.npz", goal_embedding=goal_z,
                                    **{source+"__"+method: value for source, metrics in costs.items() for method, value in metrics.items()})
            print(json.dumps(dict(case=case, status="scored")), flush=True)
    cf_metrics = {group: {source: {projection: {target: probe_metrics(np.concatenate(value[0]), np.concatenate(value[1]))
                                              for target, value in target_values.items()}
                                  for projection, target_values in projections_values.items()}
                         for source, projections_values in sources.items()}
                  for group, sources in counterfactual.items()}
    boot = np.random.default_rng(a.seed).integers(0, len(cases), (a.bootstrap, len(cases)))
    keys = ["position_error_m", "orientation_error_deg", "physical_cost", "position_success", "endpoint_pose_success", "strict_success", "physical_rank", "position_rank", "orientation_rank"]
    summary, paired = {}, {}
    methods = list(costs["true"])
    for group in ["nonoracle", "all"]:
        summary[group], paired[group] = {}, {}
        for source in ["true", "predicted"]:
            summary[group][source], paired[group][source] = {}, {}
            select = lambda method: [r for r in records if r["group"]==group and r["source"]==source and r["method"]==method]
            baseline = select("original")
            for method in methods:
                rows = select(method)
                assert [r["case"] for r in rows] == cases
                summary[group][source][method] = {key: ci([r[key] for r in rows], boot) for key in keys}
                paired[group][source][method] = {key: ci([None if r[key] is None or b[key] is None else r[key]-b[key]
                                                          for r, b in zip(rows, baseline)], boot) for key in keys}
    result = dict(summary=summary, paired_differences_from_original=paired, probes=probes,
                  counterfactual_probes=cf_metrics, projection_audit=projection_audit, original_cost_agreement=agreement,
                  records=records, args=vars(a), manifest_sha256=sha(manifest_path), source_sha256=sha(__file__),
                  checkpoint_weights_sha256=metadata["checkpoint_weights_sha256"], old_score_script_sha256=metadata["script_sha256"],
                  train_episodes=ep_train.tolist(), validation_episodes=ep_val.tolist(),
                  episode_metadata_sha256={name: sha(Path(a.data)/(name+".npy")) for name in ["episode_ids", "offsets", "lengths", "train_episodes", "validation_episodes"]},
                  scope="Frozen-metric causal intervention, conditional on fitted train-only privileged-state probes and fixed candidate pool. It is not a trained policy or a proposed benchmark method. Removing one linear effector readout subspace need not remove all effector information and may remove block information; refit probes quantify this. Case bootstrap is conditional on weights, readout fit and candidate pool.")
    write(out/"results.json", result)
    lines = ["# Cube frozen-metric intervention", "", result["scope"], "", f"Probe fit: {a.episodes} train episodes x {a.frames} frames; separate validation set of the same size; fixed standardized ridge alpha={a.ridge_alpha}.", ""]
    for group in ["nonoracle", "all"]:
        for source in ["true", "predicted"]:
            lines += [f"## {group}: {source} endpoints", "", "| Metric | Position success % | Endpoint pose success % | Sustained success % | Mean position error m | Physical rank |", "|---|---:|---:|---:|---:|---:|"]
            for method in methods:
                s = summary[group][source][method]
                values = [s[k]["mean"] for k in ["position_success", "endpoint_pose_success", "strict_success", "position_error_m", "physical_rank"]]
                lines.append("| "+method+" | "+" | ".join("undefined" if v is None else f"{v*(100 if i<3 else 1):.3f}" for i,v in enumerate(values))+" |")
            lines += [""]
    lines += ["Full bootstrap intervals, paired differences, original-cost identity checks, and train/validation/counterfactual probes are in results.json. The block-readout metric uses the encoded goal; its physical-goal variant separately diagnoses readout calibration error. Neither uses optimal-action labels.", ""]
    (out/"RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(dict(status="complete", output=str(out))), flush=True)


if __name__ == "__main__":
    main()
