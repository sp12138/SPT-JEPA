"""Prepare fixed carrier normalization from training images only.

Uses the current native model initialization and benchmark loader; does not
train, read physical labels, construct validation data, or update a checkpoint.
Root orchestrates any requested CUDA invocation; this helper defaults to CPU.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from as_jepa_lewm.carrier import CarrierLeWM, PROJECTION_SEED
from as_jepa_tools.benchmark_data import BenchmarkWindows
from as_jepa_tools.benchmark_spatial_probe import episode_indices
from vforge.ops import OpsConfig, initialize_ops


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def calibration_statistics(values):
    if values.ndim != 2 or values.shape[-1] != 64 or len(values) < 2:
        raise ValueError("Require at least two 64-dimensional training carrier measurements")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Nonfinite training carrier measurement")
    values = values.double()
    mean = values.mean(0)
    raw_scale = values.std(0, unbiased=False)
    # A declared fixed numerical floor; no validation or per-task tuning.
    scale = raw_scale.clamp_min(1e-6)
    return mean.float(), scale.float(), raw_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Original task/seed config, not a checkpoint")
    parser.add_argument("--output", required=True, help="New small calibration JSON; existing files are refused")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--selection-seed", type=int, default=1042)
    parser.add_argument("--source-commit", default="unspecified")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Do not overwrite a frozen calibration: " + str(output))
    if args.episodes < 2 or args.batch_size < 1:
        raise ValueError("Need >=2 training episodes and a positive batch size")
    cfg = json.loads(Path(args.config).read_text())
    if cfg.get("dataset_kind") != "benchmark_hdf5":
        raise ValueError("This gate calibrates only the established benchmark datasets")
    if cfg.get("initial_encoder_checkpoint") or cfg.get("calibrate_state_std") is not None:
        raise ValueError("Carrier requires fresh initialization, not prior learned encoder/calibration")
    model_kwargs = dict(cfg["model"])
    for key in ["carrier_mean", "carrier_scale"]:
        model_kwargs.pop(key, None)
    model_kwargs["carrier_chart"] = "identity"
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(int(cfg["seed"]))
    backend = "sdpa" if device.type == "cuda" else "sdpa_math"
    initialize_ops(OpsConfig(backend=backend, allow_fallback=False), device=device, parallel_state=None)
    model = CarrierLeWM(**model_kwargs).to(device).eval()
    encoder_sha = tensor_hash(model.encoder.state_dict())
    projection_sha = tensor_hash({"carrier_projection": model.carrier_projection})
    dataset = BenchmarkWindows(cfg["data"], window=model.window, split="train", seed=cfg["seed"])
    indices = episode_indices(dataset, args.episodes, args.selection_seed)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=0)
    values, episodes, starts, rows = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            # Only each sampled window's initial image is measured. The native
            # dataset defines exact resizing and episode/window selection.
            pixels = model.normalize(batch["pixels"][:, :1].to(device))
            values.append(model.raw_carrier(pixels)[:, 0].cpu())
            episodes.extend(int(v) for v in batch["episode"])
            starts.extend(int(v) for v in batch["start"])
            rows.extend(int(v) for v in batch["rows"][:, 0])
    values = torch.cat(values)
    mean, scale, raw_scale = calibration_statistics(values)
    assert tensor_hash(model.encoder.state_dict()) == encoder_sha
    assert len(episodes) == len(set(episodes))
    root = Path(cfg["data"])
    train_episode_hash = file_hash(root / "train_episodes.npy")
    assert set(episodes).issubset(set(int(v) for v in np.load(root / "train_episodes.npy")))
    result = dict(
        scope="Frozen random encoder; training images only; no physical targets or optimizer updates.",
        source_commit=args.source_commit, config=str(Path(args.config)), config_sha256=file_hash(args.config),
        carrier_source_sha256=file_hash(__import__("as_jepa_lewm.carrier", fromlist=["__file__"]).__file__),
        helper_source_sha256=file_hash(__file__), dataset=str(root),
        dataset_card_sha256=file_hash(root / "dataset_card.json"), train_episode_ids_sha256=train_episode_hash,
        encoder_initial_sha256=encoder_sha, projection_sha256=projection_sha,
        projection_seed=PROJECTION_SEED, model_seed=int(cfg["seed"]), selection_seed=args.selection_seed,
        device=str(device), attention_backend=backend, input_precision="float32", selected_window_indices=indices,
        training_episode_ids=episodes, starts=starts, measured_h5_rows=rows,
        measured_frames=len(values), calibration_exposure="one initial frame per selected training window/episode",
        scale_floor=1e-6, scale_floor_coordinates=int((raw_scale < 1e-6).sum()),
        raw_std_min=float(raw_scale.min()), raw_std_mean=float(raw_scale.mean()), raw_std_max=float(raw_scale.max()),
        model_kwargs=dict(carrier_mean=mean.tolist(), carrier_scale=scale.tolist()),
        parameter_counts=model.carrier_parameter_counts(),
        chart_initial_norm_certificate=model.carrier_chart_module.norm_certificate())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(output), frames=len(values), encoder_sha256=encoder_sha,
                         scale_floor_coordinates=result["scale_floor_coordinates"])), flush=True)


if __name__ == "__main__":
    main()
