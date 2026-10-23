#!/usr/bin/env python3
"""FVD from paired raw RGB videos; see README.md for the exact protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parent
WEIGHTS_SHA256 = "bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2"
WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
PROTOCOL_VERSION = "i3d-kinetics400-styleganv-logits-v1"


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_array(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str((a.shape, a.dtype.str)).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def frechet_distance(real, fake, ddof=1):
    """Float64 Gaussian W2^2, with stable low-rank covariance factors.

    C_r = A.T @ A, C_f = B.T @ B.
    tr(sqrt(C_r**(1/2) @ C_f @ C_r**(1/2))) = ||A @ B.T||_*.
    This avoids complex sqrtm output on singular small-sample covariance.
    """
    x, y = np.asarray(real, dtype=np.float64), np.asarray(fake, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not x.shape[1]:
        raise ValueError("Features must have shapes [N,D] and [M,D], with the same D > 0.")
    if ddof not in (0, 1) or min(len(x), len(y)) < 2:
        raise ValueError("At least two videos per set are required; ddof must be 0 or 1.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite features.")

    def factor(z):
        centered = (z - z.mean(axis=0)) / math.sqrt(len(z) - ddof)
        trace = float(np.square(centered).sum())
        if len(z) <= z.shape[1]:
            return centered, trace
        # Bound the final SVD by D x D when evaluating large datasets.
        cov = centered.T @ centered
        values, vectors = np.linalg.eigh((cov + cov.T) * 0.5)
        return np.sqrt(np.maximum(values, 0))[:, None] * vectors.T, trace

    a, trace_x = factor(x)
    b, trace_y = factor(y)
    mean_term = float(np.square(x.mean(axis=0) - y.mean(axis=0)).sum())
    cross = float(np.linalg.svd(a @ b.T, compute_uv=False).sum())
    raw = mean_term + trace_x + trace_y - 2.0 * cross
    tolerance = 1e-9 * max(1.0, mean_term + trace_x + trace_y)
    if raw < -tolerance:
        raise FloatingPointError(f"Unexpected negative Frechet distance: {raw}")
    return {
        "fvd": float(max(0.0, raw)),
        "mean_term": mean_term,
        "covariance_term": float(max(0.0, trace_x + trace_y - 2.0 * cross)),
        "raw_fvd": float(raw),
        "covariance_ddof": ddof,
        "num_real": len(x),
        "num_fake": len(y),
        "feature_dim": x.shape[1],
    }


def frame_indices(frames, skip_first, stride):
    if frames < 16 or skip_first < 0 or stride < 1:
        raise ValueError("frames must be >=16, skip-first >=0, stride >=1.")
    return skip_first + np.arange(frames, dtype=np.int64) * stride


def load_clip(path, frames, skip_first, stride, expected_fps=None):
    """Strict uint8 RGB THWC; select one deterministic clip, no padding/resampling."""
    path = Path(path).resolve()
    indices = frame_indices(frames, skip_first, stride)
    actual_fps = None
    if path.suffix.lower() == ".npy":
        if expected_fps is None:
            raise ValueError(f"NPY has no FPS metadata; specify --fps (or manifest protocol.fps): {path}")
        source = np.load(path, mmap_mode="r", allow_pickle=False)
        if source.ndim != 4 or source.shape[-1] != 3 or source.dtype != np.uint8:
            raise ValueError(f"Expected uint8 [T,H,W,3] RGB, got {source.shape}/{source.dtype}: {path}")
        if source.shape[1] < 1 or source.shape[2] < 1:
            raise ValueError(f"Empty image: {path}")
        if len(source) <= indices[-1]:
            raise ValueError(f"Too short: {path}: {len(source)} frames, need index {indices[-1]}. No padding.")
        clip = np.ascontiguousarray(source[indices])
        source_shape = list(source.shape)
        actual_fps = float(expected_fps)
        fps_origin = "user-declared (NPY)"
    else:
        import av
        selected = []
        wanted = set(indices.tolist())
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError(f"No video stream: {path}")
            stream = container.streams.video[0]
            stream.thread_count = 2
            if stream.average_rate is None:
                raise ValueError(f"No FPS metadata: {path}")
            actual_fps = float(stream.average_rate)
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    selected.append(frame.to_ndarray(format="rgb24"))
                if index == indices[-1]:
                    break
        if len(selected) != frames:
            raise ValueError(f"Too short: {path}: decoded only {len(selected)}/{frames} selected frames. No padding.")
        clip = np.ascontiguousarray(np.stack(selected))
        source_shape = [None, *clip.shape[1:]]  # full duration need not be decoded
        fps_origin = "container average_rate; frame-index sampling"
    if not np.isfinite(actual_fps) or actual_fps <= 0:
        raise ValueError(f"Invalid FPS: {actual_fps}: {path}")
    if expected_fps is not None and not math.isclose(actual_fps, expected_fps, rel_tol=0, abs_tol=1e-3):
        raise ValueError(f"FPS mismatch {actual_fps} vs expected {expected_fps}: {path}")
    info = {
        "path": str(path), "source_shape": source_shape, "selected_shape": list(clip.shape),
        "fps": actual_fps, "fps_origin": fps_origin,
        "first_frame": int(indices[0]), "last_frame": int(indices[-1]), "stride": stride,
        "selected_rgb_sha256": digest_array(clip),
    }
    return clip, info


def load_pairs(manifest=None, real_dir=None, fake_dir=None, pattern="*.mp4"):
    metadata, suggested = {}, {}
    if manifest is not None:
        manifest = Path(manifest).resolve()
        payload = json.loads(manifest.read_text())
        entries = payload["pairs"]
        suggested = payload.get("protocol", {})
        metadata = payload.get("metadata", {})
        base = manifest.parent
        pairs = [{"id": str(e["id"]), "real": str((base / e["real"]).resolve()),
                  "fake": str((base / e["fake"]).resolve())} for e in entries]
    else:
        def index(root):
            root = Path(root).resolve()
            out = {}
            for path in sorted(root.glob(pattern)):
                if not path.is_file():
                    continue
                key = str(path.relative_to(root).with_suffix(""))
                if key in out:
                    raise ValueError(f"Duplicate relative stem {key} in {root}")
                out[key] = str(path)
            return out
        real, fake = index(real_dir), index(fake_dir)
        if real.keys() != fake.keys():
            raise ValueError(f"Unmatched IDs: real-only={sorted(real.keys()-fake.keys())[:10]}, "
                             f"fake-only={sorted(fake.keys()-real.keys())[:10]}")
        pairs = [{"id": key, "real": real[key], "fake": fake[key]} for key in sorted(real)]
    if len(pairs) < 2:
        raise ValueError("At least two matched videos are required.")
    for column in ("id", "real", "fake"):
        values = [p[column] for p in pairs]
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {column} entries would inflate the sample count.")
    for pair in pairs:
        for side in ("real", "fake"):
            if not Path(pair[side]).is_file():
                raise FileNotFoundError(pair[side])
    return pairs, suggested, metadata


def preprocess(clip, device):
    import torch
    import torch.nn.functional as F
    # Same operation order as the pinned detector's rescale=True, resize=True.
    x = torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous().unsqueeze(0).to(device)
    x = x.to(torch.float32) / 255.0 * 2.0 - 1.0
    b, c, t, h, w = x.shape
    x = F.interpolate(x.reshape(b, c*t, h, w), size=(224, 224),
                      mode="bilinear", align_corners=False, antialias=False)
    return x.reshape(b, c, t, 224, 224)


class I3DExtractor:
    def __init__(self, weights, device="cuda:0", cache_dir=ROOT / "cache", batch_size=1):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only cpu/cuda are supported by this evaluator.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; use --device cpu.")
        self.weights = Path(weights).resolve()
        self.weights_hash = digest_file(self.weights)
        if self.weights_hash != WEIGHTS_SHA256:
            raise ValueError("Unexpected I3D checkpoint SHA256; this protocol requires the pinned weights.")
        if batch_size < 1:
            raise ValueError("batch-size must be >=1.")
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Evaluation only; this process does not change training-process flags.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model = None
        self.identity = {
            "version": PROTOCOL_VERSION, "weights_sha256": self.weights_hash,
            "implementation_sha256": digest_file(__file__),
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(), "device_type": self.device.type,
            "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
            "precision": "float32; no autocast; TF32 disabled",
            "batch_size": batch_size,
            "resize": "bilinear 224x224, align_corners=False, antialias=False, no crop",
            "normalization": "uint8 RGB / 255 * 2 - 1",
            "feature": "400 logits, before softmax, temporal mean",
        }

    def _infer(self, clips):
        torch = self.torch
        if self.model is None:
            self.model = torch.jit.load(str(self.weights), map_location=self.device).eval()
        with torch.inference_mode():
            batch = torch.cat([preprocess(c, self.device) for c in clips], dim=0)
            result = self.model(batch, rescale=False, resize=False, return_features=True)
            result = result.float().cpu().numpy()
        if result.shape != (len(clips), 400) or not np.isfinite(result).all():
            raise ValueError(f"Invalid I3D features: shape={result.shape}")
        return result

    def extract(self, paths, protocol, label):
        features, records, pending = [], [], []
        hits = 0

        def flush():
            if not pending:
                return
            vectors = self._infer([p[1] for p in pending])
            for (index, _, key, cache_path), vector in zip(pending, vectors):
                features[index] = vector
                if cache_path:
                    tmp = cache_path.with_name(cache_path.name + "." + uuid.uuid4().hex + ".tmp")
                    with tmp.open("wb") as f:
                        np.savez_compressed(f, key=np.array(key), feature=vector)
                    os.replace(tmp, cache_path)
            pending.clear()

        for index, path in enumerate(paths):
            clip, info = load_clip(path, protocol["frames"], protocol["skip_first"],
                                   protocol["stride"], protocol["fps"])
            if protocol["fps"] is None:
                protocol["fps"] = info["fps"]
            key = hashlib.sha256(canonical_json({
                "detector": self.identity, "protocol": protocol,
                "selected_rgb_sha256": info["selected_rgb_sha256"],
            }).encode()).hexdigest()
            cache_path = self.cache_dir / (key + ".npz") if self.cache_dir else None
            vector = None
            if cache_path and cache_path.exists():
                with np.load(cache_path, allow_pickle=False) as data:
                    if data["key"].item() != key:
                        raise ValueError(f"Corrupt feature cache: {cache_path}")
                    vector = data["feature"].copy()
                if vector.shape != (400,) or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid cached feature: {cache_path}")
                hits += 1
            features.append(vector)
            info.update({"cache_key": key, "cache_hit": vector is not None})
            records.append(info)
            if vector is None:
                pending.append((index, clip, key, cache_path))
            if len(pending) >= self.batch_size:
                flush()
            print(f"[{label}] {index+1}/{len(paths)}; cache hits={hits}", file=sys.stderr, flush=True)
        flush()
        return np.stack(features), records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="JSON with pairs: [{id,real,fake}]")
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--fake-dir", type=Path)
    parser.add_argument("--pattern", default="*.mp4", help="Directory glob, e.g. '*.npy' or '**/*.mp4'")
    parser.add_argument("--frames", type=int, help="Selected RGB frames per video (default 128 or manifest)")
    parser.add_argument("--skip-first", type=int, help="Skip source frames (default 0 or manifest)")
    parser.add_argument("--stride", type=int, help="Temporal frame-index stride (default 1 or manifest)")
    parser.add_argument("--fps", type=float, help="Source FPS, required for NPY; validate, never resample")
    parser.add_argument("--cov-ddof", type=int, choices=(0, 1), default=1,
                        help="1: sample covariance (default); 0: StyleGAN-V dataset population covariance")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/i3d_torchscript.pt")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--save-features", type=Path, help="Optional NPZ of all extracted feature vectors")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if bool(args.manifest) == bool(args.real_dir or args.fake_dir):
        parser.error("Use --manifest OR both --real-dir and --fake-dir.")
    if not args.manifest and not (args.real_dir and args.fake_dir):
        parser.error("--real-dir and --fake-dir must be provided together.")
    if args.num_threads < 1:
        parser.error("--num-threads must be >=1.")
    pairs, suggested, metadata = load_pairs(args.manifest, args.real_dir, args.fake_dir, args.pattern)
    protocol = {}
    for key, fallback in (("frames", 128), ("skip_first", 0), ("stride", 1), ("fps", None)):
        supplied = getattr(args, key)
        protocol[key] = supplied if supplied is not None else suggested.get(key, fallback)
    frame_indices(protocol["frames"], protocol["skip_first"], protocol["stride"])
    if protocol["fps"] is not None and (not np.isfinite(protocol["fps"]) or protocol["fps"] <= 0):
        raise ValueError("fps must be finite and positive.")
    import torch
    torch.set_num_threads(args.num_threads)
    started = time.time()
    extractor = I3DExtractor(args.weights, args.device, None if args.no_cache else args.cache_dir, args.batch_size)
    real, real_info = extractor.extract([p["real"] for p in pairs], protocol, "real")
    fake, fake_info = extractor.extract([p["fake"] for p in pairs], protocol, "fake")
    result = frechet_distance(real, fake, args.cov_ddof)
    notes = [
        "FVD compares distributions of whole-video features; lower is better. It is not a per-case error.",
        "Exactly one selected clip per source video; no random crop, augmentation, padding, or FPS resampling.",
        "Compare only identical detector, covariance convention, N, frame selection, FPS, and source case bank.",
    ]
    if len(pairs) < 400:
        notes.append("Small-sample exploratory score: fewer than 400 videos; covariance is rank deficient "
                     "and the estimate is sample-size sensitive. Do not treat 16 cases as a final ranking.")
    result.update({
        "protocol": protocol, "detector": extractor.identity, "metadata": metadata,
        "manifest": str(args.manifest.resolve()) if args.manifest else None,
        "manifest_sha256": digest_file(args.manifest) if args.manifest else None,
        "real_features_sha256": digest_array(real), "fake_features_sha256": digest_array(fake),
        "elapsed_seconds": time.time() - started,
        "pairs": [{**p, "real_info": r, "fake_info": f} for p, r, f in zip(pairs, real_info, fake_info)],
        "notes": notes,
    })
    if args.save_features:
        args.save_features.parent.mkdir(parents=True, exist_ok=True)
        with args.save_features.open("wb") as f:
            np.savez_compressed(f, real=real, fake=fake, ids=np.array([p["id"] for p in pairs]),
                                protocol=np.array(canonical_json(protocol)),
                                detector=np.array(canonical_json(extractor.identity)))
        result["features_file"] = str(args.save_features.resolve())
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("fvd", "num_real", "num_fake", "feature_dim", "elapsed_seconds")}))
    print(f"Result: {args.output.resolve()}")


if __name__ == "__main__":
    main()
