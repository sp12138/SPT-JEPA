"""Fixed 128 training clips x 8 VJPs; frozen weights, no planner or optimizer."""
import argparse
import importlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

from experiments.carrier_progress_20261031.metric_core import digest, verify_manifest
from experiments.decoder_geometry_20261031.core import (
    PROBES, SEED, response, spectrum, state_hash, training_point, vjp_gram)

B = Path("/efs/prescottsun/SPT-JEPA")
ORIGINAL = B/"runs/carrier_learn6000_20261031/source-record.json"


def finalize(vectors):
    """QR gives nonnegative quadratic costs without an eigenvalue floor."""
    vectors = torch.as_tensor(vectors, dtype=torch.float64, device="cpu")
    assert vectors.ndim == 3 and torch.isfinite(vectors).all()
    flat = vectors.flatten(0, 1)
    assert flat.shape[0] >= flat.shape[1]
    raw = flat.T @ flat / len(flat)
    trace = raw.trace()
    assert torch.isfinite(trace) and trace > 0
    _, triangular = torch.linalg.qr(flat/math.sqrt(len(flat)), mode="reduced")
    scale = flat.shape[1]/trace
    factor = triangular.T * scale.sqrt()
    # Haar QR sign correction; one predeclared rotation for every model/task.
    q, r = torch.linalg.qr(torch.randn((flat.shape[1], flat.shape[1]),
        dtype=torch.float64, generator=torch.Generator().manual_seed(SEED)))
    q = q * torch.where(torch.diag(r) < 0, -1., 1.)
    rotated_factor = q.T @ factor
    normalized = raw * scale
    rotated = q.T @ normalized @ q
    torch.testing.assert_close(factor @ factor.T, normalized, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(rotated_factor @ rotated_factor.T, rotated, atol=1e-12, rtol=1e-10)
    arrays = dict(decoder_gram_raw=raw, decoder_gram=normalized, rotated_gram=rotated,
                  decoder_factor=factor, rotated_factor=rotated_factor, rotation=q)
    return {k:v.numpy() for k,v in arrays.items()}, dict(raw=spectrum(raw),
        normalized=spectrum(normalized), rotated=spectrum(rotated), scalar_multiplier=float(scale),
        formula="G=mean_{128 clips,8 probes}(u u^T); normalized=64*G/trace(G)",
        factorization="thin QR of stacked VJP vectors; no floor, eigenvalue clipping, or identity mixture")


def verify_sources(source):
    original = json.loads(ORIGINAL.read_text())
    assert original["source_commit"].startswith("8e0f821")
    manifests = [json.loads(Path(x["manifest"]).read_text()) for x in [source, original]]
    runtime_manifest = json.loads(Path(source["runtime_manifest"]).read_text())
    files = {}
    for relative in ["as_jepa_lewm/model.py", "as_jepa_lewm/robust.py", "as_jepa_lewm/carrier.py",
                     "as_jepa_tools/benchmark_data.py"]:
        paths = [Path(x["source"])/relative for x in [source, original]]
        sha = digest(paths[0]); assert sha == digest(paths[1])
        assert all(sha == m["files"][relative] for m in manifests)
        files[str(paths[0])] = sha
    runtime_files = ["models/as_jepa/"+x+".py" for x in ["flow", "renderer", "pooling", "pixels", "codes", "constants"]]
    runtime_files += ["models/rep_world/"+x+".py" for x in ["world", "layers", "config"]]
    runtime_files += ["ops/_attention/dispatch.py"]
    for name in runtime_files:
        relative = "runtime/vforge/"+name
        paths = [Path(x["runtime_source"])/relative for x in [source, original]]
        sha = digest(paths[0]); assert sha == digest(paths[1]) == runtime_manifest["files"][relative]
        files[str(paths[0])] = sha
    for name in ["core.py", "calibrate.py", "prepare.py"]:
        relative = "experiments/decoder_geometry_20261031/"+name
        path = Path(__file__).with_name(name)
        assert path.resolve() == (Path(source["source"])/relative).resolve()
        assert digest(path) == manifests[0]["files"][relative]
    actual = {}
    for name in ["as_jepa_lewm.model", "as_jepa_lewm.robust", "as_jepa_lewm.carrier", "as_jepa_tools.benchmark_data",
                 "vforge.models.as_jepa.flow", "vforge.models.as_jepa.renderer", "vforge.ops._attention.dispatch"]:
        path = Path(importlib.import_module(name).__file__)
        assert digest(path) == files[str(path)]
        actual[name] = dict(path=str(path), sha256=digest(path))
    return dict(original_source=original, original_record_sha256=digest(ORIGINAL),
                byte_equal_files=files, actual_imports=actual)


def main():
    p = argparse.ArgumentParser()
    for name in ["checkpoint", "manifest", "source-record", "output", "expected-weights-sha256"]:
        p.add_argument("--"+name, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(); out = Path(args.output)
    if any((out/name).exists() for name in ["calibration.json", "response-vectors.npz", "matrices.npz", "points.json"]):
        raise FileExistsError("Preserve completed/partial calibration artifacts")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = json.loads(Path(args.source_record).read_text()); sources = verify_sources(source)
    from as_jepa_lewm.factory import build_model, model_target
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    from vforge.ops import OpsConfig, initialize_ops
    checkpoint = Path(args.checkpoint); cfg = json.loads((checkpoint.parent/"config.json").read_text())
    assert checkpoint.name == "model-step0006000" and cfg["dataset_kind"] == "benchmark_hdf5"
    assert cfg.get("model_class", "robust") in ["carrier", "robust"] and cfg.get("robust_variant")
    assert cfg["model"].get("carrier_chart", "identity") == "identity"
    assert json.loads((checkpoint/"config.json").read_text()) == {"_target_":model_target(cfg), **cfg["model"]}
    weights = checkpoint/"weights.pt"; assert digest(weights) == args.expected_weights_sha256
    device = torch.device(args.device)
    torch.set_num_threads(2); torch.manual_seed(SEED); torch.set_float32_matmul_precision("highest")
    initialize_ops(OpsConfig(backend="sdpa_math", allow_fallback=False), device=device, parallel_state=None)
    model = build_model(cfg).to(device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    assert model.state_dim == 64 and model.window == 8 and model.renderer is not None
    before = state_hash(model)
    dataset = BenchmarkWindows(cfg["data"], window=8, split="train")
    manifest = json.loads(Path(args.manifest).read_text()); verify_manifest(dataset, manifest)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    vectors, points = [], []; vjp_seconds = 0.; loop_start = time.perf_counter()
    for index, entry in enumerate(manifest["windows"]):
        item = dataset[entry["index"]]
        assert item["episode"] == entry["episode"] and item["rows"].tolist() == entry["rows"]
        batch = {k:v.unsqueeze(0).to(device) if torch.is_tensor(v) else torch.tensor([v], device=device)
                 for k,v in item.items()}
        z, x, t, anchor, meta = training_point(model, batch, index)
        if device.type == "cuda": torch.cuda.synchronize(device)
        tick = time.perf_counter()
        _, values = vjp_gram(response(model, x, t, anchor), z, PROBES, index)
        if device.type == "cuda": torch.cuda.synchronize(device)
        vjp_seconds += time.perf_counter()-tick
        assert values.shape == (8, 64) and torch.isfinite(values).all()
        vectors.append(values.cpu()); points.append(dict(**meta, episode=entry["episode"], rows=entry["rows"],
            response_trace=float(values.square().sum()/PROBES)))
        if (index+1) % 32 == 0: print(json.dumps(dict(clips=index+1, seconds=time.perf_counter()-loop_start)), flush=True)
    dataset.file.close()
    vectors = torch.stack(vectors); assert vectors.shape == (128, 8, 64)
    matrices, report = finalize(vectors)
    assert all(report[key]["psd"] for key in ["raw", "normalized", "rotated"])
    after = state_hash(model); assert before == after and digest(weights) == args.expected_weights_sha256
    assert not any(m.training for m in model.modules())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    np.savez_compressed(out/"matrices.npz", **matrices)
    np.savez_compressed(out/"response-vectors.npz", vectors=vectors.numpy())
    (out/"points.json").write_text(json.dumps(points, indent=2)+"\n")
    result = dict(passed=True, checkpoint=str(checkpoint), weights_sha256=args.expected_weights_sha256,
        config_sha256=digest(checkpoint/"config.json"), training_config_sha256=digest(checkpoint.parent/"config.json"),
        manifest=str(Path(args.manifest)), manifest_sha256=digest(args.manifest), source=source, source_audit=sources,
        source_record_sha256=digest(args.source_record), seed=SEED, clips=128, probes_per_clip=8, dimensions=64,
        spectrum=report, state_before_sha256=before, state_after_sha256=after, parameters_frozen=True,
        parameter_grads_absent=True, model_eval=True, point_count=len(points),
        forecast_points=sum(x["forecast"] for x in points), zero_response_points=sum(x["response_trace"] == 0 for x in points),
        artifacts={name:digest(out/name) for name in ["matrices.npz", "response-vectors.npz", "points.json"]},
        vjp_seconds=vjp_seconds, loop_seconds=time.perf_counter()-loop_start, total_seconds=time.perf_counter()-started,
        peak_device_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        torch_version=torch.__version__, device=str(device), attention="sdpa_math", dtype="float32; Gram/QR float64 CPU",
        metric_names=["identity", "decoder", "rotated"],
        scope="Training-only global velocity-response geometry. No validation outcomes, physical labels, optimizer, changed weights or planning. Fixed cohort/time/noise/condition/probes; not a convergence or task-benefit claim.")
    (out/"calibration.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(passed=True, output=str(out), spectrum=report, seconds=result["total_seconds"])), flush=True)


if __name__ == "__main__": main()
