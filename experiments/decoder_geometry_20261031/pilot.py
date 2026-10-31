"""One training clip / one JVP / eight VJPs. Never launches calibration or planning."""
import argparse
import importlib
import json
from pathlib import Path
import time
import torch

from experiments.carrier_progress_20261031.metric_core import digest, verify_manifest
from experiments.decoder_geometry_20261031.core import (
    PROBES, SEED, directional_check, response, spectrum, state_hash, training_point, vjp_gram)


def main():
    p = argparse.ArgumentParser()
    for name in ["checkpoint", "manifest", "source-record", "output"]:
        p.add_argument("--"+name, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("A pilot always requires a new output directory")
    source = json.loads(Path(args.source_record).read_text())
    source_manifest = json.loads(Path(source["manifest"]).read_text())
    for name in ["core.py", "pilot.py"]:
        relative = "experiments/decoder_geometry_20261031/"+name
        assert digest(Path(__file__).with_name(name)) == source_manifest["files"][relative]
        assert Path(__file__).resolve() == (Path(source["source"])/"experiments/decoder_geometry_20261031/pilot.py").resolve()
    checkpoint = Path(args.checkpoint)
    cfg = json.loads((checkpoint.parent/"config.json").read_text())
    exported = json.loads((checkpoint/"config.json").read_text())
    assert cfg["dataset_kind"] == "benchmark_hdf5"
    selected = cfg.get("model_class", "robust")
    assert selected in ["carrier", "robust"] and cfg.get("robust_variant", False)
    from as_jepa_lewm.factory import build_model, model_target
    assert exported == {"_target_":model_target(cfg), **cfg["model"]}
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    from vforge.ops import OpsConfig, initialize_ops
    device = torch.device(args.device)
    torch.set_num_threads(2); torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    initialize_ops(OpsConfig(backend="sdpa_math", allow_fallback=False), device=device, parallel_state=None)
    model = build_model(cfg).to(device).eval().requires_grad_(False)
    assert model.state_dim == 64 and model.renderer is not None
    weights_hash = digest(checkpoint/"weights.pt")
    model.load_state_dict(torch.load(checkpoint/"weights.pt", map_location="cpu", weights_only=True), strict=True)
    before = state_hash(model)
    ds = BenchmarkWindows(cfg["data"], window=model.window, split="train")
    manifest = json.loads(Path(args.manifest).read_text()); verify_manifest(ds, manifest)
    entry = manifest["windows"][0]
    item = ds[entry["index"]]
    assert item["episode"] == entry["episode"] and item["rows"].tolist() == entry["rows"]
    batch = {k:v.unsqueeze(0).to(device) if torch.is_tensor(v) else torch.tensor([v], device=device)
             for k,v in item.items()}
    if ds.file is not None: ds.file.close()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    started = time.perf_counter()
    z, x, t, anchor, metadata = training_point(model, batch, index=0)
    function = response(model, x, t, anchor)
    direction_report, direction = directional_check(function, z)
    if device.type == "cuda": torch.cuda.synchronize(device)
    directional_seconds = time.perf_counter()-started
    started_vjp = time.perf_counter()
    gram, vectors = vjp_gram(function, z, PROBES)
    if device.type == "cuda": torch.cuda.synchronize(device)
    vjp_seconds = time.perf_counter()-started_vjp
    eigen = spectrum(gram)
    assert eigen["psd"] and eigen["numerical_rank"] <= PROBES
    assert state_hash(model) == before and digest(checkpoint/"weights.pt") == weights_hash
    assert not any(m.training for m in model.modules())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    finite_ok = all(x["relative_error"] < .03 for x in direction_report["finite_differences"])
    response_nonzero = direction_report["jvp_norm"] > 0 and eigen["trace"] > 0
    # A noisy 8-probe estimate is reported, never selected using its agreement.
    sketch_energy = float(direction @ gram @ direction)
    paths = {}
    for module in ["as_jepa_lewm.factory", "as_jepa_lewm.model", "as_jepa_lewm.robust",
                   "as_jepa_lewm.carrier", "vforge.models.as_jepa.flow",
                   "vforge.models.as_jepa.renderer", "vforge.ops._attention.dispatch"]:
        path = Path(importlib.import_module(module).__file__)
        paths[module] = dict(path=str(path), sha256=digest(path))
    report = dict(passed=finite_ok and response_nonzero, checkpoint=str(checkpoint),
        checkpoint_sha256=weights_hash, config_sha256=digest(checkpoint/"config.json"),
        training_config_sha256=digest(checkpoint.parent/"config.json"),
        source=source, source_record_sha256=digest(args.source_record), imported_sources=paths,
        manifest=str(Path(args.manifest)), manifest_sha256=digest(args.manifest), train_entry=entry,
        point=metadata, seed=SEED, probes=PROBES, directional_check=direction_report,
        sketch_directional_energy=sketch_energy, spectrum=eigen,
        state_before_after_equal=True, parameters_frozen=True, parameter_grads_absent=True,
        attention="sdpa_math", dtype="float32", torch_version=torch.__version__, device=str(device),
        directional_seconds=directional_seconds, eight_vjp_seconds=vjp_seconds,
        rough128_clip_vjp_seconds=128*vjp_seconds,
        peak_device_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        scope="Technical one-point pilot only. The rank<=8 matrix is not a global calibrated metric and must not be used in planning. No validation or physical labels read; no parameter/buffer updates.")
    out.mkdir(parents=True)
    torch.save(dict(gram=gram.cpu(), vectors=vectors.cpu()), out/"pilot-response.pt")
    (out/"pilot.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]: raise SystemExit("Technical pilot did not pass; do not dispatch calibration")


if __name__ == "__main__": main()
