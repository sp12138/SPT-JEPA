"""One authorized read-only precision measurement; uses frozen campaign source.

No updates or configuration mutation. This review script is not training code.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from as_jepa_lewm.factory import build_model
from as_jepa_tools.benchmark_data import BenchmarkWindows
from experiments.ideas_validate_20261031.audit import verify_manifest
from vforge.ops import OpsConfig, initialize_ops


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(reference, other, sigma=None):
    reference, other = reference.double(), other.double()
    error = other-reference
    coordinate_std = reference.flatten(0, 1).std(0, unbiased=False)
    rms_std = float(coordinate_std.square().mean().sqrt())
    rms = float(error.square().mean().sqrt())
    maximum = float(error.abs().max())
    result = dict(rms_error=rms, max_abs_error=maximum,
                  observed_rms_coordinate_std=rms_std,
                  observed_min_coordinate_std=float(coordinate_std.min()),
                  rms_over_observed_std=rms/rms_std,
                  max_over_observed_std=maximum/rms_std,
                  max_coordinatewise_error_over_own_std=float((error.abs()/coordinate_std.clamp_min(1e-30)).max()))
    if sigma is not None:
        result.update(rms_over_sigma=rms/sigma, max_over_sigma=maximum/sigma)
    return result


def main():
    base = Path("/efs/prescottsun/SPT-JEPA/runs")
    root = base/"ideas_validate600_20261031"
    source = root/"source-v1"
    pre = base/"ideas_validate_preflight_20261031"
    out = Path("/efs/prescottsun/SPT-JEPA/reports/carrier-precision-review-20261031")
    out.mkdir(parents=True, exist_ok=True)
    destination = out/"precision-review.json"
    if destination.exists():
        raise FileExistsError(destination)
    started = time.time()
    pid = os.getpid()
    (out/"pid.json").write_text(json.dumps(dict(pid=pid,host="dev2",physical_gpu=4,started=started))+"\n")
    print(json.dumps(dict(event="start",pid=pid,host="dev2",physical_gpu=4)),flush=True)
    torch.set_num_threads(2)
    device = torch.device("cuda", 0)
    initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=device,parallel_state=None)
    rows = []
    with torch.no_grad():
        for task in ["tworoom", "cube"]:
            run = pre/f"{task}-carrier-bounded-seed1"
            config = run/"config.json"
            cfg = json.loads(config.read_text())
            manifest_path = source/f"experiments/ideas_validate_20261031/audit-windows-{task}.json"
            manifest = json.loads(manifest_path.read_text())
            dataset = BenchmarkWindows(cfg["data"],window=cfg["model"]["window"],split="validation")
            verify_manifest(dataset,manifest)
            selected = manifest["windows"][:16]
            loader = DataLoader(Subset(dataset,[row["index"] for row in selected]),batch_size=4,num_workers=0)
            # Cache these same observed bytes for all precision/checkpoint pairs.
            pixels = [batch["pixels"] for batch in loader]
            for step,folder in [(0,"initial-model"),(2,"model-step0000002")]:
                checkpoint = run/folder/"weights.pt"
                model = build_model(cfg).to(device).eval()
                model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=True)
                before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                measurements = {}
                for precision,enabled in [("highest",False),("high",True)]:
                    torch.set_float32_matmul_precision(precision)
                    torch.backends.cuda.matmul.allow_tf32=enabled
                    raw, encoded, chart_common = [], [], []
                    for block in pixels:
                        normalized = model.normalize(block.to(device))
                        value = model.raw_carrier(normalized)
                        # Exact encode composition, reusing raw rather than
                        # evaluating the identical frozen image network twice.
                        state = model.carrier_chart_module((value-model.carrier_mean)/model.carrier_scale)
                        raw.append(value.cpu()); encoded.append(state.cpu())
                    measurements[precision] = dict(raw=torch.cat(raw),encoded=torch.cat(encoded))
                    if precision == "highest":
                        common = ((measurements[precision]["raw"].to(device)-model.carrier_mean)/model.carrier_scale)
                    for block in common.split(4):
                        chart_common.append(model.carrier_chart_module(block).cpu())
                    measurements[precision]["chart_common_input"] = torch.cat(chart_common)
                unchanged = all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items())
                assert unchanged
                row = dict(task=task,step=step,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                           config_sha256=sha(config),manifest_sha256=sha(manifest_path),windows=selected,
                           raw_carrier=compare(measurements["highest"]["raw"],measurements["high"]["raw"]),
                           encoded_state=compare(measurements["highest"]["encoded"],measurements["high"]["encoded"],model.sigma0),
                           chart_on_identical_input=compare(measurements["highest"]["chart_common_input"],measurements["high"]["chart_common_input"],model.sigma0),
                           state_dict_unchanged=unchanged,chart_certificate=model.carrier_chart_module.norm_certificate())
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k in ["task","step","raw_carrier","encoded_state","chart_on_identical_input"]}),flush=True)
                del model,before,measurements,common
                torch.cuda.empty_cache()
    source_record=json.loads((root/"source-record.json").read_text())
    report=dict(scope="One read-only dev2 GPU4 numerical review, no optimizer/source/config changes",pid=pid,
                physical_gpu=4,cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                device_name=torch.cuda.get_device_name(0),torch_version=torch.__version__,
                source_commit=source_record["source_commit"],source_manifest_sha256=sha(root/"source-v1.json"),
                script_path=str(Path(__file__)),script_sha256=sha(__file__),
                compared_settings=[dict(float32_matmul_precision="highest",allow_tf32=False),
                                   dict(float32_matmul_precision="high",allow_tf32=True)],
                cases_per_task=16,frames_per_case=8,rows=rows,seconds=time.time()-started)
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    print(json.dumps(dict(event="complete",pid=pid,output=str(destination),seconds=report["seconds"])),flush=True)


if __name__ == "__main__":
    main()
