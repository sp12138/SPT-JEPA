"""Finish matched endpoints with pinned weights and checked nested subprocess results."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def verify_probes(path, expected=None):
    status = json.loads(Path(path).read_text())
    rows = status.get("results", [])
    if not rows or (expected is not None and len(rows) != expected):
        raise RuntimeError("Probe batch did not finish all requested inner calls")
    failures = [x for x in rows if x.get("error")]
    if status.get("state") != "finished" or failures:
        raise RuntimeError("Probe batch contains incomplete or failed inner calls: " + str(failures))
    for result in status["results"]:
        if not (Path(result["output"]) / "probes.json").exists():
            raise RuntimeError("Missing numerical probe result: " + result["output"])


def main():
    p = argparse.ArgumentParser()
    for key in ("parent", "output", "data", "pose", "i3d", "upstream", "planning-data", "mode"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--no-quality", action="store_true")
    p.add_argument("--pixel-feedback", action="store_true")
    p.add_argument("--expected-step", type=int)
    a = p.parse_args()
    parent, out = Path(a.parent), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    exports = sorted(x for x in parent.glob("model-step*") if (x / "weights.pt").exists())
    if not exports:
        raise RuntimeError("No portable checkpoint; export the completed Native DCP first")
    source = exports[-1]
    step = int(source.name.split("step")[1])
    if a.expected_step is not None and step != a.expected_step:
        raise RuntimeError(f"Expected step {a.expected_step}, found {step}")
    pin = out / "checkpoint"
    pin.mkdir(exist_ok=True)
    cfg = json.loads((source / "config.json").read_text())
    if a.pixel_feedback:
        if cfg.get("_target_") != "vforge.models.as_jepa.semantic_carry.SemanticCarry":
            raise ValueError("Pixel feedback override is only for the matched semantic carry control")
        cfg["use_semantic_feedback"] = False
    if not (pin / "weights.pt").exists():
        os.link(source / "weights.pt", pin / "weights.pt")
    config_text = json.dumps(cfg, indent=2)
    if (pin / "config.json").exists() and json.loads((pin / "config.json").read_text()) != cfg:
        raise RuntimeError("Pinned evaluation configuration changed")
    (pin / "config.json").write_text(config_text)
    weights_hash = hashlib.sha256((pin / "weights.pt").read_bytes()).hexdigest()
    manifest = dict(parent=str(parent), checkpoint=str(source), step=step,
                    weights_sha256=weights_hash, model_config=cfg, data=a.data, pose=a.pose,
                    pixel_feedback_control=a.pixel_feedback)
    previous = out / "checkpoint_manifest.json"
    if previous.exists() and json.loads(previous.read_text()) != manifest:
        raise RuntimeError("Pinned evaluation identity changed")
    atomic(previous, manifest)
    py = os.sys.executable
    probes = []
    for variant in (["mean", "noisy", "mean-fp32", "noisy-fp32"] if a.mode in ["L2", "S2"] else ["mean"]):
        probes.append(dict(model=parent.name, checkpoint=str(pin), data=a.data,
                           output=str(out / "probes" / variant), noisy=variant.startswith("noisy"),
                           fp32=variant.endswith("fp32"), train=50000, test=5000))
    (out / "probe_jobs.json").write_text(json.dumps(probes, indent=2))
    jobs = [dict(id="probes", artifact=str(out / "probes/status.json"),
                 command=[py, "-B", "-u", "-m", "as_jepa_tools.probe_batch",
                          "--jobs", str(out / "probe_jobs.json"), "--output", str(out / "probes")])]
    if a.mode in ["L2", "S2"]:
        jobs.append(dict(id="collapse", artifact=str(out / "collapse/metrics.json"),
                         command=[py, "-B", "-u", "-m", "as_jepa_tools.collapse_audit",
                                  "--checkpoint", str(pin), "--data", a.data, "--pose", a.pose,
                                  "--output", str(out / "collapse")]))
    if a.mode != "L0" and not a.no_quality:
        jobs.extend([
            dict(id="quality", artifact=str(out / "quality/metrics.json"),
                 command=[py, "-B", "-u", "-m", "as_jepa_tools.quality_complete",
                          "--checkpoint", str(pin), "--data", a.data, "--pose", a.pose,
                          "--output", str(out / "quality"), "--i3d", a.i3d, "--cases", "256", "--media"]),
            dict(id="quality-postprocess", artifact=str(out / "quality-additional/metrics.json"),
                 command=[py, "-B", "-u", "-m", "as_jepa_tools.quality_postprocess",
                          "--quality", str(out / "quality"), "--output", str(out / "quality-additional"),
                          "--i3d", a.i3d]),
        ])
        if cfg.get("domain", "standard") == "standard":
            jobs.append(dict(id="counterfactual", artifact=str(out / "counterfactual/counterfactual.json"),
                             command=[py, "-B", "-u", "-m", "as_jepa_tools.counterfactual",
                                      "--checkpoint", str(pin), "--data", a.data, "--pose", a.pose,
                                      "--output", str(out / "counterfactual"), "--cases", "64", "--steps", "50", "--media"]))
    if a.mode.startswith("L") and cfg.get("domain", "standard") == "standard":
        for seed in [42, 43, 44]:
            folder = out / "planning" / ("seed" + str(seed))
            jobs.append(dict(id="planning-" + str(seed), artifact=str(folder / "metrics.json"),
                             command=[py, "-B", "-u", "-m", "as_jepa_tools.planning",
                                      "--checkpoint", str(pin), "--output", str(folder),
                                      "--upstream", a.upstream, "--data-root", a.planning_data,
                                      "--test-episodes", str(Path(a.data) / "test_episodes.npy"),
                                      "--seed", str(seed), "--episodes", "50"]))
    if a.mode!="L0" and not a.no_quality:
        jobs.append(dict(id="anchor-swaps",artifact=str(out/"anchor-swaps/metrics.json"),
                         command=[py,"-B","-u","-m","as_jepa_tools.swap_gallery",
                                  "--checkpoint",str(pin),"--data",a.data,"--pose",a.pose,
                                  "--cases",str(out/"quality/cases.json"),"--output",str(out/"anchor-swaps")]))
    if a.mode=="L0" or not a.no_quality:
        gallery_command=[py,"-B","-u","-m","as_jepa_tools.gallery_complete",
                         "--checkpoint",str(pin),"--output",str(out/"gallery-complete"),
                         "--planning",str(out/"planning")]
        gallery_command+=(["--planning-only"] if a.mode=="L0" else ["--quality",str(out/"quality"),"--data",a.data])
        jobs.append(dict(id="gallery-complete",artifact=str(out/"gallery-complete/complete.json"),command=gallery_command))
    atomic(out / "jobs.json", jobs)
    results = []
    for job in jobs:
        atomic(out / "status.json", dict(state="running", stage=job["id"], completed=len(results),
                                         total=len(jobs), updated_at=time.time()))
        done = out / (job["id"] + ".done.json")
        try:
            if done.exists():
                if not Path(job["artifact"]).exists():
                    raise RuntimeError("Completed stage is missing its artifact: " + job["id"])
                if job["id"] == "probes":
                    verify_probes(job["artifact"], expected=len(probes))
                results.append(json.loads(done.read_text()))
                continue
            # Adopt a complete numerical artifact after a supervisor interruption.
            if Path(job["artifact"]).exists():
                if job["id"] == "probes":
                    verify_probes(job["artifact"], expected=len(probes))
                record = dict(id=job["id"], returncode=0, adopted_complete_artifact=True)
            else:
                started = time.time()
                with (out / (job["id"] + ".log")).open("a") as log:
                    rc = subprocess.run(job["command"], stdout=log, stderr=subprocess.STDOUT).returncode
                if rc:
                    raise RuntimeError(f"{job['id']} exited with {rc}")
                if not Path(job["artifact"]).exists():
                    raise RuntimeError("Stage exited without numerical artifact: " + job["id"])
                if job["id"] == "probes":
                    verify_probes(job["artifact"], expected=len(probes))
                record = dict(id=job["id"], returncode=0, seconds=time.time() - started)
            atomic(done, record)
            results.append(record)
        except Exception as error:
            atomic(out / "status.json", dict(state="failed", stage=job["id"], completed=len(results),
                                             total=len(jobs), results=results, error=str(error),
                                             updated_at=time.time()))
            raise
    atomic(out / "status.json", dict(state="finished", completed=len(results), total=len(jobs),
                                     results=results, updated_at=time.time()))
    print(json.dumps(dict(model=parent.name, step=step, stages=len(jobs))))


if __name__ == "__main__":
    main()
