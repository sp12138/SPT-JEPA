"""Confirm candidate outcomes on declared windows excluded from earlier diagnostics."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    for key in ["checkpoint","data","pose","i3d","output","exclude-quality","exclude-counterfactual","upstream","planning-data"]:
        p.add_argument("--"+key,required=True)
    a=p.parse_args()
    checkpoint,out=Path(a.checkpoint),Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((checkpoint/"config.json").read_text())
    digest=hashlib.sha256((checkpoint/"weights.pt").read_bytes()).hexdigest()
    manifest=dict(checkpoint=str(checkpoint),weights_sha256=digest,config=cfg,
                  quality_selection_seed=20261028,counterfactual_selection_seed=142,
                  planning_seeds=[242,243,244],exclude_quality=a.exclude_quality,
                  exclude_counterfactual=a.exclude_counterfactual,
                  scope="Confirmation windows exclude overlapping diagnostic clips; same original test episodes,not a new dataset or training seed.")
    identity=out/"protocol.json"
    if identity.exists() and json.loads(identity.read_text())!=manifest:
        raise RuntimeError("Confirmation identity changed")
    identity.write_text(json.dumps(manifest,indent=2))
    py=os.sys.executable
    jobs=[
        dict(id="quality",artifact=str(out/"quality/metrics.json"),
             command=[py,"-B","-u","-m","as_jepa_tools.quality_complete","--checkpoint",str(checkpoint),
                      "--data",a.data,"--pose",a.pose,"--i3d",a.i3d,"--output",str(out/"quality"),
                      "--cases","256","--media","--selection-seed","20261028","--exclude-cases",a.exclude_quality]),
        dict(id="quality-postprocess",artifact=str(out/"quality-additional/metrics.json"),
             command=[py,"-B","-u","-m","as_jepa_tools.quality_postprocess","--quality",str(out/"quality"),
                      "--output",str(out/"quality-additional"),"--i3d",a.i3d]),
    ]
    mode=cfg.get("mode","S2" if "ASJepaSPT" in cfg.get("_target_","") else "S1")
    if cfg.get("domain","standard")=="standard":
        jobs.append(dict(id="counterfactual",artifact=str(out/"counterfactual/counterfactual.json"),
                         command=[py,"-B","-u","-m","as_jepa_tools.counterfactual","--checkpoint",str(checkpoint),
                                  "--data",a.data,"--pose",a.pose,"--output",str(out/"counterfactual"),
                                  "--cases","64","--steps","50","--selection-seed","142",
                                  "--exclude-cases",a.exclude_counterfactual]))
        if mode.startswith("L"):
            for seed in [242,243,244]:
                dest=out/"planning"/("seed"+str(seed))
                jobs.append(dict(id="planning-"+str(seed),artifact=str(dest/"metrics.json"),
                                 command=[py,"-B","-u","-m","as_jepa_tools.planning","--checkpoint",str(checkpoint),
                                          "--upstream",a.upstream,"--data-root",a.planning_data,"--output",str(dest),
                                          "--test-episodes",str(Path(a.data)/"test_episodes.npy"),
                                          "--seed",str(seed),"--episodes","50"]))
    else:
        if cfg["domain"]=="color":
            jobs.append(dict(id="color-swaps",artifact=str(out/"color-swaps/swaps.json"),
                             command=[py,"-B","-u","-m","as_jepa_tools.color_swaps","--checkpoint",str(checkpoint),
                                      "--data",a.data,"--pose",a.pose,"--output",str(out/"color-swaps"),
                                      "--cases","128","--steps","50"]))
    (out/"jobs.json").write_text(json.dumps(jobs,indent=2))
    results=[]
    for job in jobs:
        (out/"status.json").write_text(json.dumps(dict(state="running",stage=job["id"],completed=len(results),total=len(jobs),updated_at=time.time()),indent=2))
        if Path(job["artifact"]).exists():
            results.append(dict(id=job["id"],returncode=0,adopted_complete_artifact=True))
            continue
        start=time.time()
        with (out/(job["id"]+".log")).open("a") as log:
            rc=subprocess.run(job["command"],stdout=log,stderr=subprocess.STDOUT).returncode
        if rc or not Path(job["artifact"]).exists():
            (out/"status.json").write_text(json.dumps(dict(state="failed",stage=job["id"],returncode=rc,results=results,updated_at=time.time()),indent=2))
            raise RuntimeError("Incomplete confirmation stage: "+job["id"])
        results.append(dict(id=job["id"],returncode=rc,seconds=time.time()-start))
    if hashlib.sha256((checkpoint/"weights.pt").read_bytes()).hexdigest()!=digest:
        raise RuntimeError("Checkpoint changed during confirmation")
    (out/"status.json").write_text(json.dumps(dict(state="finished",results=results,updated_at=time.time()),indent=2))
    print(json.dumps(dict(checkpoint=str(checkpoint),stages=len(jobs),scope=manifest["scope"])))


if __name__=="__main__":
    main()
