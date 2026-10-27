"""Export aggregate campaign measurements, keeping historical protocols explicit."""
import argparse,csv,datetime,json
from pathlib import Path


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None


def aggregate(value):
    if isinstance(value,dict):
        return {k:aggregate(v) for k,v in value.items() if k not in ["details","episode_successes","start_steps","predictions","truth"] and not (k=="episodes" and isinstance(v,list))}
    if isinstance(value,list):return [aggregate(v) for v in value]
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument("--campaign",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    c=Path(a.campaign);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);plan=read(c/"plan.json");measurements=[];runs=[]
    kinds={"probes.json":"probe","metrics.json":"metric","counterfactual.json":"counterfactual","swaps.json":"color_swap",
           "decomposition.json":"posterior_forecast","teacher_forcing.json":"native_context","feedback_clipping.json":"feedback_clip"}
    for job in plan["jobs"]:
        root=Path(job["output"]);source=read(root/"actual_source.json") or read(root/"source_manifest.json") or read(root/"execution.json") or {}
        latest=sorted(p for p in root.glob("model-step*") if (p/"weights.pt").exists() and (p/"config.json").exists())
        runs.append(dict(id=job["id"],host=job["host"],gpu=job["gpu"],hold=job.get("hold",False),parent=job.get("parent"),
            output=str(root),model_size=read(root/"model_size.json"),status=aggregate(read(root/"status.json")),exit=read(c/("exit-"+job["id"]+".json")),
            execution=read(root/"execution.json"),actual_source=source,training_config=read(root/"config.json"),
            latest_portable_checkpoint=str(latest[-1]) if latest else None,
            latest_portable_model_config=read(latest[-1]/"config.json") if latest else None))
        if not root.exists():continue
        for file in sorted(root.rglob("*.json")):
            if file.name not in kinds:continue
            parts=file.relative_to(root).parts
            if any(x.startswith("model-step") or x in ["checkpoint","checkpoints","source","runtime"] for x in parts):continue
            metric=read(file)
            if not isinstance(metric,dict):continue
            kind=kinds[file.name]
            if kind=="metric":
                if "episode_successes" in metric.get("metrics",{}):
                    kind="planning";successes=metric["metrics"]["episode_successes"];metric["trial_count"]=len(successes);metric["success_count"]=int(sum(successes))
                elif "h7_agent_ratio" in metric:kind="generation"
                else:continue
            final=root.name.startswith("final-");supp=root.name.startswith(("late-probes-","quality-late-","native-context-","native-feedback-","color-geometry-","decode-decomposition-"))
            tier="final" if final else "final diagnostic" if supp else "historical exploratory"
            flags=[]
            if tier=="historical exploratory":flags.append("Historical protocol; not pooled into final comparisons. See EVALUATION_PROTOCOL.md.")
            if "-dot-" in str(file) and kind=="generation" and "pose-v2" not in str(metric.get("pose_weights","")):
                flags.append("Dot pose scores excluded: v1/unrecorded pose evaluator fails real-dot-frame validation.")
            if final and (read(root/"status.json") or {}).get("state")!="finished":flags.append("Finalizer incomplete at collection time.")
            if any((read(Path(folder)/"status.json") or {}).get("state")!="finished" for folder in read(root/"required_components.json") or []):flags.append("Required parallel components incomplete at collection time.")
            if root.name.endswith("smoke8"):flags.append("Eight-case diagnostic contract only; excluded from substantive result claims.")
            measurements.append(dict(experiment=job["id"],measurement=str(file.relative_to(c)),path=str(file),kind=kind,tier=tier,
                source_commit=source.get("source_commit"),flags=flags,result=aggregate(metric)))
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/"campaign-configurations.json").write_text(json.dumps(dict(collected_at=stamp,deadline=plan["deadline"],runs=runs),indent=2,allow_nan=False))
    (out/"all-aggregate-measurements.json").write_text(json.dumps(dict(collected_at=stamp,measurements=measurements,
        scope="All discovered aggregate evaluation results and their protocols. Historical and excluded measurements are retained for provenance, not treated as final evidence. Per-case arrays, media, weights and training logs remain on EFS."),indent=2,allow_nan=False))
    columns=["experiment","measurement","kind","tier","source_commit","cases","sampler_steps","feature_precision","agent_r2","block_position_r2","nuisance_r2",
             "h1_agent_ratio","h3_agent_ratio","h7_agent_ratio","h13_agent_ratio","h7_block_ratio","h7_moving_block_ratio","success_rate",
             "action_identification_accuracy","anchor_style_transfer_projection","state_style_transfer_projection","flags"]
    with (out/"all-aggregate-measurements.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in measurements:
            metric=row["result"];line={k:row[k] for k in ["experiment","measurement","kind","tier","source_commit"]}
            line.update({k:metric.get(k) for k in columns if k not in line and k!="flags"})
            if row["kind"]=="planning":line["success_rate"]=metric["metrics"]["success_rate"];line["cases"]=metric["trial_count"]
            line["flags"]=" | ".join(row["flags"]);writer.writerow(line)
    print(json.dumps(dict(runs=len(runs),measurements=len(measurements),output=str(out)),indent=2))


if __name__=="__main__":main()
