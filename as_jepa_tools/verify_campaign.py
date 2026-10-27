"""Validate final campaign evidence before publishing aggregate results."""
import argparse,datetime,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None


def digest(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--campaign",required=True);p.add_argument("--output",required=True);p.add_argument("--require-complete",action="store_true");a=p.parse_args()
    c=Path(a.campaign);code=c.parent.parent/"code";pending=[];errors=[];records=[];source_checks={};reference=None;planner_cases={};test_ids=set(np.load(c.parent.parent/"data/pusht/test_episodes.npy").tolist())
    def check(condition,message):
        if not condition:errors.append(message)
    def finished(root,label):
        status=read(root/"status.json") or {}
        if status.get("state")!="finished":pending.append(dict(job=label,state=status.get("state","not_started")));return False
        check(not any(r.get("returncode",0) for r in status.get("results",[])),label+": nonzero stage return code")
        return True
    jobs=read(c/"finalization_registry.json")["jobs"]
    for job in jobs:
        root=Path(job["output"]);label=job["parent"]
        if not finished(root,label):continue
        if not all(finished(Path(folder),label+":component:"+Path(folder).name) for folder in read(root/"required_components.json") or []):continue
        manifest=read(root/"checkpoint_manifest.json");origin=Path(manifest["checkpoint"]);pin=root/"checkpoint";cfg=manifest["model_config"]
        if cfg["mode"]!="L0":check((root/"generation/mean/metrics.json").exists(),label+": missing mean generation")
        if cfg["mode"]=="L2":check((root/"generation/samples/metrics.json").exists(),label+": missing sampled generation")
        if cfg.get("domain","standard")=="standard":
            check(len(list((root/"planning").glob("seed*/metrics.json")))==3,label+": expected three planning seeds")
            if cfg["mode"]!="L0":check((root/"counterfactual/counterfactual.json").exists(),label+": missing action counterfactual")
        check(os.path.samefile(origin/"weights.pt",pin/"weights.pt"),label+": weight pin differs from parent export")
        check((origin/"config.json").read_bytes()==(pin/"config.json").read_bytes(),label+": model config pin differs")
        probes=read(root/"probes/status.json") or {};expected=read(root/"probe_jobs.json")
        check(probes.get("state")=="finished" and probes.get("completed")==len(expected),label+": incomplete probes")
        check(not any(r.get("error") for r in probes.get("results",[])),label+": inner probe failure")
        for pp in expected:
            score=read(Path(pp["output"])/"probes.json")
            check(score is not None,label+": missing probe output")
            if score:check(score["train_frames"]==20000 and score["test_frames"]==5000,label+": probe split count")
        for folder in [root/"generation/mean",root/"generation/samples"]:
            score=read(folder/"metrics.json")
            if score is None:continue
            cases=read(folder/"cases.json");arr=np.load(folder/"per_case.npz")
            check(len(cases)==score["cases"],label+": generation case count")
            check("pose-v2" in score["pose_weights"],label+": pose evaluator is not v2")
            check(score.get("sampler_steps",score.get("steps"))==50,label+": Euler step mismatch")
            identities=[(r["episode"],r["start"]) for r in cases]
            if folder.name=="mean":
                if reference is None:reference=identities
                else:check(identities==reference,label+": mean rollout case identities differ")
                check(len(cases)==256,label+": mean rollout case budget differs")
            for entity in ["agent","block"]:
                values=arr[entity];copies=arr["copy_"+entity]
                check(values.shape[0]==len(cases) and values.shape[-1]==13,label+": array shape")
                check(bool(np.isfinite(values).all()),label+": nonfinite pose errors")
                ratio=float(values[...,6].mean()/copies[...,6].mean())
                check(abs(ratio-score["h7_"+entity+"_ratio"])<1e-5,label+": stored ratio disagrees with per-case data")
        for folder in sorted((root/"planning").glob("seed*")):
            score=read(folder/"metrics.json")
            if not score:errors.append(label+": missing planner metric");continue
            key=score["seed"];ids=list(zip(score["episodes"],score["start_steps"]));success=score["metrics"]["episode_successes"]
            check(len(success)==50 and len(ids)==50,label+": planning trial count")
            check(score.get("split")=="heldout" and all(ep in test_ids for ep,_ in ids),label+": planning episodes are not held out")
            if key in planner_cases:check(ids==planner_cases[key],label+": planning cases differ across models")
            else:planner_cases[key]=ids
            check(abs(np.mean(success)*100-score["metrics"]["success_rate"])<1e-8,label+": planning rate")
        cf=read(root/"counterfactual/counterfactual.json")
        if cf:
            check(cf["cases"]==64 and len(cf["details"])==64,label+": counterfactual count")
            check(abs(np.mean([np.mean(x["correct"]) for x in cf["details"]])-cf["action_identification_accuracy"])<1e-8,label+": counterfactual aggregation")
        parent=Path(manifest["parent"]);src=read(parent/"actual_source.json") or read(parent/"source_manifest.json") or read(parent/"execution.json") or {}
        if src.get("source_commit") and src.get("source"):
            for name in ["as_jepa_lewm/model.py","as_jepa_lewm/train.py"]:
                path=Path(src["source"])/name;key=src["source_commit"]+":"+name
                if key not in source_checks:
                    blob=subprocess.check_output(["git","-C",str(code),"show",key]);expected_hash=hashlib.sha256(blob).hexdigest();actual_hash=digest(path)
                    source_checks[key]=dict(path=str(path),sha256=actual_hash,passed=actual_hash==expected_hash)
                    check(actual_hash==expected_hash,"Immutable training source differs: "+str(path))
        records.append(dict(model=label,step=manifest["checkpoint_step"],weights_sha256=digest(pin/"weights.pt"),training_source=src))
    for registry in ["late_native_quality_registry.json","native_context_registry.json","native_feedback_registry.json","color_geometry_registry.json","long_budget_diagnostics_registry.json"]:
        for job in read(c/registry) or []:
            root=Path(job["output"])
            if not finished(root,job["id"]):continue
            if registry in ["native_context_registry.json","native_feedback_registry.json"]:
                score=read(root/"results"/("teacher_forcing.json" if registry=="native_context_registry.json" else "feedback_clipping.json")) or {}
                check(score.get("cases")==256 and score.get("control",{}).get("passed",False),job["id"]+": matched-context intervention contract")
    mid=c/"budget24k-main-dev2";manifest=read(mid/"checkpoint_manifest.json")
    if manifest:
        check(digest(Path(manifest["checkpoint"])/"weights.pt")==manifest["weights_sha256"],"Earlier24k checkpoint pin changed")
        probe_status=read(mid/"probes/status.json") or {};check(probe_status.get("completed")==3 and not any(x.get("error") for x in probe_status.get("results",[])),"Earlier24k probes failed/incomplete")
        cases=read(mid/"generation/cases.json") or [];check([(x["episode"],x["start"]) for x in cases]==reference,"Earlier24k generation case IDs")
        for seed in [42,43,44]:
            score=read(c/f"budget24k-planning-seed{seed}-dev2/results/metrics.json")
            if score:check(list(zip(score["episodes"],score["start_steps"]))==planner_cases[seed],"Earlier24k planner case IDs")
        long_cases=read(c/"decode-decomposition-long50k-dev2/results/L2-standard-calibrated-b001-50k-seed0-n128/cases.json")
        old_cases=read(c/"decode-decomposition-batch-dev15/results/L2-standard-calibrated-b001-seed0-n128/cases.json")
        check(bool(long_cases) and long_cases==old_cases,"Posterior decomposition case IDs differ across25k/50k")
    for name in ["late-probes-standard-dev2","late-probes-native-dev2"]:
        root=c/name;status=read(root/"results/status.json") or {}
        if status.get("state")!="finished":pending.append(dict(job=name,state=status.get("state")));continue
        check(not any(x.get("error") for x in status["results"]),name+": probe failure")
        check(status["completed"]==len(read(root/"jobs.json")),name+": missing probe entries")
        for item in read(root/"checkpoint_manifests.json"):
            check(digest(Path(item["checkpoint"])/"weights.pt")==item["weights_sha256"],name+": pinned weights changed")
    result=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete=not pending and not errors,
                final_models_checked=len(records),expected_final_models=len(jobs),pending=pending,errors=errors,models=records,immutable_source_checks=source_checks,
                pose_v2_sha256=digest(c/"pose-v2/weights.pt"),test_episode_ids_sha256=digest(c.parent.parent/"data/pusht/test_episodes.npy"),
                scope="Checks final counts, nested probe failures, pinned weights, source bytes, common cases, evaluator identity and numeric aggregation. Does not establish scientific performance.")
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:result[k] for k in ["complete","final_models_checked","expected_final_models","pending","errors"]},indent=2))
    if errors or (a.require_complete and pending):raise SystemExit(1)


if __name__=="__main__":main()
