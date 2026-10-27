"""GPU7 evaluates immutable exports and records evidence for early stopping."""
import json,os,time,traceback,fcntl
from pathlib import Path
import torch
from as_jepa_tools.evaluate import load_model,evaluate
from vforge.ops import OpsConfig,initialize_ops


def main():
    base=Path("/efs/prescottsun/SPT-JEPA");c=base/"runs/insights12h_20261027";pose=c/"pose/weights.pt"
    lock=(c/"evaluation-worker.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.set_float32_matmul_precision("high");initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda",0),parallel_state=None)
    deadline=json.loads((c/"plan.json").read_text())["deadline"];seen=set();last={}
    while time.time()<deadline:
        if not pose.exists():time.sleep(10);continue
        plan=json.loads((c/"plan.json").read_text());names=[job["id"] for job in plan["jobs"]]+["S2-native-b4-seed0"]
        candidates=[]
        for name in names:
            run=c/name
            if not run.exists():continue
            exports=sorted(run.glob("model-step*"))
            if not exports:continue
            folder=exports[-1];step=int(folder.name.split("step")[1])
            if not (folder/"config.json").exists() or not (folder/"weights.pt").exists():continue
            key=f"{name}:{step}"
            if key in seen:continue
            if step<2000:continue
            metrics=run/"eval"/f"step{step:07d}"/"metrics.json"
            if metrics.exists():seen.add(key);continue
            # Prioritize a matched first look, then prices near the useful range.
            priority=(0 if name not in last else 1,last.get(name,0),0 if "standard" in name else 1,name)
            candidates.append((priority,name,folder,step,key))
        if not candidates:time.sleep(15);continue
        _,name,folder,step,key=sorted(candidates)[0];run=c/name;out=run/"eval"/f"step{step:07d}"
        state=dict(state="evaluating",model=name,step=step,started_at=time.time(),pid=os.getpid());(c/"evaluation-worker.json").write_text(json.dumps(state,indent=2))
        try:
            model=load_model(folder);result=evaluate(model,base/"data/pusht",out,count=32,steps=20,pose_path=pose,batch_size=8)
            result["checkpoint_step"]=step;result["source_checkpoint"]=str(folder)
            (out/"metrics.json").write_text(json.dumps(result,indent=2))
            seen.add(key);last[name]=step
            # Rate diagnostics alone cannot prove collapse. Require useless posterior rendering,
            # near-constant states and no material change when replacing state with zeros.
            posterior=result.get("posterior_agent_by_horizon");anchor=result.get("anchor_agent_by_horizon")
            ratio=posterior/max(1e-6,anchor) if isinstance(posterior,(int,float)) and isinstance(anchor,(int,float)) else None
            collapsed=(step>=8000 and model.mode=="L2" and model.domain=="standard" and model.beta>0
                       and ratio is not None and ratio>=.9 and result.get("feature_std",1)<.01 and result.get("zero_state_intervention_mse",1)<.0002)
            if collapsed and not (run/"STOP").exists():
                reason=dict(reason="early stop: useless posterior reconstruction plus near-constant, intervention-insensitive state",step=step,posterior_to_anchor_agent_error=ratio,feature_std=result["feature_std"],zero_state_intervention_mse=result["zero_state_intervention_mse"],evidence=str(out/"metrics.json"),time=time.time())
                (run/"early_stop.json").write_text(json.dumps(reason,indent=2));(run/"STOP").write_text(json.dumps(reason))
            with (c/"evaluation-index.jsonl").open("a") as f:f.write(json.dumps(dict(model=name,step=step,posterior_to_anchor_ratio=ratio,early_stop=collapsed,h7_agent_ratio=result.get("h7_agent_ratio"),h7_block_ratio=result.get("h7_block_ratio"),output=str(out),finished_at=time.time()))+"\n")
            del model;torch.cuda.empty_cache()
        except Exception as error:
            out.mkdir(parents=True,exist_ok=True);(out/"evaluation_error.txt").write_text(traceback.format_exc());seen.add(key)
            with (c/"evaluation-errors.jsonl").open("a") as f:f.write(json.dumps(dict(model=name,step=step,error=repr(error),time=time.time()))+"\n")
            torch.cuda.empty_cache()
    (c/"evaluation-worker.json").write_text(json.dumps(dict(state="deadline_reached",time=time.time()),indent=2))

if __name__=="__main__":main()
