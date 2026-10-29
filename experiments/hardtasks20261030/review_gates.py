"""Collect complete short gates for an explicit research review, never auto-approve."""
import argparse,json,time
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--output",required=True);p.add_argument("--task",choices=["tworoom","cube"]);a=p.parse_args();r=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
 protocol=json.loads((r/"protocol.json").read_text());records=[];missing=[];initial=[]
 for entry in protocol["training_runs"]:
  if a.task and entry["task"]!=a.task:continue
  name=entry["id"];run=r/name;health=r/("health-"+name)/"health.json"
  exits=[r/"exits"/(n+".json") for n in [name,"health-"+name]]
  if not health.exists() or not all(x.exists() for x in exits):missing.append(name);continue
  assert all(json.loads(x.read_text())["returncode"]==0 for x in exits),name
  h=json.loads(health.read_text());assert h["audit"]["step"]==600 and all(v for k,v in h["audit"].items() if k not in ["step","gradient_max"])
  assert all(v>0 for v in h["audit"]["gradient_max"].values())
  s=h["summary"];uses={}
  for level in [.2,.5,.8,.95]:
   f=s[f"fm_t{level}_forecast"];z=s[f"fm_t{level}_zero"];w=s[f"fm_t{level}_wrong_actions"]
   uses[str(level)]=dict(forecast_gain_over_zero=(z-f)/z,correct_action_gain=(w-f)/w,forecast_mse=f,posterior_mse=s[f"fm_t{level}_posterior"])
  trace=[json.loads(x) for x in (run/"metrics.jsonl").read_text().splitlines()]
  records.append(dict(**entry,health=h,state_use=uses,training_trace=trace))
 for task in sorted(set(x["task"] for x in records)):
  for seed in [1,2]:
   ref=torch.load(r/f"{task}-original-seed{seed}-gate600/initial-model/weights.pt",map_location="cpu",weights_only=True)
   for row in [x for x in records if x["task"]==task and x["training_seed"]==seed]:
    weights=torch.load(r/row["id"]/"initial-model/weights.pt",map_location="cpu",weights_only=True);equal=set(weights)==set(ref) and all(torch.equal(v,weights[k]) for k,v in ref.items());assert equal,row["id"];initial.append(dict(id=row["id"],exact=True,tensors=len(ref)));del weights
   del ref
 report=dict(created_at=time.time(),complete=not missing,missing=missing,records=records,initial_matching=initial,approved_for_long_training=False,scope="Numerical and functional diagnostics only. Early feature/action weakness is not proof of persistent collapse. Explicit review is required before a bounded learning stage.")
 (out/"gate-review.json").write_text(json.dumps(report,indent=2)+"\n")
 for row in records:print(json.dumps(dict(id=row["id"],feature_std=row["health"]["feature_std"],state_use=row["state_use"],last_metrics=row["training_trace"][-1])),flush=True)
 print(json.dumps(dict(complete=not missing,completed=len(records),missing=missing)),flush=True)
if __name__=="__main__":main()
