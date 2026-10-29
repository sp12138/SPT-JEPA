"""Release retained slots only after every immutable legacy endpoint completes."""
import json,hashlib,time
from pathlib import Path

def main():
 b=Path("/efs/prescottsun/SPT-JEPA");out=b/"reports/harder-task-redirect-20261030";marker=out/"PUSHT-CLOSURE-REVIEWED.json";parents=[b/"runs"/x for x in ["forecast_confirm_20261030","promising_newseed_epoch2_20261030"]]
 if marker.exists():return
 while True:
  records=[];ready=True
  for r in parents:
   plan=json.loads((r/"plan.json").read_text());ids=[x["id"] for x in plan["jobs"]];assert len(ids)==len(set(ids));exits=[]
   for j in plan["jobs"]:
    p=r/"exits"/(j["id"]+".json")
    if not p.exists():ready=False;continue
    v=json.loads(p.read_text());assert v["returncode"]==0,(r,j["id"],v);exits.append(v)
    command=j.get("command",[])
    if "as_jepa_tools.train_stage" in command:
     stage=json.loads((Path(j["output"])/"stage-complete.json").read_text());assert stage["stage_complete"] and stage["stop_step"]==int(command[command.index("--stop-step")+1])
   for host in sorted(set(j["host"] for j in plan["jobs"])):
    state=json.loads((r/"workers"/(host+".json")).read_text())
    if state.get("running"):ready=False
   records.append(dict(root=str(r),expected=len(ids),completed=len(exits),plan_sha256=hashlib.sha256((r/"plan.json").read_bytes()).hexdigest()))
  if ready:
   report=dict(created_at=time.time(),parents=records,all_exit_zero=True,all_exact_training_caps=True,no_running_legacy_jobs=True,scope="Automated operational closure audit; no research-performance or novelty claim. Allows only already-reviewed harder-task queues to use released slots.")
   for r in parents:(r/"HOLD_DISPATCH").write_text("All planned endpoints completed; retain outputs and do not resume old grids.\n")
   tmp=marker.with_suffix(".tmp");tmp.write_text(json.dumps(report,indent=2)+"\n");tmp.replace(marker);print(json.dumps(report),flush=True);return
  print(json.dumps(dict(waiting=True,parents=records,at=time.time())),flush=True);time.sleep(300)
if __name__=="__main__":main()
