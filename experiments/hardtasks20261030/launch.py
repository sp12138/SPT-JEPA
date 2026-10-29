"""Launch one durable dispatcher on its specified authorized host."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
b=Path("/efs/prescottsun/SPT-JEPA");r=b/"runs/hardtasks_gate600_20261030";host=sys.argv[1]
assert host in ["dev2","dev13","dev15"] and host+"-" in os.uname().nodename
plan=json.loads((r/"plan.json").read_text());source=Path(plan["jobs"][0]["source"]["source"])
workers=r/"workers";workers.mkdir(exist_ok=True);record=workers/(host+"-launch.json");assert not record.exists()
log=(workers/(host+".log")).open("ab")
p=subprocess.Popen(["/efs/prescottsun/project_jepa/envs/lewm/bin/python","-B","-u",str(source/"as_jepa_tools/workflow.py"),"--plan",str(r/"plan.json"),"--host",host],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=source)
d=dict(host=host,pid=p.pid,started_at=time.time(),plan_sha256=hashlib.sha256((r/"plan.json").read_bytes()).hexdigest());record.write_text(json.dumps(d,indent=2));print(json.dumps(d))
