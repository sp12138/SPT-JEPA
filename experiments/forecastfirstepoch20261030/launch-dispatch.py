from pathlib import Path
import datetime,hashlib,json,os,socket,subprocess,sys,time
b=Path("/efs/prescottsun/SPT-JEPA");s=b/"runs/forecast_firstepoch_20261030";alias=sys.argv[1]
assert alias in ["dev2","dev13","dev15"] and ("prescottsun-"+alias+"-") in socket.gethostname()
source=json.loads((s/"dispatcher-source.json").read_text());manifest=json.loads(Path(source["manifest"]).read_text());worker=Path(source["source"])/"as_jepa_tools/workflow.py"
assert hashlib.sha256(worker.read_bytes()).hexdigest()==manifest["files"]["as_jepa_tools/workflow.py"]
state=s/"workers"/(alias+".json")
if state.exists():
    old=json.loads(state.read_text());pid=old.get("pid")
    if pid and (Path("/proc")/str(pid)/"cmdline").exists():
        cmd=(Path("/proc")/str(pid)/"cmdline").read_bytes().replace(b"\0",b" ").decode()
        if str(s/"plan.json") in cmd:raise RuntimeError("Existing dispatcher; do not duplicate: "+str(pid))
plan=json.loads((s/"plan.json").read_text());env={**os.environ,**plan["environment"],"PYTHONPATH":source["pythonpath"],"PYTHONDONTWRITEBYTECODE":"1"}
for name,folder in [("CUDA_CACHE_PATH","cuda"),("TRITON_CACHE_DIR","triton"),("TORCHINDUCTOR_CACHE_DIR","inductor")]:
    path=b/"cache"/(folder+"-stability-"+alias);path.mkdir(parents=True,exist_ok=True);env[name]=str(path)
py="/efs/prescottsun/project_jepa/envs/lewm/bin/python"
with (s/"workers"/("dispatcher-"+alias+".log")).open("ab") as log:
    proc=subprocess.Popen([py,"-B","-u",str(worker),"--plan",str(s/"plan.json"),"--host",alias],cwd=source["source"],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(2)
assert proc.poll() is None,"Dispatcher exited; inspect launch log"
record={"state":"launched","alias":alias,"host":socket.gethostname(),"pid":proc.pid,"at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"plan":str(s/"plan.json"),"source":source,"launcher_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(s/"workers"/("launch-"+alias+".json")).write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record))
