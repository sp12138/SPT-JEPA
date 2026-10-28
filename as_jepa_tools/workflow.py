"""Durable per-host experiment dispatch; training remains owned by existing trainers."""
import argparse, fcntl, hashlib, json, os, signal, socket, subprocess, time
from pathlib import Path

def read(path):
    try: return json.loads(Path(path).read_text())
    except (OSError, ValueError): return None

def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument("--plan",required=True);p.add_argument("--host",required=True);a=p.parse_args()
    planpath=Path(a.plan);plan=read(planpath);root=Path(plan["root"]);state=root/"workers";state.mkdir(exist_ok=True)
    lock=(state/(a.host+".lock")).open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    running={};finished={};hostname=socket.gethostname()
    while True:
        plan=read(planpath)
        for gpu,item in list(running.items()):
            rc=item["process"].poll()
            if rc is not None:
                job=item["job"];record=dict(id=job["id"],returncode=rc,started_at=item["started_at"],finished_at=time.time(),pid=item["process"].pid,host=hostname,gpu=gpu)
                atomic(root/"exits"/(job["id"]+".json"),record);item["log"].close();finished[job["id"]]=record;del running[gpu]
        stopping=(root/"HOLD_DISPATCH").exists()
        deadline=plan.get("deadline")
        if deadline is not None and time.time()>deadline:stopping=True
        jobs=[j for j in plan["jobs"] if j["host"]==a.host]
        if not stopping:
            for job in jobs:
                gpu=job["gpu"];ident=job["id"];output=Path(job["output"])
                if gpu in running or job.get("hold",False):continue
                if (root/"exits"/(ident+".json")).exists():continue
                if not all(Path(x).exists() for x in job.get("requires",[])):continue
                if not all((read(root/"exits"/(x+".json")) or {}).get("returncode")==0 for x in job.get("depends_on",[])):continue
                try:
                    apps=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
                    if apps:continue
                except subprocess.CalledProcessError:continue
                source=job["source"];manifest=read(source["manifest"])
                for relative,expected in manifest["files"].items():
                    actual=hashlib.sha256((Path(manifest["source"])/relative).read_bytes()).hexdigest()
                    if actual!=expected:raise RuntimeError("Frozen source changed: "+relative)
                if job.get("config_sha256"):
                    assert hashlib.sha256(Path(job["config"]).read_bytes()).hexdigest()==job["config_sha256"]
                output.mkdir(parents=True,exist_ok=True)
                env={**os.environ,**plan.get("environment",{}),**job.get("environment",{}),"CUDA_VISIBLE_DEVICES":str(gpu),"PYTHONPATH":source["pythonpath"],"PYTHONDONTWRITEBYTECODE":"1"}
                log=(output/"dispatch.log").open("ab")
                process=subprocess.Popen(job["command"],cwd=source["source"],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                started=time.time();running[gpu]=dict(job=job,process=process,log=log,started_at=started)
                atomic(output/"dispatch.json",dict(id=ident,host=hostname,gpu=gpu,pid=process.pid,started_at=started,command=job["command"],source=source))
        atomic(state/(a.host+".json"),dict(state="held" if stopping else "running",updated_at=time.time(),host=hostname,pid=os.getpid(),running={str(g):dict(id=v["job"]["id"],pid=v["process"].pid,started_at=v["started_at"]) for g,v in running.items()},queued=len(jobs)))
        time.sleep(10)

if __name__=="__main__":main()
