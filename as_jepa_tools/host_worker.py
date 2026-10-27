"""Persistent per-host GPU queue with a hard campaign deadline."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path


def atomic(path,value):
    path=Path(path);temporary=path.with_suffix(".tmp");temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument("--plan",required=True);p.add_argument("--host",required=True);a=p.parse_args();planpath=Path(a.plan);plan=json.loads(planpath.read_text());base=Path(plan["root"]);host=a.host
    record=base/"runs"/plan["campaign"]/f"worker-{host}.json";running={};finished={};deadline=plan["deadline"]
    # An exclusive lock avoids duplicate schedulers on replacement or repeated commands.
    import fcntl
    lock=record.with_suffix(".lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    envbase={**os.environ,"PYTHONPATH":plan["pythonpath"],"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","PYTHONDONTWRITEBYTECODE":"1","HF_HOME":str(base/"cache/huggingface"),"MPLCONFIGDIR":str(base/"cache/matplotlib"),"USE_FLAX":"0","USE_TF":"0","WANDB_MODE":"disabled","TOKENIZERS_PARALLELISM":"false"}
    while True:
        plan=json.loads(planpath.read_text());jobs=[j for j in plan["jobs"] if j["host"]==host]
        for gpu,job in list(running.items()):
            code=job["process"].poll()
            if code is not None:
                finished[job["id"]]=dict(returncode=code,finished=time.time(),started=job["started"],gpu=gpu,pid=job["process"].pid)
                atomic(base/"runs"/plan["campaign"]/f"exit-{job['id']}.json",finished[job["id"]]);job["log"].close();del running[gpu]
        if time.time()>=deadline:
            for job in running.values():
                output=Path(job["output"]);output.mkdir(parents=True,exist_ok=True);(output/"STOP").write_text("campaign deadline")
            # Leave a short grace period for the training callback to save its full checkpoint.
            grace=time.time()+180
            while running and time.time()<grace:
                for gpu,job in list(running.items()):
                    if job["process"].poll() is not None:job["log"].close();del running[gpu]
                if running:time.sleep(2)
            for job in running.values():os.killpg(job["process"].pid,signal.SIGTERM)
            atomic(record,dict(state="deadline_reached",finished=finished,updated_at=time.time()));break
        if (base/"data/pusht/ready.json").exists():
            used=set(running)
            for job in jobs:
                ident=job["id"];gpu=job["gpu"]
                if ident in finished or gpu in used or any(x["id"]==ident for x in running.values()):continue
                if job.get("hold",False):continue
                output=Path(job["output"]);output.mkdir(parents=True,exist_ok=True)
                if (output/"status.json").exists() and json.loads((output/"status.json").read_text()).get("state")=="finished":finished[ident]=dict(returncode=0,previous=True);continue
                env={**envbase,"CUDA_VISIBLE_DEVICES":str(gpu)};log=(output/"launcher.log").open("ab")
                command=job["command"];process=subprocess.Popen(command,env=env,cwd=plan["source"],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                started=time.time();running[gpu]=dict(id=ident,process=process,log=log,started=started,output=str(output));used.add(gpu)
                atomic(output/"execution.json",dict(pid=process.pid,host=host,gpu=gpu,started_at=started,command=command,source_commit=plan["source_commit"],source=plan["source"],deadline=deadline))
        atomic(record,dict(state="running",running={str(g):dict(id=j["id"],pid=j["process"].pid,started=j["started"],output=j["output"]) for g,j in running.items()},finished=finished,updated_at=time.time()))
        # Keep the worker alive for adaptive jobs added to the same persistent plan.
        time.sleep(10)

if __name__=="__main__":main()
