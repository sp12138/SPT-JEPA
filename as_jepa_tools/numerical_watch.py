"""Stop only an identified campaign-owned training process after nonfinite step logs."""
import argparse
import fcntl
import json
import math
import os
import signal
import time
from pathlib import Path


def bad_tail(rows):
    steps=[x for x in rows if "loss" in x and "step" in x]
    return len(steps)>=2 and all(not math.isfinite(float(row["loss"])) for row in steps[-2:])


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return None


def atomic(path,value):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2))
    temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument("--plan",required=True);p.add_argument("--host",required=True);p.add_argument("--once",action="store_true");a=p.parse_args()
    root=Path(a.plan).parent
    lock=(root/"workers"/("numerics-"+a.host+".lock")).open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        plan=read(a.plan);worker=read(root/"workers"/(a.host+".json"))
        if plan and worker and time.time()-worker["updated_at"]<60:
            byid={j["id"]:j for j in plan["jobs"]}
            for active in worker["running"].values():
                job=byid.get(active["id"])
                if not job or not job.get("config") or job["host"]!=a.host:continue
                output=Path(job["output"])
                if root not in output.parents:raise RuntimeError("Out-of-campaignoutput")
                logs=output/"train_steps.jsonl"
                if not logs.exists():logs=output/"metrics.jsonl"
                if not logs.exists():continue
                # Last64KB contains at least two project step records; ignore incomplete last write.
                with logs.open("rb") as stream:
                    stream.seek(max(0,logs.stat().st_size-65536))
                    lines=stream.read().decode(errors="replace").splitlines()
                rows=[]
                for line in lines:
                    try:rows.append(json.loads(line))
                    except ValueError:pass
                if not bad_tail(rows):continue
                pid=active["pid"]
                try:actual=Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
                except OSError:continue
                expected=job["command"]
                if actual[1:]!=expected[1:]:continue
                if os.getpgid(pid)!=pid:continue
                marker=output/"numerical-stop.json"
                if marker.exists():continue
                record=dict(job=job["id"],host=a.host,pid=pid,command=actual,detected_at=time.time(),
                            reason="Lasttwologgedlossesnonfinite;source/confignotmodified;lastfinitecheckpointsremainpreserved.",
                            last_records=[x for x in rows if "loss" in x and "step" in x][-2:],
                            action="SIGTERMidentifiedcampaignprocessgrouponly")
                atomic(marker,record)
                os.killpg(pid,signal.SIGTERM)
        atomic(root/"workers"/("numerics-"+a.host+"-heartbeat.json"),dict(updated_at=time.time(),pid=os.getpid(),host=a.host))
        if a.once:break
        time.sleep(15)


if __name__=="__main__":
    main()
