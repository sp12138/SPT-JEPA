"""Bounded evaluator supervisor; takes a recorded command without a shell."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


plan = json.loads(Path(sys.argv[1]).read_text())
job = plan["jobs"][int(sys.argv[2])]
start = time.time()
result = {"name": job["name"], "supervisor_pid": os.getpid()}
try:
    env = dict(os.environ)
    env.update(job["environment"])
    with open(job["log"], "w") as log:
        process = subprocess.Popen(job["command"], cwd=job["cwd"], env=env,
                                   stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        result["evaluator_pid"] = process.pid
        Path(job["exit"] + ".started").write_text(json.dumps(result) + "\n")
        try:
            result["returncode"] = process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            result["returncode"] = 124
except Exception:
    result["returncode"] = 125
    result["exception"] = traceback.format_exc()
finally:
    result["seconds"] = time.time() - start
    Path(job["exit"]).write_text(json.dumps(result) + "\n")
