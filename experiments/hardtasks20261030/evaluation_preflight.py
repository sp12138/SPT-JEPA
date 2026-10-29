"""Two-case execution gate; it makes no performance claim."""
from pathlib import Path
import json,os,subprocess,sys,hashlib,time
b=Path("/efs/prescottsun/SPT-JEPA");root=b/"runs/hardtasks_gate600_20261030";source=root/"source-v1";data=b/"data/hardtasks-v1/cube-metadata";out=root/"evaluation-preflight-cube";out.mkdir(exist_ok=True)
env={**os.environ,"MUJOCO_GL":"egl","PYOPENGL_PLATFORM":"egl","LD_LIBRARY_PATH":str(b/"environments/egl-lib/usr/lib/x86_64-linux-gnu")+":/usr/lib64:"+os.environ.get("LD_LIBRARY_PATH",""),"__EGL_VENDOR_LIBRARY_FILENAMES":str(b/"environments/egl-lib/10_nvidia.json"),"MUJOCO_EGL_DEVICE_ID":os.environ.get("CUDA_VISIBLE_DEVICES","0")}
records=[]
for method in ["random","released-lewm"]:
 path=out/method;path.mkdir(exist_ok=False);checkpoint="random" if method=="random" else str(b/"weights/hardtasks/cube")
 command=[sys.executable,"-B","-u","-m","as_jepa_tools.benchmark_evaluate","--manifest",str(data/"manifests/validation-strict.json"),"--data",str(data),"--checkpoint",checkpoint,"--output",str(path),"--limit","2"]
 with (path/"evaluation.log").open("w") as log:result=subprocess.run(command,env=env,cwd=source,stdout=log,stderr=subprocess.STDOUT)
 record=dict(method=method,returncode=result.returncode);(path/"exit.json").write_text(json.dumps(record));records.append(record)
 if result.returncode:raise SystemExit(result.returncode)
 metrics=json.loads((path/"metrics.json").read_text());assert len(metrics["metrics"]["episode_successes"])==2
 if method!="random":assert metrics["checkpoint_audit"]["strict_equal"]
marker=b/"reports/harder-task-redirect-20261030/CUBE-EVALUATOR-PREFLIGHT-PASSED.json";marker.write_text(json.dumps(dict(time=time.time(),records=records,scope="Execution only; no performance inference from two cases"),indent=2))
