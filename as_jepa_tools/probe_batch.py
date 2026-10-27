"""Evaluate several immutable checkpoints while importing the GPU stack once."""
import argparse,contextlib,json,time,traceback
from pathlib import Path
import torch
from as_jepa_tools.probes import main as probe
from vforge.ops import initialize_ops,OpsConfig


def main():
    p=argparse.ArgumentParser();p.add_argument("--jobs",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda",0),parallel_state=None)
    results=[]
    for job in json.loads(Path(a.jobs).read_text()):
        if (out/"STOP").exists():break
        dest=Path(job["output"]);dest.mkdir(parents=True,exist_ok=True);started=time.time();error=None
        args=["--checkpoint",job["checkpoint"],"--data",job["data"],"--output",str(dest),
              "--train",str(job.get("train",20000)),"--test",str(job.get("test",5000))]
        if job.get("noisy",False):args.append("--noisy")
        if job.get("fp32",False):args.append("--fp32")
        try:
            with (dest/"probe.log").open("w") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                probe(args,initialize=False)
        except Exception:
            error=traceback.format_exc();(dest/"error.txt").write_text(error)
        results.append(dict(**job,error=error,seconds=time.time()-started))
        (out/"status.json").write_text(json.dumps(dict(state="running",completed=len(results),results=results,updated_at=time.time()),indent=2))
        torch.cuda.empty_cache()
    (out/"status.json").write_text(json.dumps(dict(state="finished",completed=len(results),results=results,updated_at=time.time()),indent=2))


if __name__=="__main__":main()
