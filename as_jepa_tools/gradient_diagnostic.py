"""Measure flow/rate gradients through the state without performing an optimizer update."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from as_jepa_lewm.model import ASJepaLeWM
from as_jepa_tools.data import PushTWindows
from as_jepa_tools.evaluate import load_model
from vforge.ops import initialize_ops,OpsConfig


def stats(first,second):
    aa=torch.cat([g.detach().float().reshape(-1) for g in first if g is not None])
    bb=torch.cat([g.detach().float().reshape(-1) for g in second if g is not None])
    na=aa.norm();nb=bb.norm()
    return dict(flow_norm=float(na),priced_rate_norm=float(nb),
                rate_to_flow=float(nb/na) if na>0 else None,
                cosine=float(torch.dot(aa,bb)/(na*nb)) if na>0 and nb>0 else None)


def main():
    p=argparse.ArgumentParser();p.add_argument("--jobs",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda",0),parallel_state=None)
    results=[]
    for job in json.loads(Path(a.jobs).read_text()):
        started=time.time();torch.manual_seed(0)
        if job.get("checkpoint"):
            model=load_model(job["checkpoint"]);cfg=json.loads((Path(job["config"])).read_text())
        else:
            cfg=json.loads(Path(job["config"]).read_text());model=ASJepaLeWM(**cfg["model"]).cuda()
            if job.get("initial_scale"):
                scale=json.loads(Path(job["initial_scale"]).read_text())
                with torch.no_grad():
                    mean=torch.tensor(scale["mean_before"],device="cuda").reshape(4,16).mean(0)
                    model.pool.head.weight.mul_(scale["scale"]);model.pool.head.bias.sub_(mean).mul_(scale["scale"])
        if model.mode!="L2":raise ValueError("Gradient diagnostic currently covers the priced LeWM model")
        model.train();data=PushTWindows(cfg["data"],window=model.window,gap=cfg.get("gap",8),split="train")
        indices=np.random.default_rng(job.get("batch_seed",90)).choice(len(data),8,replace=False)
        batch=next(iter(DataLoader(Subset(data,indices),batch_size=8,num_workers=0)))
        batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        original=model.encode;captured=[]
        def capture(info):
            result=original(info);captured.append(result["emb"]);return result
        model.encode=capture
        torch.manual_seed(20261027)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss,metrics=model.loss(batch,step=job.get("step",0),ramp_steps=cfg.get("ramp_steps",0))
        mu=captured[0]
        params=[v for name,v in model.named_parameters() if name.startswith("pool.head.") or name.startswith("encoder.")]
        gf=torch.autograd.grad(metrics["fm"],[mu]+params,retain_graph=True,allow_unused=True)
        beta=model.beta*min(1.,job.get("step",0)/max(1,cfg.get("ramp_steps",0))) if cfg.get("ramp_steps",0) else model.beta
        gr=torch.autograd.grad(beta*metrics["rate"],[mu]+params,allow_unused=True)
        result=dict(id=job["id"],batch_seed=job.get("batch_seed",90),step=job.get("step",0),checkpoint=job.get("checkpoint"),
                    state_std=float(mu.float().std((0,1)).mean()),sigma0=model.sigma0,
                    fm=float(metrics["fm"]),rate=float(metrics["rate"]),beta=beta,
                    state_gradient=stats(gf[:1],gr[:1]),encoder_head_gradient=stats(gf[1:],gr[1:]),
                    seconds=time.time()-started,scope="Single fixed train minibatch, BF16 forward, no optimizer update")
        results.append(result);(out/"gradients.json").write_text(json.dumps(results,indent=2))
        del model,mu,loss,metrics,gf,gr,params,captured
        torch.cuda.empty_cache()
    print(json.dumps(results,indent=2))


if __name__=="__main__":main()
