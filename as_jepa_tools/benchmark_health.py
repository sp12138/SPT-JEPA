"""Full-state audit and matched-noise state/forecast diagnostics on validation episodes."""
import argparse,json,math
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,Subset
from as_jepa_tools.benchmark_data import BenchmarkWindows
from as_jepa_lewm.model import ASJepaLeWM
from as_jepa_lewm.robust import RobustLeWM
from vforge.models.as_jepa.pixels import pack
from vforge.ops import initialize_ops,OpsConfig

def finite(x):
    if torch.is_tensor(x):return bool(torch.isfinite(x).all())
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,(list,tuple)):return all(finite(v) for v in x)
    if isinstance(x,float):return math.isfinite(x)
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--output",required=True);p.add_argument("--cases",type=int,default=16);a=p.parse_args()
    torch.set_num_threads(2);root=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((root/"config.json").read_text());state=torch.load(root/"last.ckpt",map_location="cpu",weights_only=False);step=int(state["global_step"])
    weights=torch.load(root/f"model-step{step:07d}/weights.pt",map_location="cpu",weights_only=True)
    audit=dict(step=step,model_finite=finite(state["state_dict"]),optimizer_finite=finite(state["optimizer_states"]),scheduler_finite=finite(state["lr_schedulers"]),cursor=state["campaign_samples"]==step*cfg["batch_size"],rng_present=bool(state.get("campaign_rng")),optimizer_present=bool(state["optimizer_states"]),scheduler_present=bool(state["lr_schedulers"]),portable_equal=all(torch.equal(v,state["state_dict"]["model."+k]) for k,v in weights.items()))
    assert all(v for k,v in audit.items() if k!="step"),audit
    trace=[json.loads(x) for x in (root/"metrics.jsonl").read_text().splitlines()];assert all(finite(x) for x in trace)
    audit["gradient_max"]={k:max(x.get("grad_norm_"+k,0.) for x in trace) for k in ["encoder","predictor","renderer"]};assert all(v>0 for v in audit["gradient_max"].values())
    del state
    initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda"),parallel_state=None)
    model=(RobustLeWM if cfg.get("robust_variant") else ASJepaLeWM)(**cfg["model"]).cuda().eval();model.load_state_dict(weights,strict=True);del weights
    ds=BenchmarkWindows(cfg["data"],window=cfg["model"]["window"],split="validation")
    ids=np.random.default_rng(1042).choice(len(ds),a.cases,replace=False);loader=DataLoader(Subset(ds,ids.tolist()),batch_size=2,num_workers=0)
    rows=[];embeddings=[]
    with torch.no_grad():
        for batch in loader:
            batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()};torch.manual_seed(1042+len(rows))
            pixels,render,anchor=model.prepare(batch);info=model.encode(dict(pixels=pixels,action=batch["action"]));mu=info["emb"];b,w,d=mu.shape
            noise=torch.randn_like(mu);rho=getattr(model,"rho",0.)
            for t in range(1,w):noise[:,t]=rho*noise[:,t-1]+math.sqrt(1-rho*rho)*noise[:,t]
            noisy=mu+model.sigma0*noise
            def forecast(actions, first=None):
                acts=model.action_encoder(actions);history=[noisy[:,0] if first is None else first]
                for t in range(w-1):history.append(model.predict(torch.stack(history,1),acts[:,:t+1])[:,-1])
                return torch.stack(history[1:],1)
            prior=forecast(batch["action"]);wrong=forecast(torch.roll(batch["action"],1,1))
            wrong_state=forecast(batch["action"],mu.flip(0)[:,0]+model.sigma0*noise[:,0])
            y=pack(render[:,1:].flatten(0,1));anc=pack(anchor).repeat_interleave(w-1,0);eps=torch.randn_like(y)
            conditions=dict(posterior=noisy[:,1:],shuffled_posterior=mu.flip(0)[:,1:]+model.sigma0*noise[:,1:],forecast=prior,wrong_state=wrong_state,wrong_actions=wrong,carry=noisy[:,:1].expand(-1,w-1,-1),zero=torch.zeros_like(prior))
            values={}
            for level in [.2,.5,.8,.95]:
                t=torch.full((len(y),1),level,device=y.device)
                for name,z in conditions.items():
                    z=z.reshape(-1,model.state_slots,model.state_width)
                    with model.renderer.anchored(anc):v=model.renderer((1-level)*y+level*eps,t,condition=z)
                    values[f"fm_t{level}_{name}"]=(y-eps-v).square().flatten(1).mean(1).reshape(b,w-1).mean(1).cpu().tolist()
            for i in range(b):rows.append(dict(episode=int(batch["episode"][i]),start=int(batch["start"][i]),**{k:v[i] for k,v in values.items()}))
            embeddings.append(mu.cpu().flatten(0,1))
    emb=torch.cat(embeddings);report=dict(audit=audit,cases=rows,summary={k:float(np.mean([x[k] for x in rows])) for k in rows[0] if k.startswith("fm_")},feature_std=float(emb.std(0).mean()),feature_signal_to_posterior_noise=float(emb.std(0).mean())/model.sigma0,parameters=sum(p.numel() for p in model.parameters()),inference_shift=model.shift,training_shift=model.training_shift,scope="Validation-only matched-noise diagnostics. State permutations preserve posterior-noise draws; zero-state sensitivity alone can be a constant-bias effect. Numerical health is not learned control or generation success.")
    assert finite(report);(out/"health.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({k:v for k,v in report.items() if k!="cases"}),flush=True)
if __name__=="__main__":main()
