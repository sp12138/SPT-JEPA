"""Audit Native feedback drift using the existing sampler and true past frames."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from as_jepa_tools.data import PushTWindows
from as_jepa_tools.evaluate import load_model,pose_net,select_windows,gallery
from as_jepa_tools.summarize_final import interval_ratio
from vforge.models.as_jepa.pixels import pack
from vforge.ops import initialize_ops,OpsConfig


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for name in ["checkpoint","baseline","data","pose","output"]:p.add_argument("--"+name,required=True)
    p.add_argument("--cases",type=int,default=256);p.add_argument("--steps",type=int,default=50);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.set_float32_matmul_precision("high")
    initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda",0),parallel_state=None)
    model=load_model(a.checkpoint);assert model.mode in ["S0","S1"];pose=pose_net(a.pose)
    data=PushTWindows(a.data,window=16,gap=8,split="test");indices=select_windows(data,256)[:a.cases]
    cases=[dict(episode=int(data.locate(i)[0]),start=int(data.locate(i)[1])) for i in indices]
    baseline=Path(a.baseline);baseline_cases=json.loads((baseline/"cases.json").read_text())
    assert cases==baseline_cases[:len(cases)],"Control case identities differ"
    baseline_metric=json.loads((baseline/"metrics.json").read_text())
    assert baseline_metric["sampler_steps"]==a.steps and baseline_metric["pose_weights"]==str(Path(a.pose))
    base=np.load(baseline/"per_case.npz");loader=DataLoader(Subset(data,indices),batch_size=4,num_workers=2,pin_memory=True)
    values={};images=[];started=time.time()
    for batch_id,batch in enumerate(loader):
        batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
        with torch.autocast("cuda",dtype=torch.bfloat16):
            _,frames,_=model.prepare(batch);anchor=frames[:,0];original=model.causal_semantic_encoder.next_context;counter=[0]
            def true_context(source):
                target=3+counter[0];lo=max(0,target-(model.window-1));truth=frames[:,lo:target]
                latents=pack(truth.flatten(0,1)).reshape(len(frames),target-lo,48,16,16).transpose(1,2)
                assert latents.shape==source.shape
                counter[0]+=1
                return original(latents)
            model.causal_semantic_encoder.next_context=true_context
            try:
                generated=model.rollout_frames(frames[:,:3],batch["action"],anchor,horizon=13,steps=a.steps,seed=1000+batch_id).float().clamp(-1,1)
            finally:model.causal_semantic_encoder.next_context=original
            assert counter[0]==13
            pred=pose(generated.flatten(0,1)).float().reshape(len(frames),13,6)*512
            truth=batch["state"][:,3:,:4].float()
            for key,metric in [("agent",(pred[:,:,:2]-truth[:,:,:2]).norm(dim=-1)),
                               ("block",(pred[:,:,2:4]-truth[:,:,2:4]).norm(dim=-1)),
                               ("pixel_mse",(generated-frames[:,3:]).square().mean((-3,-2,-1)))]:
                values.setdefault(key,[]).append(metric.cpu().numpy())
            if len(images)<8:images.extend([(frames[i].cpu(),generated[i].cpu(),anchor[i].cpu()) for i in range(min(len(frames),8-len(images)))])
    arrays={k:np.concatenate(v) for k,v in values.items()};episodes=[r["episode"] for r in cases]
    h1={entity:float(np.max(np.abs(arrays[entity][:,0]-base[entity][:len(cases),0]))) for entity in ["agent","block"]}
    control=dict(max_h1_error_difference_px=h1,tolerance_px=.5,passed=all(x<=.5 for x in h1.values()))
    report=dict(model=model.mode,checkpoint=str(Path(a.checkpoint).resolve()),baseline=str(baseline),pose_weights=a.pose,cases=len(cases),steps=a.steps,
                seconds=time.time()-started,precision="BF16 autocast; FP32 matmul high",control=control,horizons={},
                intervention="Only next_context input history is replaced by true past frames; actions, initial anchor and the existing 13-step sampler/RNG are unchanged",
                scope="Teacher forcing is an oracle diagnostic, not deployable rollout. One trained seed; common cases and sampling noise.")
    for h in [1,3,7,13]:
        report["horizons"][str(h)]={}
        for entity in ["agent","block"]:
            numerator=arrays[entity][:,h-1];denominator=base["copy_"+entity][:len(cases),h-1]
            autoregressive=base[entity][:len(cases),h-1]
            report["horizons"][str(h)][entity]=dict(teacher_forced_px=float(numerator.mean()),autoregressive_px=float(autoregressive.mean()),
                copy_px=float(denominator.mean()),teacher_forced_ratio_ci=interval_ratio(numerator,denominator,episodes),
                autoregressive_ratio_ci=interval_ratio(autoregressive,denominator,episodes))
    (out/"teacher_forcing.json").write_text(json.dumps(report,indent=2));(out/"cases.json").write_text(json.dumps(cases,indent=2))
    np.savez(out/"per_case.npz",**arrays);gallery(images,out)
    print(json.dumps(dict(model=model.mode,cases=len(cases),seconds=report["seconds"],control=control,horizons=report["horizons"]),indent=2))
    if not control["passed"]:raise RuntimeError("One-step matched-noise control failed")


if __name__=="__main__":main()
