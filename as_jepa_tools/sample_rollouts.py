"""Sample noisy latent dynamics on the same held-out cases as mean-rollout evaluation."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from as_jepa_tools.data import PushTWindows
from as_jepa_tools.evaluate import load_model,pose_net,select_windows
from vforge.ops import initialize_ops,OpsConfig


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--data",required=True)
    p.add_argument("--pose",required=True);p.add_argument("--output",required=True);p.add_argument("--cases",type=int,default=256)
    p.add_argument("--steps",type=int,default=50);p.add_argument("--samples",type=int,default=4);p.add_argument("--fp32",action="store_true")
    a=p.parse_args();torch.set_num_threads(2)
    model=load_model(a.checkpoint)
    if model.mode not in {"L2","S2"}:raise ValueError("Latent sampling requires an AS-JEPA noisy prior")
    torch.set_float32_matmul_precision("highest" if a.fp32 or getattr(model,"fp32_pool",False) else "high")
    initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda",0),parallel_state=None)
    pose=pose_net(a.pose);data=PushTWindows(a.data,window=16,gap=8,split="test");data.open()
    ids=select_windows(data,a.cases);arrays={};cases=[];pictures=[];started=time.time()
    for case,index in enumerate(ids):
        batch=data[index];ep,start=data.locate(index)
        batch={k:v[None].cuda() if torch.is_tensor(v) else torch.tensor([v],device="cuda") for k,v in batch.items()}
        row=int(data.offsets[ep])+start-model.anchor_gap
        batch["anchor224"]=torch.from_numpy(np.array(data.p224[row])).permute(2,0,1)[None].cuda()
        batch["anchor"]=torch.from_numpy(np.array(data.p64[row])).permute(2,0,1)[None].cuda()
        batch["anchor_row"]=torch.tensor([row],device="cuda")
        generator=torch.Generator(device="cuda").manual_seed(20261027+int(index)*101)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=not a.fp32):
            pixels,frames,anchor=model.prepare(batch);mu=model.encode(dict(pixels=pixels[:,:3]))["emb"].float()
            states=mu.expand(a.samples,-1,-1)+model.sigma0*torch.randn((a.samples,*mu.shape[1:]),device="cuda",generator=generator)
            history=list(states.unbind(1));action=model.action_encoder(batch["action"].expand(a.samples,-1,-1))
            for horizon in range(13):
                lo=max(0,len(history)-(model.window-1))
                predicted=model.predict(torch.stack(history[lo:],1),action[:,lo:len(history)])[:,-1].float()
                history.append(predicted+model.sigma0*torch.randn(predicted.shape,device="cuda",generator=generator))
            future=torch.stack(history[3:],1);aa=anchor.expand(a.samples*13,-1,-1,-1)
            rendered=[]
            flat=future.flatten(0,1)
            for lo in range(0,len(flat),64):
                rendered.append(model.render_states(flat[lo:lo+64],aa[lo:lo+64],steps=a.steps,seed=31000+int(index)*13+lo))
            generated=torch.cat(rendered).float().reshape(a.samples,13,3,64,64).clamp(-1,1)
            estimate=pose(generated.flatten(0,1)).float().reshape(a.samples,13,6)
            copy=pose(frames[:,2]).float();truth=batch["state"][:,3:,:4].float()
            metrics=dict(agent=(estimate[:,:,:2]*512-truth[:,:,:2]).norm(dim=-1),
                         block=(estimate[:,:,2:4]*512-truth[:,:,2:4]).norm(dim=-1),
                         copy_agent=(copy[:,:2,None].transpose(1,2)*512-truth[:,:,:2]).norm(dim=-1).expand(a.samples,-1),
                         copy_block=(copy[:,2:4,None].transpose(1,2)*512-truth[:,:,2:4]).norm(dim=-1).expand(a.samples,-1),
                         pixel_mse=(generated-frames[:,3:]).square().mean((-3,-2,-1)))
        for key,value in metrics.items():arrays.setdefault(key,[]).append(value.cpu().numpy())
        state=batch["state"][0].float();moving=float((state[:,2:4]-state[0,2:4]).norm(dim=-1).max())>10
        cases.append(dict(episode=int(ep),start=int(start),window_index=int(index),moving_block=moving,latent_seed=20261027+int(index)*101))
        if len(pictures)<8:pictures.append(torch.cat([frames[0,3:4].float(),generated[:,0]],0).cpu())
    arrays={k:np.stack(v) for k,v in arrays.items()};moving=np.array([r["moving_block"] for r in cases])
    result=dict(checkpoint=str(Path(a.checkpoint).resolve()),cases=len(cases),samples=a.samples,steps=a.steps,
                feature_precision="FP32" if a.fp32 else "BF16",pose_weights=a.pose,seconds=time.time()-started,
                mean_by_horizon={k:v.mean((0,1)).tolist() for k,v in arrays.items()},
                h7_agent_ratio=float(arrays["agent"][:,:,6].mean()/arrays["copy_agent"][:,:,6].mean()),
                h7_block_ratio=float(arrays["block"][:,:,6].mean()/arrays["copy_block"][:,:,6].mean()),
                h7_moving_block_ratio=float(arrays["block"][moving,:,6].mean()/arrays["copy_block"][moving,:,6].mean()),
                scope="Four sampled latent rollouts per fixed case; context mu+sigma noise and each prior transition mean+sigma noise; independent renderer seeds; no best-of-samples selection")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/"metrics.json").write_text(json.dumps(result,indent=2))
    (out/"cases.json").write_text(json.dumps(cases,indent=2));np.savez(out/"per_case.npz",**arrays)
    from PIL import Image,ImageDraw
    tile=112;canvas=Image.new("RGB",((a.samples+1)*tile,30+len(pictures)*tile),"white");draw=ImageDraw.Draw(canvas)
    for j,label in enumerate(["Truth"]+[f"Sample {i+1}" for i in range(a.samples)]):draw.text((j*tile+3,8),label,fill="black")
    for i,row in enumerate(pictures):
        for j,frame in enumerate(row):
            rgb=((frame+1)*127.5).byte().permute(1,2,0).numpy()
            canvas.paste(Image.fromarray(rgb).resize((tile,tile)),(j*tile,30+i*tile))
    canvas.save(out/"gallery.png");print(json.dumps({k:v for k,v in result.items() if k!="mean_by_horizon"},indent=2))


if __name__=="__main__":main()
