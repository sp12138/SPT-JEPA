"""Separate prior forecast error from reconstruction with the true future state."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from as_jepa_tools.data import PushTWindows
from as_jepa_tools.evaluate import load_model,predict_states,pose_net,select_windows
from vforge.ops import initialize_ops,OpsConfig


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--data",required=True)
    p.add_argument("--pose",required=True);p.add_argument("--output",required=True);p.add_argument("--cases",type=int,default=128)
    p.add_argument("--steps",type=int,default=50);p.add_argument("--fp32",action="store_true");a=p.parse_args()
    cfg=json.loads((Path(a.checkpoint)/"config.json").read_text())
    torch.set_num_threads(2);torch.manual_seed(20261027)
    torch.set_float32_matmul_precision("highest" if a.fp32 or cfg.get("fp32_pool",False) else "high")
    initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda",0),parallel_state=None)
    model=load_model(a.checkpoint);pose=pose_net(a.pose)
    if model.mode not in {"L1","L2"} or model.domain!="standard":raise ValueError("This diagnostic currently covers standard-domain LeWM renderers")
    data=PushTWindows(a.data,window=16,gap=8,split="test");data.open();indices=select_windows(data,a.cases)
    loader=DataLoader(Subset(data,indices),batch_size=8,num_workers=2,pin_memory=True)
    horizons=[1,7,13];future=[h-1 for h in horizons];frames_idx=[h+2 for h in horizons]
    values={};records=[];images=[];started=time.time()
    def append(name,value):values.setdefault(name,[]).append(value.float().cpu().numpy())
    for batch_index,batch in enumerate(loader):
        batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
        rows=[int(data.offsets[int(ep)])+int(start)-model.anchor_gap for ep,start in zip(batch["episode"].cpu(),batch["start"].cpu())]
        batch["anchor224"]=torch.from_numpy(np.array(data.p224[rows])).permute(0,3,1,2).cuda()
        batch["anchor"]=torch.from_numpy(np.array(data.p64[rows])).permute(0,3,1,2).cuda();batch["anchor_row"]=torch.tensor(rows,device="cuda")
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=not a.fp32):
            pixels,frames,anchor=model.prepare(batch);mu=model.encode(dict(pixels=pixels))["emb"][:,frames_idx].float()
            predicted=predict_states(model,pixels[:,:3],batch["action"],13)[:,future]
            condition={"posterior_mean":mu,"forecast_mean":predicted,"zero_state":torch.zeros_like(mu)}
            if model.mode=="L2":
                generator=torch.Generator(device="cuda").manual_seed(980+batch_index)
                condition["posterior_noisy"]=mu+model.sigma0*torch.randn(mu.shape,device="cuda",generator=generator)
            rendered={}
            target=batch["state"][:,frames_idx,:4].float()
            for name,states in condition.items():
                generated=model.render_states(states.flatten(0,1),anchor.repeat_interleave(3,0),steps=a.steps,seed=1250+batch_index).reshape(len(mu),3,3,64,64).float().clamp(-1,1)
                rendered[name]=generated
                position=pose(generated.flatten(0,1)).float().reshape(len(mu),3,6)[:,:,:4]*512
                append(name+"_agent",(position[:,:,:2]-target[:,:,:2]).norm(dim=-1))
                append(name+"_block",(position[:,:,2:4]-target[:,:,2:4]).norm(dim=-1))
                append(name+"_pixel_mse",(generated-frames[:,frames_idx]).square().mean((-3,-2,-1)))
            copy=frames[:,2:3].expand(-1,3,-1,-1,-1)
            copy_pose=pose(copy.flatten(0,1)).float().reshape(len(mu),3,6)[:,:,:4]*512
            real_pose=pose(frames[:,frames_idx].flatten(0,1)).float().reshape(len(mu),3,6)[:,:,:4]*512
            for name,position in [("copy",copy_pose),("real_floor",real_pose)]:
                append(name+"_agent",(position[:,:,:2]-target[:,:,:2]).norm(dim=-1))
                append(name+"_block",(position[:,:,2:4]-target[:,:,2:4]).norm(dim=-1))
        if len(images)<8:
            for i in range(min(len(mu),8-len(images))):
                images.append(torch.stack([frames[i,frames_idx[0]],rendered["posterior_mean"][i,0],rendered["forecast_mean"][i,0],
                                          frames[i,frames_idx[1]],rendered["posterior_mean"][i,1],rendered["forecast_mean"][i,1],
                                          rendered.get("posterior_noisy",rendered["posterior_mean"])[i,1],rendered["zero_state"][i,1]]).cpu())
        records.extend([dict(episode=int(ep),start=int(start)) for ep,start in zip(batch["episode"].cpu(),batch["start"].cpu())])
    arrays={k:np.concatenate(v) for k,v in values.items()}
    report=dict(checkpoint=str(Path(a.checkpoint).resolve()),cases=len(records),horizons=horizons,steps=a.steps,
                feature_precision="FP32" if a.fp32 else "BF16",pose_weights=a.pose,seconds=time.time()-started,
                mean_error={key:value.mean(0).tolist() for key,value in arrays.items()},
                scope="True-future posterior states are an oracle diagnostic, not deployable rollout; forecast and posterior share anchor and renderer noise; no optimization")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/"decomposition.json").write_text(json.dumps(report,indent=2))
    (out/"cases.json").write_text(json.dumps(records,indent=2));np.savez(out/"per_case.npz",**arrays)
    from PIL import Image,ImageDraw
    tile=112;canvas=Image.new("RGB",(tile*8,30+len(images)*tile),"white");draw=ImageDraw.Draw(canvas)
    for j,label in enumerate(["Truth h1","Posterior h1","Forecast h1","Truth h7","Posterior h7","Forecast h7","Noisy post h7","Zero state h7"]):draw.text((j*tile+2,8),label,fill="black")
    for i,row in enumerate(images):
        for j,frame in enumerate(row):
            rgb=((frame.float().clamp(-1,1)+1)*127.5).byte().permute(1,2,0).numpy()
            canvas.paste(Image.fromarray(rgb).resize((tile,tile)),(j*tile,30+i*tile))
    canvas.save(out/"gallery.png");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
