"""Matched posterior state/anchor swaps on episode-colour PushT."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from as_jepa_tools.data import PushTWindows
from as_jepa_tools.evaluate import load_model,pose_net,select_windows
from vforge.ops import initialize_ops,OpsConfig


def background(images):
    size=8
    corners=torch.cat([images[:,:,:size,:size].flatten(2),images[:,:,:size,-size:].flatten(2),
                       images[:,:,-size:,:size].flatten(2),images[:,:,-size:,-size:].flatten(2)],-1)
    return (corners.float().median(-1).values+1)*127.5


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--data",required=True)
    p.add_argument("--pose",required=True);p.add_argument("--output",required=True);p.add_argument("--cases",type=int,default=128)
    p.add_argument("--steps",type=int,default=50);p.add_argument("--fp32",action="store_true");a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(55);torch.set_float32_matmul_precision("highest" if a.fp32 else "high")
    initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda",0),parallel_state=None)
    model=load_model(a.checkpoint)
    if model.domain!="color":raise ValueError("Colour swap protocol requires domain=color")
    pose=pose_net(a.pose);data=PushTWindows(a.data,window=16,gap=8,split="test");data.open()
    ids=select_windows(data,a.cases);loader=DataLoader(Subset(data,ids),batch_size=8,num_workers=2,pin_memory=True)
    values={};records=[];pictures=[];started=time.time()
    def save(key,value):values.setdefault(key,[]).append(value.detach().float().cpu().numpy())
    for batch_index,batch in enumerate(loader):
        batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
        rows=[int(data.offsets[int(ep)])+int(start)-model.anchor_gap for ep,start in zip(batch["episode"].cpu(),batch["start"].cpu())]
        batch["anchor224"]=torch.from_numpy(np.array(data.p224[rows])).permute(0,3,1,2).cuda()
        batch["anchor"]=torch.from_numpy(np.array(data.p64[rows])).permute(0,3,1,2).cuda()
        batch["anchor_row"]=torch.tensor(rows,device="cuda")
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=not a.fp32):
            pixels,frames,anchor=model.prepare(batch);mu=model.encode(dict(pixels=pixels))["emb"][:,3]
            own=model.render_states(mu,anchor,steps=a.steps,seed=910+batch_index)
            anchor_swap=model.render_states(mu,anchor.flip(0),steps=a.steps,seed=910+batch_index)
            state_swap=model.render_states(mu.flip(0),anchor,steps=a.steps,seed=910+batch_index)
            noise=torch.randn(mu.shape,device=mu.device,generator=torch.Generator(device=mu.device).manual_seed(710+batch_index))
            noisy=model.render_states(mu.float()+model.sigma0*noise,anchor,steps=a.steps,seed=910+batch_index)
            poses={k:pose(v.clamp(-1,1)).float()[:,:4]*512 for k,v in [("own",own),("anchor_swap",anchor_swap),("state_swap",state_swap),("noisy",noisy)]}
        own_bg=background(anchor);donor_bg=background(anchor.flip(0));own_render_bg=background(own)
        for key,image in [("own",own),("anchor_swap",anchor_swap),("state_swap",state_swap),("noisy",noisy)]:
            bg=background(image)
            save(key+"_bg",bg);save(key+"_bg_own_error",(bg-own_bg).norm(dim=-1))
            save(key+"_bg_donor_error",(bg-donor_bg).norm(dim=-1))
            save(key+"_agent_error",(poses[key][:,:2]-batch["state"][:,3,:2].float()).norm(dim=-1))
            save(key+"_block_error",(poses[key][:,2:]-batch["state"][:,3,2:4].float()).norm(dim=-1))
            save(key+"_image_change",(image.float()-own.float()).square().mean((1,2,3)))
        delta=donor_bg-own_bg;denom=delta.square().sum(-1)
        save("palette_distance",delta.norm(dim=-1))
        save("anchor_style_transfer_projection",((background(anchor_swap)-own_render_bg)*delta).sum(-1)/denom.clamp_min(1))
        save("state_style_transfer_projection",((background(state_swap)-own_render_bg)*delta).sum(-1)/denom.clamp_min(1))
        save("anchor_swap_agent_displacement",(poses["anchor_swap"][:,:2]-poses["own"][:,:2]).norm(dim=-1))
        save("state_swap_agent_displacement",(poses["state_swap"][:,:2]-poses["own"][:,:2]).norm(dim=-1))
        if len(pictures)<16:
            for i in range(min(len(anchor),16-len(pictures))):
                pictures.append(torch.stack([anchor[i],anchor.flip(0)[i],frames[i,3],own[i],anchor_swap[i],state_swap[i],noisy[i]]).float().cpu())
        records.extend([dict(episode=int(ep),start=int(start)) for ep,start in zip(batch["episode"].cpu(),batch["start"].cpu())])
    arrays={k:np.concatenate(v) for k,v in values.items()};eligible=arrays["palette_distance"]>30
    report=dict(checkpoint=str(Path(a.checkpoint).resolve()),cases=len(records),steps=a.steps,
                feature_precision="FP32" if a.fp32 else "BF16",palette_pairs_over_30=int(eligible.sum()),
                mean={k:float(v.mean()) for k,v in arrays.items() if v.ndim==1},
                anchor_style_transfer_projection=float(arrays["anchor_style_transfer_projection"][eligible].mean()),
                state_style_transfer_projection=float(arrays["state_style_transfer_projection"][eligible].mean()),
                seconds=time.time()-started,
                scope="Common posterior states and renderer noise; reversed within-batch donors; corner-median background measures appearance, pose is independent v2")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/"swaps.json").write_text(json.dumps(report,indent=2))
    (out/"cases.json").write_text(json.dumps(records,indent=2));np.savez(out/"per_case.npz",**arrays)
    from PIL import Image,ImageDraw
    tile=112;canvas=Image.new("RGB",(tile*7,tile*len(pictures)+30),"white");draw=ImageDraw.Draw(canvas)
    for j,label in enumerate(["own anchor","donor anchor","truth","own state","swap anchor","swap state","noisy state"]):draw.text((j*tile+2,8),label,fill="black")
    for i,row in enumerate(pictures):
        for j,frame in enumerate(row):
            rgb=((frame.clamp(-1,1)+1)*127.5).byte().permute(1,2,0).numpy()
            canvas.paste(Image.fromarray(rgb).resize((tile,tile)),(j*tile,30+i*tile))
    canvas.save(out/"gallery.png");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
