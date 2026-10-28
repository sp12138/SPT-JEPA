"""Matched mean-state anchor swaps with complete videos and pose metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from as_jepa_tools.data import PushTWindows
from as_jepa_tools.evaluate import load_model, predict_states, pose_net
from as_jepa_tools.quality_complete import video
from vforge.ops import OpsConfig, initialize_ops


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for key in ["checkpoint","data","cases","pose","output"]:p.add_argument("--"+key,required=True)
    p.add_argument("--pairs",type=int,default=8)
    p.add_argument("--steps",type=int,default=50)
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/"metrics.json").exists():raise RuntimeError("Preserve completedswapoutput")
    torch.set_num_threads(2);torch.manual_seed(42);torch.set_float32_matmul_precision("highest")
    initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda",0),parallel_state=None)
    cfg=json.loads((Path(a.checkpoint)/"config.json").read_text())
    model=load_model(a.checkpoint)
    if model.mode=="S0":
        (out/"metrics.json").write_text(json.dumps(dict(applicable=False,reason="S0hasnoanchor",pairs=0),indent=2));return
    data=PushTWindows(a.data,window=16,gap=8,split="test");data.open();pose=pose_net(a.pose)
    candidates=json.loads(Path(a.cases).read_text());selected=[];episodes=set()
    for record in candidates:
        if record["episode"] in episodes:continue
        selected.append(record);episodes.add(record["episode"])
        if len(selected)>=a.pairs*2:break
    if len(selected)<a.pairs*2:raise RuntimeError("Insufficientdistinctepisodes")
    weights_hash=hashlib.sha256((Path(a.checkpoint)/"weights.pt").read_bytes()).hexdigest()
    def batch(record):
        item=data[int(record["index"])]
        value={key:value[None].cuda() if torch.is_tensor(value) else torch.tensor([value],device="cuda") for key,value in item.items()}
        anchor_row=int(data.offsets[int(record["episode"])])+int(record["start"])-model.anchor_gap
        value["anchor224"]=torch.from_numpy(np.array(data.p224[anchor_row])).permute(2,0,1)[None].cuda()
        value["anchor"]=torch.from_numpy(np.array(data.p64[anchor_row])).permute(2,0,1)[None].cuda()
        value["anchor_row"]=torch.tensor([anchor_row],device="cuda")
        return value,anchor_row
    results=[]
    def rgb(value):
        return ((value.float().clamp(-1,1)+1)*127.5).byte().permute(0,2,3,1).cpu().numpy()
    for pair in range(a.pairs):
        appearance,dynamics=selected[pair*2:pair*2+2]
        donor,donor_anchor_row=batch(appearance);source,source_anchor_row=batch(dynamics)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            _,donor_frames,donor_anchor=model.prepare(donor)
            pixels,frames,own_anchor=model.prepare(source)
            if model.mode=="S1":
                own=model.rollout_frames(frames[:,:3],source["action"],own_anchor,13,a.steps,81000+pair)[0]
                swapped=model.rollout_frames(frames[:,:3],source["action"],donor_anchor,13,a.steps,81000+pair)[0]
            else:
                states=predict_states(model,pixels[:,:3],source["action"],13)[0]
                own=model.render_states(states,own_anchor.expand(13,-1,-1,-1),steps=a.steps,seed=81000+pair)
                swapped=model.render_states(states,donor_anchor.expand(13,-1,-1,-1),steps=a.steps,seed=81000+pair)
            images=dict(own=own.float().clamp(-1,1),swapped=swapped.float().clamp(-1,1),real=frames[0,3:].float())
            estimated={key:pose(value).float() for key,value in images.items()}
        target=source["state"][0,3:]
        errors={}
        for key,value in estimated.items():
            errors[key+"_agent_px"]=(value[:,:2]*512-target[:,:2]).norm(dim=-1).cpu().tolist()
            errors[key+"_block_px"]=(value[:,2:4]*512-target[:,2:4]).norm(dim=-1).cpu().tolist()
            errors[key+"_angle_degrees"]=(torch.remainder(torch.atan2(value[:,4],value[:,5])-target[:,4]+torch.pi,2*torch.pi)-torch.pi).abs().mul(180/torch.pi).cpu().tolist()
        folder=out/f"{pair:02d}";folder.mkdir(exist_ok=True)
        Image.fromarray(rgb(donor_anchor)[0]).save(folder/"A_anchor.png")
        Image.fromarray(rgb(own_anchor)[0]).save(folder/"B_anchor.png")
        rows=[rgb(images["real"]),rgb(images["own"]),rgb(images["swapped"])]
        for name,frames_rgb in zip(["B_truth","B_own_anchor","B_states_A_anchor"],rows):video(folder/(name+".mp4"),frames_rgb)
        canvas=Image.new("RGB",(90+13*64,22+3*64),"white");draw=ImageDraw.Draw(canvas)
        for i,(label,sequence) in enumerate(zip(["Btruth","Bown","swapA"],rows)):
            draw.text((2,25+i*64),label,fill="black")
            for h,frame in enumerate(sequence):canvas.paste(Image.fromarray(frame),(90+h*64,22+i*64))
        canvas.save(folder/"grid.png")
        metadata=dict(pair=pair,appearance_A=appearance,dynamics_B=dynamics,appearance_anchor_row=donor_anchor_row,
                      dynamics_anchor_row=source_anchor_row,checkpoint=a.checkpoint,checkpoint_sha256=weights_hash,
                      model_config=cfg,shared_renderer_seed=81000+pair,steps=a.steps,errors=errors,
                      actions=source["action"][0].cpu().tolist(),primary_anchor_gap=model.anchor_gap,
                      scope="Matchedsamplernoise,truth/context/actionsofB;onlyanchorA/Bdiffers.AS-JEPAmeanstatesarefixedbetweenrenders.Nativepixelfeedbackcanchangeaftereachrender;realposefloorisreported.")
        (folder/"meta.json").write_text(json.dumps(metadata,indent=2));results.append(metadata)
    (out/"metrics.json").write_text(json.dumps(dict(applicable=True,pairs=len(results),steps=a.steps,results=results,scope="Qualitativeanchorswapdiagnostic;notafinalgenerationgate."),indent=2))
    print(json.dumps(dict(pairs=len(results),steps=a.steps)))


if __name__=="__main__":
    main()
