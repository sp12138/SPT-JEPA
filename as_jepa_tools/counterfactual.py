"""Counterfactual action following under matched simulator resets and common sampler noise."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from as_jepa_tools.evaluate import load_model,predict_states,pose_net
from as_jepa_tools.data import PushTWindows
from vforge.ops import initialize_ops,OpsConfig


def costs(prediction,truth):
    # (candidate prediction, candidate simulator, horizon, entity, xy)
    squared=(prediction[:,None]-truth[None]).reshape(4,4,-1,2,2)**2
    return squared.sum(-1).mean((2,3))


def gallery(cases,output):
    from PIL import Image,ImageDraw
    tile=96;times=[0,3,7];canvas=Image.new("RGB",(7*tile,len(cases)*8*tile),"white")
    draw=ImageDraw.Draw(canvas)
    for case,record in enumerate(cases):
        for action in range(4):
            y=(case*8+action*2)*tile
            draw.text((2,y+2),f"case {record['case']} action {action}",fill="black")
            for j,t in enumerate(times):
                for row,frames in enumerate([record["truth"],record["prediction"]]):
                    im=Image.fromarray(frames[action,t]).resize((tile,tile))
                    canvas.paste(im,((j+1)*2*tile,y+row*tile))
    canvas.save(output/"gallery.png")


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--data",required=True)
    p.add_argument("--pose",required=True);p.add_argument("--output",required=True);p.add_argument("--cases",type=int,default=32)
    p.add_argument("--steps",type=int,default=50);p.add_argument("--fp32",action="store_true");a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42)
    initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda",0),parallel_state=None)
    from stable_worldmodel.envs.pusht import PushT as PushTEnv
    from PIL import Image
    data=PushTWindows(a.data,window=16,gap=8,split="test");data.open()
    model=load_model(a.checkpoint);pose=pose_net(a.pose);rng=np.random.default_rng(42)
    if model.domain!="standard":raise ValueError("Counterfactual simulator protocol currently covers the standard domain")
    if model.mode=="L0":raise ValueError("L0 has no pixel renderer; evaluate it with P0 planning")
    candidate=rng.choice(len(data),min(5000,len(data)),replace=False)
    eligible=[int(i) for i in candidate if data.locate(i)[1]>=8][:a.cases]
    env=PushTEnv(render_mode="rgb_array")
    root=Path(a.data);raw=np.load(root/"action_raw.npy",mmap_mode="r");raw_rows=np.load(root/"raw_rows.npy")
    state=np.load(root/"states.npy",mmap_mode="r");card=json.loads((root/"dataset_card.json").read_text())
    mean=np.asarray(card["action_mean"]);std=np.asarray(card["action_std"])
    records=[];truth_all=[];prediction_all=[];pose_truth_all=[];galleries=[];started=time.time()
    for case,index in enumerate(eligible):
        batch=data[index];ep,start=data.locate(index);rows=batch["rows"].numpy()
        origin=int(rows[2]);initial=np.array(state[origin]);origin_raw=int(raw_rows[origin])
        donors=[origin_raw]+[int(raw_rows[int(data[int(rng.integers(len(data)))]["rows"][2])]) for _ in range(3)]
        truth=[];normalized=[];truth_images=[]
        for donor in donors:
            sequence=np.nan_to_num(np.array(raw[donor:donor+40]),nan=0.)
            if len(sequence)!=40:raise ValueError("Incomplete donor control sequence")
            env.reset(seed=case);env._set_state(initial.copy());trajectory=[];images=[]
            for t,action in enumerate(sequence):
                env.step(action)
                if (t+1)%5==0:
                    trajectory.append(np.r_[np.array(env.agent.position),np.array(env.block.position),float(env.block.angle)])
                    images.append(np.asarray(Image.fromarray(env.render()).resize((64,64),Image.Resampling.BOX)))
            truth.append(np.asarray(trajectory));truth_images.append(np.stack(images))
            normalized.append(((sequence-mean)/std).reshape(8,10))
        truth=np.asarray(truth);truth_images=np.asarray(truth_images)
        pixels=batch["pixels"][:3][None].cuda();frames=batch["render"][:3][None].cuda().float()/127.5-1.
        anchor_row=int(data.offsets[ep])+start-model.anchor_gap if model.mode not in {"S0","S1"} else int(rows[0])
        anchor=torch.from_numpy(np.array(data.p64[anchor_row])).permute(2,0,1)[None].cuda().float()/127.5-1.
        past=batch["action"][:2].numpy();actions=torch.from_numpy(np.stack([np.concatenate([past,v],0) for v in normalized])).cuda().float()
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=not a.fp32):
            if model.mode in {"S0","S1"}:
                generated=torch.cat([model.rollout_frames(frames,actions[i:i+1],anchor,horizon=8,steps=a.steps,seed=123+case) for i in range(4)])
            else:
                context=model.normalize(pixels).expand(4,-1,-1,-1,-1) if model.mode.startswith("L") else frames.expand(4,-1,-1,-1,-1)
                states=predict_states(model,context,actions,8)
                # Equal noise at each horizon for all four candidate action sequences.
                generated=torch.stack([model.render_states(states[i],anchor.expand(8,-1,-1,-1),steps=a.steps,seed=123+case) for i in range(4)])
            estimate=pose(generated.flatten(0,1).clamp(-1,1)).float().reshape(4,8,6)
            positions=(estimate[:,:,:4]*512).cpu().numpy()
            real=torch.from_numpy(truth_images.copy()).permute(0,1,4,2,3).cuda().float()/127.5-1.
            real_pose=pose(real.flatten(0,1)).float().reshape(4,8,6)[:,:,:4]*512
        target=truth[:,:,:4];matrix=costs(positions,target);assignment=matrix.argmin(1)
        oracle_matrix=costs(real_pose.cpu().numpy(),target)
        replay_error=np.linalg.norm(target[0].reshape(8,2,2)-np.array(state[rows[3:11],:4]).reshape(8,2,2),axis=-1).mean(0)
        entity_matrix=((positions[:,None]-target[None]).reshape(4,4,8,2,2)**2).sum(-1).mean(2)
        rec=dict(case=case,episode=int(ep),start=int(start),source_raw_row=origin_raw,donor_raw_rows=donors,
                 assigned=assignment.tolist(),correct=(assignment==np.arange(4)).tolist(),
                 agent_correct=(entity_matrix[:,:,0].argmin(1)==np.arange(4)).tolist(),
                 block_correct=(entity_matrix[:,:,1].argmin(1)==np.arange(4)).tolist(),
                 oracle_pose_correct=(oracle_matrix.argmin(1)==np.arange(4)).tolist(),
                 cost_matrix=matrix.tolist(),recorded_action_replay_agent_error=float(replay_error[0]),
                 recorded_action_replay_block_error=float(replay_error[1]))
        records.append(rec);truth_all.append(target);prediction_all.append(positions);pose_truth_all.append(real_pose.cpu().numpy())
        if len(galleries)<4:
            pred=((generated.float().clamp(-1,1)+1)*127.5).byte().permute(0,1,3,4,2).cpu().numpy()
            galleries.append(dict(case=case,truth=truth_images,prediction=pred))
    env.close();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    def average(key):return float(np.mean([r[key] for r in records]))
    report=dict(feature_forward_precision="FP32" if a.fp32 else "BF16 autocast",checkpoint=str(Path(a.checkpoint).resolve()),cases=len(records),rollouts=4*len(records),
                action_identification_accuracy=average("correct"),agent_action_identification_accuracy=average("agent_correct"),
                block_action_identification_accuracy=average("block_correct"),oracle_pose_action_identification_accuracy=average("oracle_pose_correct"),
                chance=.25,sampler_steps=a.steps,seconds=time.time()-started,common_sampler_noise=True,
                recorded_action_replay_agent_error=average("recorded_action_replay_agent_error"),
                recorded_action_replay_block_error=average("recorded_action_replay_block_error"),
                scope="Single-seed matched-reset simulator counterfactuals; deterministic mean-state dynamics; reset cannot recover unsaved block velocity",
                details=records)
    (out/"counterfactual.json").write_text(json.dumps(report,indent=2))
    np.savez(out/"trajectories.npz",truth=np.asarray(truth_all),prediction=np.asarray(prediction_all),pose_on_simulator=np.asarray(pose_truth_all))
    gallery(galleries,out);print(json.dumps({k:v for k,v in report.items() if k!="details"},indent=2))


if __name__=="__main__":main()
