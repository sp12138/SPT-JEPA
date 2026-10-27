"""Shared held-out rollout, intervention and probe evidence for the campaign."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from vforge.ops import OpsConfig,initialize_ops
from as_jepa_tools.data import PushTWindows
from as_jepa_lewm.model import ASJepaLeWM


def load_model(folder,device="cuda"):
    folder=Path(folder);cfg=json.loads((folder/"config.json").read_text());cfg.pop("_target_",None)
    target=json.loads((folder/"config.json").read_text()).get("_target_", "as_jepa_lewm.model.ASJepaLeWM")
    if target=="vforge.models.as_jepa.model.ASJepaSPT":
        from vforge.models.as_jepa.model import ASJepaSPT
        model=ASJepaSPT(**cfg).to(device)
    elif target=="vforge.models.as_jepa.spt_baseline.ActionRepWorld":
        from vforge.models.as_jepa.spt_baseline import ActionRepWorld
        model=ActionRepWorld(**cfg).to(device)
    else:model=ASJepaLeWM(**cfg).to(device)
    model.load_state_dict(torch.load(folder/"weights.pt",map_location=device,weights_only=True),strict=True);return model.eval()

@torch.no_grad()
def predict_states(model,context,actions,horizon,sample_noise=False):
    emb=model.encode(dict(pixels=context))["emb"]
    if sample_noise and model.mode in {"L2","S2"}:emb=emb+model.sigma0*torch.randn_like(emb)
    history=list(emb.unbind(1));h=len(history);act=model.action_encoder(actions)
    for i in range(horizon):
        lo=max(0,len(history)-(model.window-1));past=torch.stack(history[lo:],1)
        nxt=model.predict(past,act[:,lo:len(history)])[:,-1]
        if sample_noise and model.mode in {"L2","S2"}:nxt=nxt+model.sigma0*torch.randn_like(nxt)
        history.append(nxt)
    return torch.stack(history[h:],1)


def select_windows(dataset,count,seed=0):
    dataset.open();rng=np.random.default_rng(seed)
    candidates=rng.permutation(len(dataset))[:min(10000,len(dataset))]
    moving=[];static=[]
    for index in candidates:
        ep,start=dataset.locate(index)
        if start<8:continue
        rows=np.arange(int(dataset.offsets[ep])+start,int(dataset.offsets[ep])+start+dataset.window)
        states=np.array(dataset.states[rows]);motion=np.linalg.norm(states[:,2:4]-states[0,2:4],axis=1).max()>10
        (moving if motion else static).append(int(index))
        if len(moving)>=count//2 and len(static)>=count-count//2:break
    selected=moving[:count//2]+static[:count-count//2]
    if len(selected)<count:
        selected=(moving+static)[:count]
    return selected


def pose_net(path):
    from torchvision.models import resnet18
    model=resnet18(weights=None,num_classes=6);model.conv1=nn.Conv2d(3,64,3,1,1,bias=False);model.maxpool=nn.Identity()
    path=Path(path)
    if path.suffix==".ckpt":
        weights=torch.load(path,map_location="cpu",weights_only=False)["state_dict"];weights={k.removeprefix("net."):v for k,v in weights.items()}
    else:weights=torch.load(path,map_location="cpu",weights_only=True)
    model.load_state_dict(weights);return model.cuda().eval()

@torch.no_grad()
def evaluate(model,root,output,count=64,steps=20,pose_path=None,batch_size=8,fp32=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_float32_matmul_precision("highest" if fp32 or getattr(model,"fp32_pool",False) else "high")
    if model.mode in {"S0","S1"}:return evaluate_pixel(model,root,output,count,steps,pose_path,batch_size,fp32=fp32)
    data=PushTWindows(root,window=16,gap=8,split="test");indices=select_windows(data,count)
    loader=DataLoader(Subset(data,indices),batch_size=batch_size,num_workers=2,pin_memory=True)
    pose=pose_net(pose_path) if pose_path else None
    errors={};latent_errors=[];shuffled_errors=[];feature_stds=[];records=[];started=time.time()
    image_rows=[];anchor_changes=[];zero_changes=[];permutation_changes=[];all_means=[]
    for batch_id,batch in enumerate(loader):
        batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=not fp32):
            pixels,frames,anchor=model.prepare(batch)
            # Primary deployment anchor is eight kept frames before the context.
            # Dataset anchor selection is replaced by this deterministic past frame.
            data.open();pastrows=[int(data.offsets[int(ep)])+int(start)-model.anchor_gap for ep,start in zip(batch["episode"].cpu(),batch["start"].cpu())]
            a224=torch.from_numpy(np.array(data.p224[pastrows])).permute(0,3,1,2).cuda()
            a64=torch.from_numpy(np.array(data.p64[pastrows])).permute(0,3,1,2).cuda()
            prior_batch={**batch,"anchor224":a224,"anchor":a64,"anchor_row":torch.tensor(pastrows,device="cuda")}
            _,_,anchor=model.prepare(prior_batch)
            if model.anchor_gap==0:anchor=frames[:,0]
            mu=model.encode(dict(pixels=pixels))["emb"]
            all_means.append(mu.float().flatten(0,1).cpu())
            predicted=predict_states(model,pixels[:,:3],batch["action"],13)
            permutation=torch.arange(len(pixels)-1,-1,-1,device="cuda")
            shuffled=predict_states(model,pixels[:,:3],batch["action"][permutation],13)
            latent_errors.append((predicted.float()-mu[:,3:].float()).square().mean(-1).cpu())
            shuffled_errors.append((shuffled.float()-mu[:,3:].float()).square().mean(-1).cpu())
            feature_stds.append(mu.float().std((0,1)).mean().cpu())
            if model.renderer is not None:
                state=predicted.flatten(0,1);aa=anchor.repeat_interleave(13,dim=0)
                rendered=[]
                for lo in range(0,len(state),64):rendered.append(model.render_states(state[lo:lo+64],aa[lo:lo+64],steps=steps,seed=1000+batch_id*100+lo))
                generated=torch.cat(rendered).reshape(len(pixels),13,3,64,64).float().clamp(-1,1)
                mse=(generated-frames[:,3:]).square().mean((-3,-2,-1)).cpu()
                copy=frames[:,2:3].expand_as(generated)
                errors.setdefault("pixel_mse",[]).append(mse)
                errors.setdefault("copy_pixel_mse",[]).append((copy-frames[:,3:]).square().mean((-3,-2,-1)).cpu())
                if pose is not None:
                    predpose=pose(generated.flatten(0,1)).float().reshape(len(pixels),13,6)
                    copypose=pose(copy.flatten(0,1)).float().reshape(len(pixels),13,6)
                    realpose=pose(frames[:,3:].flatten(0,1)).float().reshape(len(pixels),13,6)
                    true=batch["state"][:,3:,:4].float()/512
                    for name,values in [("agent",(predpose[:,:,:2]-true[:,:,:2]).norm(dim=-1)*512),("block",(predpose[:,:,2:4]-true[:,:,2:4]).norm(dim=-1)*512),("copy_agent",(copypose[:,:,:2]-true[:,:,:2]).norm(dim=-1)*512),("copy_block",(copypose[:,:,2:4]-true[:,:,2:4]).norm(dim=-1)*512),("real_agent_floor",(realpose[:,:,:2]-true[:,:,:2]).norm(dim=-1)*512),("real_block_floor",(realpose[:,:,2:4]-true[:,:,2:4]).norm(dim=-1)*512)]:errors.setdefault(name,[]).append(values.cpu())
                # Matched pure-noise seed makes these changes reflect conditioning interventions.
                own=model.render_states(mu[:,3],anchor,steps=steps,seed=77)
                zero=model.render_states(mu[:,3]*0,anchor,steps=steps,seed=77)
                noisy_own=None
                if model.mode in {"L2","S2"}:
                    noise_generator=torch.Generator(device=mu.device).manual_seed(1700+batch_id)
                    noisy_mu=mu[:,3].float()+model.sigma0*torch.randn(mu[:,3].shape,device=mu.device,generator=noise_generator)
                    noisy_own=model.render_states(noisy_mu,anchor,steps=steps,seed=77)
                swap=model.render_states(mu[:,3],anchor.flip(0),steps=steps,seed=77)
                permuted=model.render_states(mu[:,3].flip(0),anchor,steps=steps,seed=77)
                permutation_changes.append((own.float()-permuted.float()).square().mean((1,2,3)).cpu())
                zero_changes.append((own.float()-zero.float()).square().mean((1,2,3)).cpu());anchor_changes.append((own.float()-swap.float()).square().mean((1,2,3)).cpu())
                if pose is not None:
                    ownpose=pose(own.clamp(-1,1)).float();zeropose=pose(zero.clamp(-1,1)).float();anchorpose=pose(anchor).float();target=batch["state"][:,3,:2].float()/512
                    anchortruth=torch.tensor(np.array(data.states[pastrows,:2]),device=target.device)
                    if model.anchor_gap==0:anchortruth=batch["state"][:,0,:2].float()
                    moving=(target*512-anchortruth).norm(dim=-1)>10
                    for name,values in [("posterior_agent",(ownpose[:,:2]-target).norm(dim=-1)*512),("zero_agent",(zeropose[:,:2]-target).norm(dim=-1)*512),("anchor_agent",(anchorpose[:,:2]-target).norm(dim=-1)*512)]:errors.setdefault(name,[]).append(values[moving].cpu())
                    if noisy_own is not None:
                        noisy_pose=pose(noisy_own.clamp(-1,1)).float()
                        errors.setdefault("noisy_posterior_agent",[]).append(((noisy_pose[:,:2]-target).norm(dim=-1)*512)[moving].cpu())
                        errors.setdefault("posterior_noise_image_change",[]).append((noisy_own.float()-own.float()).square().mean((1,2,3)).cpu())
                if len(image_rows)<16:
                    image_rows.extend([(frames[i].cpu(),generated[i].cpu(),anchor[i].cpu()) for i in range(min(len(pixels),16-len(image_rows)))])
        for ep,start in zip(batch["episode"].cpu(),batch["start"].cpu()):records.append(dict(episode=int(ep),start=int(start),moving_block=bool(np.linalg.norm(np.array(data.states[int(data.offsets[int(ep)])+int(start):int(data.offsets[int(ep)])+int(start)+16,2:4])-np.array(data.states[int(data.offsets[int(ep)])+int(start),2:4]),axis=1).max()>10)))
    result=dict(float32_matmul_precision=torch.get_float32_matmul_precision(),feature_forward_precision="FP32" if fp32 else "BF16 autocast",pose_weights=str(pose_path),cases=len(records),sampler_steps=steps,seconds=time.time()-started,feature_std=float(torch.stack(feature_stds).mean()),latent_mse_by_horizon=torch.cat(latent_errors).mean(0).tolist(),shuffled_action_latent_mse_by_horizon=torch.cat(shuffled_errors).mean(0).tolist(),scope="single-seed exploratory fixed held-out windows; not a three-seed benchmark")
    means=torch.cat(all_means);variance=means.var(0,unbiased=False)
    result["state_std_per_dimension"]=variance.sqrt().tolist()
    if model.mode in {"L2","S2"}:
        result["noisy_state_information_upper_bound_nats"]=float(.5*torch.log1p(variance/model.sigma0**2).sum())
        result["information_bound_scope"]="Empirical diagonal-covariance Gaussian-noise channel upper bound; not an estimate of task-relevant information."
    arrays={k:torch.cat(v).numpy() for k,v in errors.items()}
    for name,values in arrays.items():result[name+"_by_horizon"]=values.mean(0).tolist() if values.ndim>1 else float(values.mean())
    if zero_changes:result["zero_state_intervention_mse"]=float(torch.cat(zero_changes).mean());result["anchor_swap_intervention_mse"]=float(torch.cat(anchor_changes).mean());result["state_permutation_intervention_mse"]=float(torch.cat(permutation_changes).mean())
    if "agent" in arrays:
        for h in [1,3,7,13]:
            result[f"h{h}_agent_ratio"]=float(arrays["agent"][:,h-1].mean()/max(1e-6,arrays["copy_agent"][:,h-1].mean()))
            result[f"h{h}_block_ratio"]=float(arrays["block"][:,h-1].mean()/max(1e-6,arrays["copy_block"][:,h-1].mean()))
    moving=np.asarray([row["moving_block"] for row in records])
    result["moving_block_cases"]=int(moving.sum())
    if "block" in arrays and moving.any():result["h7_moving_block_ratio"]=float(arrays["block"][moving,6].mean()/max(1e-6,arrays["copy_block"][moving,6].mean()))
    if "pixel_mse" in arrays:result["psnr_by_horizon"]=(10*np.log10(4/np.maximum(arrays["pixel_mse"],1e-12))).mean(0).tolist()
    (output/"metrics.json").write_text(json.dumps(result,indent=2));(output/"cases.json").write_text(json.dumps(records,indent=2));np.savez(output/"per_case.npz",**arrays)
    if image_rows:gallery(image_rows,output)
    return result


def gallery(rows,output):
    from PIL import Image,ImageDraw
    indices=[3,5,9,15];tile=96;canvas=Image.new("RGB",(tile*len(indices)*2,tile*len(rows)),"white")
    for i,(truth,pred,anchor) in enumerate(rows):
        for j,t in enumerate(indices):
            for k,frame in enumerate([truth[t],pred[t-3]]):
                image=((frame.permute(1,2,0).float().clamp(-1,1)+1)*127.5).byte().numpy();canvas.paste(Image.fromarray(image).resize((tile,tile)),((j*2+k)*tile,i*tile))
    canvas.save(output/"gallery.png")


@torch.no_grad()
def evaluate_pixel(model,root,output,count,steps,pose_path,batch_size,fp32=False):
    output=Path(output);data=PushTWindows(root,window=16,gap=8,split="test");indices=select_windows(data,count)
    loader=DataLoader(Subset(data,indices),batch_size=min(4,batch_size),num_workers=2,pin_memory=True)
    pose=pose_net(pose_path) if pose_path else None;errors={};images=[];started=time.time()
    for batch_id,batch in enumerate(loader):
        batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=not fp32):
            _,frames,_=model.prepare(batch);anchor=frames[:,0]
            generated=model.rollout_frames(frames[:,:3],batch["action"],anchor,horizon=13,steps=steps,seed=1000+batch_id).float().clamp(-1,1)
            copy=frames[:,2:3].expand_as(generated)
            errors.setdefault("pixel_mse",[]).append((generated-frames[:,3:]).square().mean((-3,-2,-1)).cpu())
            errors.setdefault("copy_pixel_mse",[]).append((copy-frames[:,3:]).square().mean((-3,-2,-1)).cpu())
            if pose is not None:
                prediction=pose(generated.flatten(0,1)).float().reshape(len(frames),13,6);copypose=pose(copy.flatten(0,1)).float().reshape(len(frames),13,6);truth=batch["state"][:,3:,:4].float()/512
                for name,values in [("agent",(prediction[:,:,:2]-truth[:,:,:2]).norm(dim=-1)*512),("block",(prediction[:,:,2:4]-truth[:,:,2:4]).norm(dim=-1)*512),("copy_agent",(copypose[:,:,:2]-truth[:,:,:2]).norm(dim=-1)*512),("copy_block",(copypose[:,:,2:4]-truth[:,:,2:4]).norm(dim=-1)*512)]:errors.setdefault(name,[]).append(values.cpu())
            if len(images)<16:images.extend([(frames[i].cpu(),generated[i].cpu(),anchor[i].cpu()) for i in range(min(len(frames),16-len(images)))])
    arrays={k:torch.cat(v).numpy() for k,v in errors.items()};result=dict(float32_matmul_precision=torch.get_float32_matmul_precision(),feature_forward_precision="FP32" if fp32 else "BF16 autocast",pose_weights=str(pose_path),cases=len(indices),sampler_steps=steps,seconds=time.time()-started,model=model.mode,scope="single-seed SPT pixel-autoregressive baseline on the same held-out window IDs")
    for key,value in arrays.items():result[key+"_by_horizon"]=value.mean(0).tolist()
    if "agent" in arrays:
        for h in [1,3,7,13]:
            result[f"h{h}_agent_ratio"]=float(arrays["agent"][:,h-1].mean()/max(1e-6,arrays["copy_agent"][:,h-1].mean()))
            result[f"h{h}_block_ratio"]=float(arrays["block"][:,h-1].mean()/max(1e-6,arrays["copy_block"][:,h-1].mean()))
    output.mkdir(parents=True,exist_ok=True);(output/"metrics.json").write_text(json.dumps(result,indent=2));(output/"cases.json").write_text(json.dumps([dict(episode=int(data.locate(i)[0]),start=int(data.locate(i)[1])) for i in indices],indent=2));np.savez(output/"per_case.npz",**arrays);gallery(images,output);return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--data",required=True);p.add_argument("--output",required=True);p.add_argument("--cases",type=int,default=64);p.add_argument("--steps",type=int,default=20);p.add_argument("--pose");p.add_argument("--fp32",action="store_true");a=p.parse_args()
    torch.set_num_threads(2);torch.set_float32_matmul_precision("high");initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda",0),parallel_state=None)
    model=load_model(a.checkpoint);result=evaluate(model,a.data,a.output,a.cases,a.steps,a.pose,fp32=a.fp32);print(json.dumps(result,indent=2))

if __name__=="__main__":main()
