"""Full quality/galleries on fixed held-out cases; media stays in EFS."""
import argparse,hashlib,importlib.util,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from as_jepa_tools.data import PushTWindows
from as_jepa_tools.evaluate import load_model,pose_net,select_windows,predict_states
from vforge.ops import initialize_ops,OpsConfig

def video(path,frames,fps=10):
    import av
    with av.open(str(path),"w") as container:
        stream=container.add_stream("libx264",rate=fps);stream.width=frames.shape[2];stream.height=frames.shape[1];stream.pix_fmt="yuv420p"
        for rgb in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format="rgb24")):container.mux(packet)
        for packet in stream.encode():container.mux(packet)

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--data",required=True);p.add_argument("--pose",required=True);p.add_argument("--output",required=True);p.add_argument("--i3d",required=True)
    p.add_argument("--cases",type=int,default=256);p.add_argument("--samples",type=int,default=4);p.add_argument("--steps",type=int,default=50);p.add_argument("--batch",type=int,default=8);p.add_argument("--fp32",action="store_true");p.add_argument("--media",action="store_true");a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/"metrics.json").exists():raise RuntimeError("Quality output already complete; use an independent directory")
    os.environ["TORCH_HOME"]=str(out.parents[2]/"cache/torch")
    torch.set_num_threads(2);torch.manual_seed(42);torch.set_float32_matmul_precision("highest" if a.fp32 else "high")
    initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda",0),parallel_state=None)
    model=load_model(a.checkpoint);pose=pose_net(a.pose)
    if model.mode=="L0":raise ValueError("L0 has no renderer")
    import lpips
    perceptual=lpips.LPIPS(net="alex",verbose=False).cuda().eval()
    from skimage.metrics import structural_similarity
    from PIL import Image,ImageDraw
    detector_path=Path(__file__).resolve().parents[1]/"jobs/fvd_reference/detector/fvd.py"
    spec=importlib.util.spec_from_file_location("project_i3d",detector_path);detector=importlib.util.module_from_spec(spec);spec.loader.exec_module(detector)
    extractor=detector.I3DExtractor(a.i3d,device="cuda:0",cache_dir=None,batch_size=4)
    # This detector fixes TF32 flags; world-model protocol is explicitly recorded below.
    data=PushTWindows(a.data,window=16,gap=8,split="test");data.open();ids=select_windows(data,a.cases)
    mean_records=[];all_truth=[];all_mean=[];arrays={};feature_real=[];feature_fake=[];feature_copy=[];started=time.time()
    for batch_id,batch in enumerate(DataLoader(Subset(data,ids),batch_size=a.batch,num_workers=2,pin_memory=True)):
        batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
        past=[int(data.offsets[int(ep)])+int(start)-(model.anchor_gap if model.mode not in {"S0","S1"} else 0) for ep,start in zip(batch["episode"].cpu(),batch["start"].cpu())]
        batch["anchor224"]=torch.from_numpy(np.array(data.p224[past])).permute(0,3,1,2).cuda();batch["anchor"]=torch.from_numpy(np.array(data.p64[past])).permute(0,3,1,2).cuda();batch["anchor_row"]=torch.tensor(past,device="cuda")
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=not a.fp32):
            pixels,frames,anchor=model.prepare(batch)
            if model.mode in {"S0","S1"}:generated=model.rollout_frames(frames[:,:3],batch["action"],frames[:,0],horizon=13,steps=a.steps,seed=1000+batch_id)
            else:
                predicted=predict_states(model,pixels[:,:3],batch["action"],13)
                flat=predicted.flatten(0,1);aa=anchor.repeat_interleave(13,0)
                generated=torch.cat([model.render_states(flat[lo:lo+64],aa[lo:lo+64],steps=a.steps,seed=1000+batch_id*100+lo) for lo in range(0,len(flat),64)]).reshape(len(frames),13,3,64,64)
            generated=generated.float().clamp(-1,1);truth=frames[:,3:].float();copy=frames[:,2:3].float().expand_as(generated)
            estimates=pose(generated.flatten(0,1)).float().reshape(len(frames),13,6)
            realpose=pose(truth.flatten(0,1)).float().reshape(len(frames),13,6)
            copypose=pose(copy.flatten(0,1)).float().reshape(len(frames),13,6)
        states=batch["state"][:,3:]
        for name,est in [("mean",estimates),("copy",copypose),("real",realpose)]:
            values=dict(agent=(est[:,:,:2]*512-states[:,:,:2]).norm(dim=-1),block=(est[:,:,2:4]*512-states[:,:,2:4]).norm(dim=-1))
            angle=torch.atan2(est[:,:,4],est[:,:,5]);values["angle_deg"]=torch.remainder(angle-states[:,:,4]+torch.pi,2*torch.pi).sub(torch.pi).abs()*180/torch.pi
            for key,value in values.items():arrays.setdefault(name+"_"+key,[]).append(value.cpu().numpy())
        mse=(generated-truth).square().mean((-3,-2,-1));arrays.setdefault("mean_psnr",[]).append((10*torch.log10(4/mse.clamp_min(1e-12))).cpu().numpy())
        lp=perceptual(generated.flatten(0,1),truth.flatten(0,1)).flatten().reshape(len(frames),13);arrays.setdefault("mean_lpips",[]).append(lp.cpu().numpy())
        gt=((frames+1)*127.5).clamp(0,255).byte().permute(0,1,3,4,2).cpu().numpy()
        pred=((generated+1)*127.5).byte().permute(0,1,3,4,2).cpu().numpy()
        copy_rgb=np.repeat(gt[:,2:3],13,axis=1)
        ssim=np.array([[structural_similarity(gt[i,h+3],pred[i,h],data_range=255,channel_axis=-1) for h in range(13)] for i in range(len(gt))])
        arrays.setdefault("mean_ssim",[]).append(ssim)
        clips=np.concatenate([gt[:,:3],pred],1);copyclips=np.concatenate([gt[:,:3],copy_rgb],1)
        feature_real.append(extractor._infer(list(gt)));feature_fake.append(extractor._infer(list(clips)));feature_copy.append(extractor._infer(list(copyclips)))
        for i,(ep,start) in enumerate(zip(batch["episode"].cpu(),batch["start"].cpu())):
            state=batch["state"][i].float();moving=float((state[:,2:4]-state[0,2:4]).norm(dim=-1).max())>10
            mean_records.append(dict(episode=int(ep),start=int(start),moving_block=moving,index=int(ids[len(mean_records)]),mean_renderer_seed=1000+batch_id*100))
        all_truth.append(gt);all_mean.append(pred)
        (out/"status.json").write_text(json.dumps(dict(state="mean_evaluating",completed=len(mean_records),total=len(ids),updated_at=time.time())))
    arrays={k:np.concatenate(v) for k,v in arrays.items()};truth=np.concatenate(all_truth);prediction=np.concatenate(all_mean);moving=np.array([r["moving_block"] for r in mean_records])
    np.save(out/"truth-rgb.npy",truth);np.save(out/"mean-rgb.npy",prediction)
    fvd=detector.frechet_distance(np.concatenate(feature_real),np.concatenate(feature_fake));copy_fvd=detector.frechet_distance(np.concatenate(feature_real),np.concatenate(feature_copy))
    sample_results=[]
    if model.mode in {"L2","S2"}:
        sampled=np.lib.format.open_memmap(out/"samples-rgb.npy",mode="w+",dtype=np.uint8,shape=(len(ids),a.samples,13,64,64,3))
        for case,index in enumerate(ids):
            batch=data[index];ep,start=data.locate(index);batch={k:v[None].cuda() if torch.is_tensor(v) else torch.tensor([v],device="cuda") for k,v in batch.items()}
            row=int(data.offsets[ep])+start-model.anchor_gap
            batch["anchor224"]=torch.from_numpy(np.array(data.p224[row])).permute(2,0,1)[None].cuda();batch["anchor"]=torch.from_numpy(np.array(data.p64[row])).permute(2,0,1)[None].cuda();batch["anchor_row"]=torch.tensor([row],device="cuda")
            generator=torch.Generator(device="cuda").manual_seed(20261027+int(index)*101)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=not a.fp32):
                pixels,frames,anchor=model.prepare(batch);mu=model.encode(dict(pixels=pixels[:,:3]))["emb"].float()
                noisy=model.context_samples(mu,a.samples,generator) if hasattr(model,"context_samples") else mu.expand(a.samples,-1,-1)+model.sigma0*torch.randn((a.samples,*mu.shape[1:]),device="cuda",generator=generator)
                history=list(noisy.unbind(1));action=model.action_encoder(batch["action"].expand(a.samples,-1,-1))
                for h in range(13):
                    lo=max(0,len(history)-(model.window-1));nxt=model.predict(torch.stack(history[lo:],1),action[:,lo:len(history)])[:,-1].float()
                    noise=model.transition_noise(nxt,generator) if hasattr(model,"transition_noise") else model.sigma0*torch.randn(nxt.shape,device="cuda",generator=generator)
                    history.append(nxt+noise)
                flat=torch.stack(history[3:],1).flatten(0,1);aa=anchor.expand(len(flat),-1,-1,-1)
                generated=torch.cat([model.render_states(flat[lo:lo+64],aa[lo:lo+64],steps=a.steps,seed=31000+int(index)*13+lo) for lo in range(0,len(flat),64)]).reshape(a.samples,13,3,64,64).float().clamp(-1,1)
                est=pose(generated.flatten(0,1)).float().reshape(a.samples,13,6)
            state=batch["state"][:,3:];angle=torch.remainder(torch.atan2(est[:,:,4],est[:,:,5])-state[:,:,4]+torch.pi,2*torch.pi).sub(torch.pi).abs()*180/torch.pi
            lp=perceptual(generated.flatten(0,1),frames[:,3:].expand(a.samples,-1,-1,-1,-1).flatten(0,1).float()).reshape(a.samples,13)
            sample_results.append(dict(agent=(est[:,:,:2]*512-state[:,:,:2]).norm(dim=-1).cpu().numpy(),block=(est[:,:,2:4]*512-state[:,:,2:4]).norm(dim=-1).cpu().numpy(),angle_deg=angle.cpu().numpy(),lpips=lp.cpu().numpy()))
            sampled[case]=((generated+1)*127.5).byte().permute(0,1,3,4,2).cpu().numpy()
            if case%8==0:(out/"status.json").write_text(json.dumps(dict(state="sample_evaluating",completed=case+1,total=len(ids),updated_at=time.time())))
        sampled.flush()
        for key in sample_results[0]:arrays["sample_"+key]=np.stack([r[key] for r in sample_results])
    if a.media:
        review=out/"gallery/fixed";review.mkdir(parents=True,exist_ok=True)
        for case in range(min(32,len(ids))):
            folder=review/f"{case:02d}";folder.mkdir(exist_ok=True)
            copy=np.repeat(truth[case,2:3],13,axis=0)
            rows=[truth[case,3:],prediction[case],copy]
            video(folder/"gt.mp4",truth[case,3:]);video(folder/"pred_mean.mp4",prediction[case]);video(folder/"copy.mp4",copy)
            if sample_results:
                for s in range(a.samples):video(folder/f"pred_sample_{s}.mp4",sampled[case,s]);rows.append(sampled[case,s])
            grid=Image.new("RGB",(13*64,len(rows)*64))
            for i,row in enumerate(rows):
                for h,frame in enumerate(row):grid.paste(Image.fromarray(frame),(h*64,i*64))
            grid.save(folder/"grid.png");(folder/"meta.json").write_text(json.dumps(mean_records[case],indent=2))
        (review/"review.csv").write_text("case,episode,start,pass_fail,reason\n"+"".join(f"{i:02d},{r['episode']},{r['start']},pending,\n" for i,r in enumerate(mean_records[:32])))
    means={key:np.asarray(value).mean(tuple(range(value.ndim-1))).tolist() for key,value in arrays.items()}
    result=dict(checkpoint=str(Path(a.checkpoint).resolve()),cases=len(ids),sampler_steps=a.steps,samples=a.samples if sample_results else 0,mean_by_horizon=means,
                h7_agent_ratio=float(arrays["mean_agent"][:,6].mean()/arrays["copy_agent"][:,6].mean()),h7_moving_block_ratio=float(arrays["mean_block"][moving,6].mean()/arrays["copy_block"][moving,6].mean()),
                fvd=fvd,copy_fvd=copy_fvd,detector=extractor.identity,
                fvd_protocol="256 matched16-frame clips:3 true context+13 generated,64->224 bilinear; this is not inheritedRGB16FVD2048",
                lpips="AlexNet LPIPS0.1.4; input[-1,1]",ssim="skimage channel_axis=-1,uint8,data_range255,defaultwin7",angle="wrapped absolute atan2(sin,cos) minus simulator angle,in degrees",
                seconds=time.time()-started,forward_precision="FP32" if a.fp32 else "BF16",tf32=False,actual_renderer_sampler=("finiteGaussianDiffusion"+str(model.density_steps)+"transitions+8bitGaussianEndpoint" if getattr(model,"method","")=="vdm_nats" else "Euler"+str(a.steps)),scope="Complete quality metrics; independent protocol directory; final32-case pass/fail review still required.")
    np.savez(out/"per-case-metrics.npz",**arrays);(out/"cases.json").write_text(json.dumps(mean_records,indent=2));(out/"metrics.json").write_text(json.dumps(result,indent=2))
    (out/"status.json").write_text(json.dumps(dict(state="finished",completed=len(ids),updated_at=time.time())))
    print(json.dumps(dict(cases=len(ids),fvd=fvd["fvd"],h7_agent_ratio=result["h7_agent_ratio"],seconds=result["seconds"]),indent=2))
if __name__=="__main__":main()
