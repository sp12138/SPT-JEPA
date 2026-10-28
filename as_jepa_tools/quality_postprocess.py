"""Add copy baselines and all-sample quality metrics from already frozen RGB banks."""
import argparse,json,importlib.util,hashlib,os
from pathlib import Path
import numpy as np
import torch
def main():
 p=argparse.ArgumentParser();p.add_argument("--quality",required=True);p.add_argument("--output",required=True);p.add_argument("--i3d",required=True);a=p.parse_args()
 root=Path(a.quality);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 os.environ["TORCH_HOME"]=str(root.parents[3]/"cache/torch");torch.set_num_threads(2)
 import lpips
 from skimage.metrics import structural_similarity
 metric=lpips.LPIPS(net="alex",verbose=False).cuda().eval()
 source=Path(__file__).resolve().parents[1]/"jobs/fvd_reference/detector/fvd.py";spec=importlib.util.spec_from_file_location("i3d_owned",source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 extractor=module.I3DExtractor(a.i3d,device="cuda:0",cache_dir=None,batch_size=4)
 truth=np.load(root/"truth-rgb.npy",mmap_mode="r");samples=np.load(root/"samples-rgb.npy",mmap_mode="r") if (root/"samples-rgb.npy").exists() else None
 all_metrics={};real_features=[];sample_features=[[] for _ in range(samples.shape[1])] if samples is not None else []
 with torch.no_grad():
  for start in range(0,len(truth),4):
   gt=np.array(truth[start:start+4]);target=gt[:,3:];copy=np.repeat(gt[:,2:3],13,axis=1)
   real_features.append(extractor._infer(list(gt)))
   def evaluate(rgb,label):
    x=torch.from_numpy(rgb.copy()).permute(0,1,4,2,3).flatten(0,1).cuda().float()/127.5-1
    y=torch.from_numpy(target.copy()).permute(0,1,4,2,3).flatten(0,1).cuda().float()/127.5-1
    lp=metric(x,y).reshape(len(rgb),13).cpu().numpy()
    ssim=np.array([[structural_similarity(target[i,h],rgb[i,h],data_range=255,channel_axis=-1) for h in range(13)] for i in range(len(rgb))])
    mse=((rgb.astype(float)-target.astype(float))**2).mean((-3,-2,-1))
    all_metrics.setdefault(label+"_lpips",[]).append(lp);all_metrics.setdefault(label+"_ssim",[]).append(ssim);all_metrics.setdefault(label+"_psnr",[]).append(10*np.log10(255**2/np.maximum(mse,1e-12)))
   evaluate(copy,"copy")
   evaluate(np.array(np.load(root/"mean-rgb.npy",mmap_mode="r")[start:start+4]),"mean_uint8")
   if samples is not None:
    for sample in range(samples.shape[1]):
     rgb=np.array(samples[start:start+4,sample]);evaluate(rgb,f"sample{sample}")
     sample_features[sample].append(extractor._infer(list(np.concatenate([gt[:,:3],rgb],axis=1))))
 arrays={k:np.concatenate(v) for k,v in all_metrics.items()}
 real=np.concatenate(real_features);scores=[module.frechet_distance(real,np.concatenate(parts)) for parts in sample_features]
 np.savez(out/"per-case-quality.npz",**arrays)
 result=dict(quality_source=str(root),metrics_sha256=hashlib.sha256((root/"metrics.json").read_bytes()).hexdigest(),cases=len(truth),mean_by_horizon={k:v.mean(0).tolist() for k,v in arrays.items()},sample_fvd=scores,scope="Mean/copy/sample perceptual comparisons here all use the same saved uint8 pixels. Each sampled-trajectory FVD uses256 real and256 fake clips; four draws reported separately,not pooled as1024 independent real cases. Copy and sample perceptual metrics use the saveduint8bank.")
 (out/"metrics.json").write_text(json.dumps(result,indent=2));print(json.dumps(dict(cases=len(truth),sample_fvd=[x["fvd"] for x in scores])))
if __name__=="__main__":main()
