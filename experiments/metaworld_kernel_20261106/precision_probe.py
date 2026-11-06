"""Frozen prediction/ranking parity and timing for native world inference."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader,Subset


def run(a):
    from as_jepa_lewm.factory import build_model
    from as_jepa_lewm.path_projection import path_cost
    from as_jepa_tools.native_rgb_pairs import NativeRGBPairs
    from experiments.metaworld_eval_20261103.evaluate import resize_tensor,model_sha
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    cfg=json.loads((a.run/'config.json').read_text());model=build_model(cfg).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(a.run/f'model-step{a.step:07d}/weights.pt',map_location='cuda',weights_only=True));before=model_sha(model)
    data=NativeRGBPairs(cfg,'test');batch=next(iter(DataLoader(Subset(data,np.linspace(0,len(data)-1,128,dtype=int).tolist()),batch_size=128)))
    current=model.encoder(batch['current_pixels'].cuda());goal=model.encoder(batch['goal_pixels'].cuda())
    past=batch['previous_control'].cuda();valid=batch['history_valid'].cuda();role=valid.new_zeros(128)
    def fields(mixed):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mixed):return model.fields(current,goal,role,past,valid)
    with torch.inference_mode():
        reference=fields(False);mixed=fields(True)
        commands=torch.rand(128,4,device='cuda',generator=torch.Generator(device='cuda').manual_seed(313))*2-1
        def prediction(values):
            drift,effect,_,_=values
            return model.warp(current,drift+torch.einsum('na,nachw->nchw',torch.cat([commands,commands.abs()],-1),effect))
        exact_image=prediction(reference);mixed_image=prediction(mixed);difference=(exact_image-mixed_image).abs()
        timing={False:[],True:[]}
        for _ in range(3):
            for flag in [False,True]:
                torch.cuda.synchronize();start=time.perf_counter();fields(flag);torch.cuda.synchronize();timing[flag].append(time.perf_counter()-start)
        raw=np.load(next(r['path']for r in json.loads(a.manifest.read_text())['rows']if r['task']=='bin-picking-v3'and r['seed']==42))
        start=model.encoder(resize_tensor([raw['initial']]).cuda());target=model.encoder(resize_tensor([raw['goal']]).cuda())
        controls=torch.rand(32,3,4,device='cuda',generator=torch.Generator(device='cuda').manual_seed(314))*2-1
        scores=[]
        for flag in [False,True]:
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=flag):scores.append(path_cost(model,start,target,start.new_zeros(1,4),start.new_zeros(1),controls))
        rank=float(spearmanr(scores[0].cpu().numpy(),scores[1].cpu().numpy()).statistic)
        samebest=int(scores[0].argmin())==int(scores[1].argmin())
    assert model_sha(model)==before
    report=dict(passed=True,model_updates=0,model_unchanged=True,physical_gpu_shared_with_owned_frozen_eval=True,
        normalized_rgb_mse_difference=float(difference.square().mean()),normalized_rgb_max_difference=float(difference.max()),
        median_seconds_fp32=float(np.median(timing[False])),median_seconds_bf16=float(np.median(timing[True])),
        path_cost_rank_spearman=rank,same_best_self_generated_word=samebest,
        path_score_relative_max=float(((scores[0]-scores[1]).abs()/(1+scores[0].abs())).max()),
        scope='Only a numerical/timing probe.No inference implementation changed and no task-success claim.BF16wouldrequire its own physics/score provenance.')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['run','manifest','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--step',type=int,required=True);run(p.parse_args())
