"""Frozen source-validation diagnostics; physical state never enters a planner.

Observed next frames test factual prediction. Zero-control forecasts are not
simulated counterfactuals and cannot establish causal action ranking.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from experiments.robocasa_compare.acquire import save
from experiments.robocasa_compare.data import Windows, STRIDE
from experiments.robocasa_compare.train import build, state_hash


def binding(dataset, index):
    rng = np.random.default_rng(np.random.SeedSequence([dataset.seed, index]))
    row = int(rng.integers(dataset.n))
    episode = int(np.searchsorted(dataset.ends, row, side="right"))
    record = dataset.records[episode]
    start = row - (int(dataset.ends[episode-1]) if episode else 0)
    return record, start


def geometry(features, poses, slots, width):
    # [B,V,T,F], with native source EE positions used for diagnostics only.
    b, views, t, _ = features.shape
    goal = features[:, :, -1:]
    difference = goal - features[:, :, :1]
    error = features-goal
    uniform = error.abs().mean((1, 3))
    slot_error = error.reshape(b, views, t, slots, width).abs().mean(-1)
    weights = difference.reshape(b, views, 1, slots, width).square().mean(-1)
    weights = torch.where(weights.sum(-1, keepdim=True)>0, weights, torch.ones_like(weights))
    weighted = ((slot_error*weights).sum(-1)/weights.sum(-1)).mean(1)
    projected = ((error*difference).sum(-1).abs()/difference.square().sum(-1).clamp_min(1e-20)).mean(1)
    distance = np.linalg.norm(poses-poses[:, -1:], axis=-1)
    results = {}
    for name, value in (("uniform",uniform),("goal_slot_weighted",weighted),("goal_direction",projected)):
        correlations = []
        for predicted, actual in zip(value.cpu().numpy(), distance):
            # Exclude the forced zero at the endpoint in both spaces.
            x, y = predicted[:-1], actual[:-1]
            if np.ptp(x)>1e-8 and np.ptp(y)>1e-5:
                correlations.append(float(spearmanr(x,y).statistic))
        results[name] = correlations
    return results


def run(args):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_device(0)
    config = json.loads((args.run/"config.json").read_text())
    model = build(config["model"], args.root/"code").cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(args.run/f"step-{args.step:06d}/weights.pt",map_location="cuda",weights_only=True))
    before = state_hash(model.state_dict())
    from vforge.ops import OpsConfig, initialize_ops
    initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda"),parallel_state=None)
    data = Windows(args.data, 8, 20261110, "validation")
    # Every model sees exactly the same 128 windows, irrespective of model seed.
    indices = list(range(128))
    loader = DataLoader(data,batch_sampler=[indices[i:i+8] for i in range(0,len(indices),8)],num_workers=0)
    zero = torch.as_tensor(-data.mean/data.std,device="cuda")[None,None]
    records, all_features, all_poses = [], [], []
    factual, stationary, copy = [], [], []
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
        for batch_index, batch in enumerate(loader):
            b,v,t = batch["pixels"].shape[:3]
            pixels = model.normalize(batch["pixels"].flatten(0,1).cuda())
            actions = batch["action"][:,None].expand(b,v,t,14).reshape(b*v,t,14).cuda()
            encoded = model.encode(dict(pixels=pixels,action=actions))
            e = encoded["emb"]
            actual = model.predict(e[:,:1],encoded["act_emb"][:,:1])[:,0].float()
            null = model.predict(e[:,:1],model.action_encoder(zero.expand(b*v,1,14)))[:,0].float()
            target = e[:,1].float()
            factual.extend((actual-target).square().mean(-1).reshape(b,v).mean(-1).cpu().tolist())
            stationary.extend((null-target).square().mean(-1).reshape(b,v).mean(-1).cpu().tolist())
            copy.extend((e[:,0].float()-target).square().mean(-1).reshape(b,v).mean(-1).cpu().tolist())
            all_features.append(e.float().reshape(b,v,t,-1).cpu())
            poses = []
            for index in indices[batch_index*8:batch_index*8+b]:
                record,start = binding(data,index)
                frame_ids = record["offset"]+start+np.arange(t)*STRIDE
                poses.append(np.asarray(data.shards[record["shard"]][1][frame_ids,7:10]).copy())
                records.append(dict(shard=record["shard"],episode=record["episode"],start=start,
                                    kitchen=record["kitchen"],split=record["split"]))
            all_poses.append(np.stack(poses))
    features = torch.cat(all_features)
    poses = np.concatenate(all_poses)
    comparisons = geometry(features,poses,model.state_slots,model.state_width)
    temporal_rms = float((features[:,:,1:]-features[:,:,:-1]).square().mean().sqrt())
    posterior_difference_std = model.sigma0*np.sqrt(2*(1-model.rho))
    f,z,c = map(np.asarray,(factual,stationary,copy))
    manifest_hash = hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    unchanged = before == state_hash(model.state_dict())
    result = dict(passed=unchanged,run=args.run.name,step=args.step,windows=len(records),
         unique_source_episodes=len({(q["shard"],q["episode"]) for q in records}),
         validation_kitchens=len({tuple(q["kitchen"]) for q in records}),manifest_sha256=manifest_hash,
         model_unchanged=unchanged,model_sha256=before,
         factual_mse=float(f.mean()),zero_control_forecast_mse=float(z.mean()),copy_mse=float(c.mean()),
         factual_better_than_zero_fraction=float(np.mean(f<z)),
         factual_better_than_copy_fraction=float(np.mean(f<c)),
         factual_to_zero_ratio=float(f.mean()/max(z.mean(),1e-20)),
         factual_to_copy_ratio=float(f.mean()/max(c.mean(),1e-20)),
         temporal_latent_rms=temporal_rms,posterior_difference_std=float(posterior_difference_std),
         temporal_signal_to_posterior_noise=temporal_rms/max(float(posterior_difference_std),1e-20),
         geometry={k:dict(mean_within_clip_spearman=float(np.mean(v)) if v else None,
                          valid_clips=len(v)) for k,v in comparisons.items()},
         source_state_used_only_for_diagnostic_distance=True,no_sgd=True,
         unsimulated_zero_controls_are_not_causal_proof=True,own_latent_errors_not_shared_physical_units=True)
    save(args.output,result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--step",type=int,default=5000)
    run(p.parse_args())
