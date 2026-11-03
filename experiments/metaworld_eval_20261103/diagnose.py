"""Read-only observed-future diagnostics; factual actions are not optimal labels."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.metaworld_eval_20261103.evaluate import model_sha, file_sha


def main(checkpoint,data,output,windows=32,tasks=50):
    import stable_worldmodel as swm
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    from vforge.ops import OpsConfig,initialize_ops
    torch.set_num_threads(2)
    initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=torch.device('cuda'),parallel_state=None)
    # Ops initialization sets backend flags; override afterward,as in MPC.
    torch.set_float32_matmul_precision('highest')
    output=Path(output);output.mkdir(exist_ok=False,parents=True)
    model=swm.wm.utils.load_pretrained(checkpoint).cuda().eval().requires_grad_(False)
    before=model_sha(model);dataset=BenchmarkWindows(data,window=8,split='test')
    index=json.loads((Path(data)/'pack-index.json').read_text())
    metadata={p['episode_id']:p for bundle in index['bundles'] for p in bundle['phases']}
    eligible={}
    for i,ep in enumerate(dataset.episodes):
        record=metadata[int(dataset.episode_ids[ep])];task=record['task_id']
        start=0 if i==0 else int(dataset.ends[i-1])
        eligible.setdefault(task,[]).extend(range(start,int(dataset.ends[i])))
    assert len(eligible)==50
    mean=torch.tensor(dataset.mean,device='cuda');std=torch.tensor(dataset.std,device='cuda')
    rows=[];manifest=[]
    with torch.inference_mode():
        for task in sorted(eligible)[:tasks]:
            rng=np.random.default_rng(20261104+task*1009)
            # Some short tasks have fewer than32 valid held-out windows.
            # Use every available window once;never invent extra samples.
            selected=rng.choice(eligible[task],size=min(windows,len(eligible[task])),replace=False).tolist()
            originals=[int(dataset.episode_ids[dataset.locate(i)[0]])//5 for i in selected]
            cross_episode=len(set(originals))>=2
            perm=[next((j for j in range(len(selected)) if originals[j]!=originals[i]),i) for i in range(len(selected))]
            features=[];actions=[];contexts=[];records=[]
            for batch in DataLoader(Subset(dataset,selected),batch_size=8,num_workers=0):
                normalized=model.normalize(batch['pixels'].cuda())
                features.append(model.encode(dict(pixels=normalized))['emb'])
                actions.append(batch['action'].cuda())
                contexts.append(normalized[:,:2])
                records.extend(zip(batch['episode'].tolist(),batch['start'].tolist()))
            z=torch.cat(features);a=torch.cat(actions);b=len(z)
            factual=a[:,1:7]
            # Physical zero is -mean/std, not a zero normalized action.
            zero=(-mean/std)[None,None,:].expand_as(factual)
            candidates=torch.stack([factual,factual.flip(1),factual[perm],zero],dim=1)
            # Use the same uncached public rollout as MPC. Its native history
            # grows with imagined frames; context2 is not a history2 override.
            pixel=torch.cat(contexts)
            info=dict(pixels=pixel[:,None].expand(-1,4,-1,-1,-1,-1),
                      action_history=a[:,None,:1].expand(-1,4,-1,-1))
            pred=model.rollout(info,candidates)['predicted_emb']
            assert pred.shape[:3]==(b,4,8) and torch.isfinite(pred).all()
            context_error=(pred[:,0,:2]-z[:,:2]).abs()
            actual=z[:,2:];copy=z[:,1:2].expand_as(actual)
            error=(pred[:,0,2:]-actual).square().mean(-1)
            copy_error=(copy-actual).square().mean(-1)
            goal_cost=(pred[:,:,-1]-z[:,-1:, :]).square().mean(-1)
            effect=(pred[:,1:,-1]-pred[:,:1,-1]).square().mean(-1)
            centered=z.flatten(0,1)-z.flatten(0,1).mean(0)
            sv=torch.linalg.svdvals(centered.float());power=sv.square();p=power/power.sum().clamp_min(1e-30)
            row=dict(task_id=task,windows=b,requested_windows=windows,available_windows=len(eligible[task]),
                     cross_episode_control_available=cross_episode,original_episodes=sorted(set(originals)),
                     latent_std=float(z.flatten(0,1).std(0).mean()),
                     effective_rank=float(torch.exp(-(p*p.clamp_min(1e-30).log()).sum())),
                     factual_mse_by_horizon=error.mean(0).tolist(),copy_mse_by_horizon=copy_error.mean(0).tolist(),
                     factual_to_copy_ratio_by_horizon=(error.mean(0)/copy_error.mean(0).clamp_min(1e-30)).tolist(),
                     factual_goal_rank_win_rate=[float((goal_cost[:,0]<goal_cost[:,k]).float().mean()) if k!=2 or cross_episode else None for k in [1,2,3]],
                     factual_goal_rank_tie_rate=[float((goal_cost[:,0]==goal_cost[:,k]).float().mean()) if k!=2 or cross_episode else None for k in [1,2,3]],
                     action_effect_mse=[float(effect[:,k].mean()) if k!=1 or cross_episode else None for k in range(3)],
                     candidate_names=['factual','time_reverse','different_original_same_task','physical_zero'],
                     factual_endpoint_mse=float(goal_cost[:,0].mean()),
                     context_encoding_max_abs=float(context_error.max()),
                     future_motion_mse=float(copy_error[:,-1].mean()))
            manifest.append(dict(task_id=task,indices=selected,episode_start=[list(x) for x in records],permutation=perm))
            rows.append(row);print(json.dumps(row),flush=True)
    after=model_sha(model);assert before==after
    manifest_path=output/'windows.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    report=dict(passed=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                checkpoint=checkpoint,weights_sha256=file_sha(Path(checkpoint)/'weights.pt'),
                model_state_before=before,model_state_after=after,model_updates=0,simulator_calls=0,
                heldout_manifest_sha256=file_sha(manifest_path),tasks=len(rows),max_windows_per_task=windows,
                rows=rows,split='test',context=2,horizon=6,
                scope='Observed future RGB is the diagnostic target. Executed actions are factual,not assumed optimal. '
                      'Alternative actions are not simulated;ranking does not establish actual counterfactual control. '
                      'Task IDs only stratify read-only reports and never enter learner or planner. '
                      'Same fixed indices/alternatives across all models;no parameter or planner changes. '
                      'Use all available windows if fewer than requested;unavailable cross-episode controls are null.')
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for arg in ['checkpoint','data','output']:parser.add_argument('--'+arg,required=True)
    parser.add_argument('--windows',type=int,default=32);parser.add_argument('--tasks',type=int,default=50)
    args=parser.parse_args();main(args.checkpoint,args.data,args.output,args.windows,args.tasks)
