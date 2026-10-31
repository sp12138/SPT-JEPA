"""Completed-output-only presentation; no model imports or new fitted readouts."""
import hashlib
import json
from pathlib import Path
import numpy as np

B=Path('/efs/prescottsun/SPT-JEPA');ROOT=B/'runs/cube_pool_reuse2000_20261031';OUT=B/'reports/cube-pool-reuse2000-20261031'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result=read(OUT/'results.json');rows=result['rows'];lookup={r['id']:r for r in rows}
assert result['all12_pass'] and (ROOT/'HOLD_DISPATCH').exists()
def pair(kind,group,metric,factor=1,fmt='.3f'):
    return '/'.join(format(lookup[f'{kind}-seed{s}']['summary'][group][metric]['mean']*factor,fmt)for s in [1,2])
def comparison(candidate,reference,group,metric):
    a,b=lookup[candidate]['cases'][group],lookup[reference]['cases'][group]
    assert [r['case']for r in a]==[r['case']for r in b]
    d=np.asarray([x[metric]-y[metric]for x,y in zip(a,b)],float)
    draws=d[np.random.default_rng(1042).integers(50,size=(5000,50))].mean(1)
    return dict(candidate=candidate,reference=reference,group=group,metric=metric,mean=float(d.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),per_case=d.tolist())
extra=[comparison(f'native12000-seed{s}',f'tf12000-seed{s}',g,k)for s in [1,2]for g in ['nonoracle','oracle_neighborhood','all']for k in ['predicted_hold_success','predicted_pose_cost']]
reports=[read(r['result'])for r in rows];zero={k:float(np.mean([r[k]for r in rows[0]['zero']]))for k in ['pose_cost','position_m','angle_deg','position_pass','angle_pass','joint_pose_pass','hold_success']}
for row in rows:assert row['zero']==rows[0]['zero']
reset_max=0.
for d in (ROOT/'encoded').iterdir():
    for p in d.glob('case*.npz'):
        with np.load(p)as f:reset_max=max(reset_max,float(f['initial_rgb_mse'].max()))
proof=dict(all12_exit0=all(read(ROOT/'exits'/(r['id']+'.json'))['returncode']==0 for r in rows),
    model_parameters_buffers_RNG_unchanged=all(r['state_before']==r['state_after']and r['rng_before']==r['rng_after']for r in reports),
    actual_calls=result['actual_calls'],encoder_maps=6,score_files=sum(len(r['cases'])for r in reports),
    local_encoded_oracle_invariant=result['local_encoded_oracle_invariant'],reset_initial_rgb_mse_max=reset_max,
    sum_timed_seconds=sum(r['seconds']for r in reports),max_allocated_GiB=max(r['peak_gib']for r in reports),
    model_sources=sorted({r['spec']['model_source']['source_commit']for r in reports}),helper_commit='ea11036bf5f62ec8e32cf464f387b66a1d064599',
    closures=[read(ROOT/f'closure-{h}.json')for h in ['dev2','dev13','dev15']],
    results=dict(path=str(OUT/'results.json'),sha256=sha(OUT/'results.json')),zero_action=zero)
(OUT/'review.json').write_text(json.dumps(dict(proof=proof,native_vs_tf_paired=extra),indent=2)+'\n')
kinds=['identity','static','state','local6000','native12000','tf12000']
lines=['# Cube common-pool ranking: all twelve held models','',
 'The new scoring confirms two distinct limitations. Better factual STATE prediction does not consistently improve ordering among nonoracle actions. Separately, accurately predicting a model’s own latent goal cost does not ensure its true encoded metric selects physically successful outcomes. These are conditional diagnoses on one old executed candidate bank, not a causal explanation or a new planning score.','',
 'All cells are seed1/seed2. Fifty cases,22 nonoracle candidates/case. Physical-oracle endpoint pose cost is2.094 and its hold3 success40% for every model; any-candidate endpoint-pose coverage46% and hold coverage40%. Pose cost is max(position/3cm,symmetry-angle/15°).','',
 '|Model|Physical↔encoded rank|Encoded↔predicted rank|Encoded selector pose cost|Predicted selector pose cost|Encoded hold %|Predicted hold %|',
 '|---|---:|---:|---:|---:|---:|---:|']
for k in kinds:
    lines.append('|'+k+'|'+ '|'.join(pair(k,'nonoracle',m,100 if m.endswith('success') else 1,'.0f' if m.endswith('success') else '.3f')for m in ['physical_encoded_rank','encoded_predicted_rank','encoded_oracle_pose_cost','predicted_pose_cost','encoded_oracle_hold_success','predicted_hold_success'])+'|')
lines+=['',
 'All201 local encoder/projection/normalization tensors match within each seed, and the actual encoded selectors agree exactly across identity/STATIC/STATE2000 and local6000. Thus STATE has not changed the actual goal geometry. Nonoracle STATE−identity rank differences are−.0098 [−.0324,.0125] and−.0284 [−.0509,−.0063]. Its hold-success differences are−4 [−12,4]pp and+8 [2,16]pp. In particular seed2 has worse global rank but better selected hold success; it would be wrong to call STATE uniformly worse or claim rank loss explains its planning outcome. Its conditional mean selected physical-cost changes,+.286 [−.151,.787] and+.008 [−.364,.394], are inconclusive.','',
 'STATE’s lower factual MSE therefore does not transfer uniformly to the action-comparison distribution. The contrary evidence matters: in the17 recorded-action-neighborhood candidates, its encoded/predicted rank increases by+.0556 and+.0363 (both conditional CIs include0), while nonoracle rank declines. All39 combined ranks also increase slightly. The mixture can hide the nonoracle behavior; neither aggregate rank nor factual average MSE is a sufficient selected-action statistic.','',
 'TF12k strongly preserves its own nonoracle cost ordering (.951/.949), yet exact encoded-goal selection gives14/18% hold success versus the physical selector40%. Its metric regrets are1.537 [.875,2.280] and.982 [.528,1.544], while signed additional prediction-selection regrets are−.210 [−.743,.273] and+.051 [−.460,.594]. This supports a goal-metric limitation on this bank, with no established sign of additional mean physical regret from prediction in this group. It does not prove a training-objective cause or identify which latent factor caused it.','',
 'For local models both losses remain: the shared metric regrets are.739 [.364,1.207] and.446 [.230,.744], with additional signed prediction regrets varying across models. Native12k has nonoracle metric regrets.728/.506 and prediction regrets.396/.193; its20/20% selected hold does not establish broad superiority from this50-case library. Cross-family exposure and capacity differ; the eight local checkpoint states are1024-dimensional, native192 and TF64. No cross-family raw latent MSE is used.','',
 '|Model|Nonoracle selected distance cm|Selected angle °|Position≤3cm %|Angle≤15° %|Endpoint joint %|Hold3 %|',
 '|---|---:|---:|---:|---:|---:|---:|']
for k in kinds:
    metrics=[('predicted_position_m',100,'.2f'),('predicted_angle_deg',1,'.2f'),('predicted_position_pass',100,'.0f'),('predicted_angle_pass',100,'.0f'),('predicted_joint_pose_pass',100,'.0f'),('predicted_hold_success',100,'.0f')]
    lines.append('|'+k+'|'+'|'.join(pair(k,'nonoracle',m,f,fmt)for m,f,fmt in metrics)+'|')
lines+=['', 'Hold success means the saved trajectory sustained the criterion at some point. Endpoint joint success is a different condition; either can exceed the other. Zero actions give18.06cm mean position error,7.77° angle,2% endpoint joint/hold success, and pose cost6.037. Their relatively good82% angle pass does not solve position.','',
 '|Model|Recorded-neighborhood selected hold %|All39 selected hold %|All39 predicted/encoded rank|All39 selected pose cost|',
 '|---|---:|---:|---:|---:|']
for k in kinds:
    lines.append('|'+k+'|'+pair(k,'oracle_neighborhood','predicted_hold_success',100,'.0f')+'|'+pair(k,'all','predicted_hold_success',100,'.0f')+'|'+pair(k,'all','encoded_predicted_rank')+'|'+pair(k,'all','predicted_pose_cost')+'|')
lines+=['',
 'Adding the17 recorded-neighborhood candidates increases hold coverage to82%; physical and encoded selectors both have56% hold success, although the physical endpoint-pose oracle is nearly exact. The56% is not a success ceiling: endpoint pose minimization does not optimize hold. In this enriched bank, TF’s high overall rank still produces26/26% selected hold, versus raw native48/50% and local6000 52/50%. Native−TF all39 case-paired hold differences and CIs are preserved in review.json. This is not deployable imitation performance or evidence of new-model CEM coverage. TF also has excellent endpoint accuracy when restricted to recorded-neighborhood candidates (mean distance.59/.62cm), which must not be erased by the all39 failure.','',
 'Even normalized encoded-selection regret can conceal task-relevant errors: TF’s all39 regret is only.00333/.00296 of the within-case encoded cost range, but selected physical pose cost is2.556/1.169 versus physical oracle.0135. Relative-to-worst-candidate error and global rank do not control gaps at the relevant minimum. Signed dynamics regret may be negative because prediction errors can cancel metric errors.','',
 'Integrity: all12×50 model/case outputs complete with exit0 and unchanged complete parameters/buffers and RNG. Actual calls equal the frozen budget:12,300 RGB,1800 encoder forwards,3000 predict,600 action_encoder;0 renderer/simulator/training/fit. The six unique encoded banks retain owner/checkpoint/geometry/pool hashes, and all600 scored files retain39 individual costs, original commands and physical truth. Helper ea11036; original model sources32fad0c/d6d0a9f/7a1ab4f/8e0f821 remain separate. Raw native eval BatchNorm was preserved.','',
 'Actual first input was the saved dataset image, matching frozen World dataset-infos injection. Candidate reset images differ by at most{:.5f} uint8 MSE; prior recorded replay discrepancy is at most3.07mm and duplicate qpos replay is exact. Raw commands were not clipped; Gaussian and old CEM families include values outside training[-1,1]. Native float32 scaler subtraction/division is explicit, rather than the old scorer’s float64 intermediate. These fixed outcomes describe one old environment realization, not repeat-averaged native evaluation. The bank lacks warm3+5, fresh cases, current-model CEM proposals and second-plan outcomes. No causal attribution or50-step closed-loop success claim follows. All bootstrap intervals resample50 cases, never39 candidates, and are conditional on fixed trained seeds.','',
 'All12 owned slots were verified empty and exact dispatchers/monitors closed; the root is held. Full data: B/reports/cube-pool-reuse2000-20261031/results.json and B/runs/cube_pool_reuse2000_20261031. Small summary/review preserve all models, all groups and paired intervals. No additional experiments are launched.']
lines=[x.format(reset_max)if '{:.5f}'in x else x for x in lines]
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(proof=proof,native_vs_tf=[x for x in extra if x['group']=='all'and x['metric']=='predicted_hold_success']),indent=2))
