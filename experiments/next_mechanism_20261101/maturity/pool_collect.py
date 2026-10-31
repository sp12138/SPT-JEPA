"""Completed-only CPU report for unchanged old-pool scoring, four12000→four23342 models."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from scipy.stats import spearmanr
B=Path('/efs/prescottsun/SPT-JEPA');GROUPS=['nonoracle','oracle_neighborhood','all']
METRICS=['coverage_hold','coverage_endpoint_pose','physical_encoded_rank','encoded_predicted_rank','physical_predicted_rank','position_encoded_rank','angle_encoded_rank','predicted_hold_success','predicted_joint_pose_pass','predicted_pose_cost','encoded_oracle_hold_success','encoded_oracle_joint_pose_pass','encoded_oracle_pose_cost','physical_oracle_hold_success','physical_oracle_joint_pose_pass','physical_oracle_pose_cost','metric_regret','signed_dynamics_regret','total_pose_regret','encoded_selection_regret_relative_range']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def interval(x):
 x=np.asarray([v for v in x if v is not None],float)
 if not len(x):return dict(mean=None,ci95=None,n=0)
 b=x[np.random.default_rng(1042).integers(len(x),size=(5000,len(x)))].mean(1)
 return dict(mean=float(x.mean()),ci95=np.quantile(b,[.025,.975]).tolist(),n=len(x))
def corr(x,y):return None if np.ptp(x)==0 or np.ptp(y)==0 else float(spearmanr(x,y).statistic)
def summarize(report,oldreport):
 s=report['spec'];conditions={k:[]for k in GROUPS};zero=[];oldcases={r['case']:r for r in oldreport['cases']}
 assert len(report['cases'])==50 and {r['case']for r in report['cases']}==set(range(50))
 for rec in report['cases']:
  assert sha(rec['file'])==rec['sha256'];oldrec=oldcases[rec['case']];assert rec['pool_sha256']==oldrec['pool_sha256']
  with np.load(rec['file'])as f,np.load(oldrec['file'])as o:
   assert sha(oldrec['file'])==oldrec['sha256']
   for key in ['raw_actions','family','physical_cost','position_error_m','orientation_error_deg','success','hold_count']:assert np.array_equal(f[key],o[key]),key
   pc,ec,ph=f['predicted_cost'],f['encoded_cost'],f['physical_cost'][:,-1];pos=f['position_error_m'][:,-1];ang=f['orientation_error_deg'][:,-1];success=f['success'].any(1);family=f['family']
   assert len(pc)==len(ec)==len(ph)==39 and np.isfinite(pc).all()and np.isfinite(ec).all()
   def truth(i):return dict(index=int(i),family=str(family[i]),pose_cost=float(ph[i]),position_m=float(pos[i]),angle_deg=float(ang[i]),position_pass=float(pos[i]<=.03),angle_pass=float(ang[i]<=15),joint_pose_pass=float(pos[i]<=.03 and ang[i]<=15),hold_success=float(success[i]))
   zero.append(dict(case=rec['case'],**truth(int(np.flatnonzero(family=='zero')[0]))))
   groups=dict(nonoracle=np.flatnonzero(np.isin(family,['zero','gaussian'])|np.char.startswith(family,'cem_')),oracle_neighborhood=np.flatnonzero(np.char.startswith(family,'recorded')),all=np.arange(39));assert {k:len(v)for k,v in groups.items()}==dict(nonoracle=22,oracle_neighborhood=17,all=39)
   for group,ids in groups.items():
    pi,ei,hi=[int(ids[np.argmin(v[ids])])for v in [pc,ec,ph]];selections={name:truth(i)for name,i in [('predicted',pi),('encoded_oracle',ei),('physical_oracle',hi)]}
    v=dict(case=rec['case'],count=len(ids),physical_encoded_rank=corr(ph[ids],ec[ids]),encoded_predicted_rank=corr(ec[ids],pc[ids]),physical_predicted_rank=corr(ph[ids],pc[ids]),position_encoded_rank=corr(pos[ids],ec[ids]),angle_encoded_rank=corr(ang[ids],ec[ids]),coverage_hold=float(success[ids].any()),coverage_endpoint_pose=float(((pos[ids]<=.03)&(ang[ids]<=15)).any()),metric_regret=float(ph[ei]-ph[hi]),signed_dynamics_regret=float(ph[pi]-ph[ei]),total_pose_regret=float(ph[pi]-ph[hi]),encoded_selection_regret_relative_range=float((ec[pi]-ec[ei])/np.ptp(ec[ids]))if np.ptp(ec[ids])else 0.,selections=selections)
    assert abs(v['total_pose_regret']-v['metric_regret']-v['signed_dynamics_regret'])<1e-10
    for selector,item in selections.items():
     for key in ['pose_cost','position_m','angle_deg','position_pass','angle_pass','joint_pose_pass','hold_success']:v[selector+'_'+key]=item[key]
    conditions[group].append(v)
 return dict(id=s['id'],kind=s['kind'],seed=s['seed'],summary={g:{k:interval([r[k]for r in rs])for k in rs[0]if k not in ['case','count','selections']}for g,rs in conditions.items()},cases=conditions,zero=zero)

def main():
 a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--output',required=True);a.add_argument('--completion-record',required=True);args=a.parse_args();root=Path(args.root);out=Path(args.output)
 plan=read(root/'candidate-plan.json');protocol=read(root/'protocol.json');assert len(plan['jobs'])==4
 completion=read(args.completion_record)
 # Actual synchronous-launcher closure and four original OS-exit files; no synthesized statuses.
 assert completion['all_four_pass'] is True
 exits={}
 for j in plan['jobs']:
  ep=root/'exits'/(j['id']+'.json');x=read(ep);assert x['id']==j['id'] and x['returncode']==0 and x['seconds']>=0 and x['end_utc'];exits[j['id']]={'record':x,'path':str(ep),'sha256':sha(ep)}
 assert len(exits)==4
 oldpath=B/'reports/cube-pool-reuse2000-20261031/results.json';old=read(oldpath);assert old['all12_pass'];oldrows={r['id']:r for r in old['rows']};rows=[];paired=[];reports=[]
 for j in plan['jobs']:
  rp=Path(j['output'])/'result.json';r=read(rp);s=read(root/'specs'/(j['id']+'.json'));assert r['spec']==s and r['passed']and r['state_before']==r['state_after']and r['rng_before']==r['rng_after'];assert r['actual_calls']==s['budget']
  oldid=j['id'].replace('23342','12000');ref=oldrows[oldid];orr=read(ref['result']);assert sha(ref['result'])==ref['result_sha256'];new=summarize(r,orr);new.update(result=str(rp),result_sha256=sha(rp));rows.append(new);reports.append(r)
  # Verify reused formulas reproduce the existing old aggregate to numerical precision.
  recomputed=summarize(orr,orr)
  for group in GROUPS:
   assert [v['case']for v in new['cases'][group]]==[v['case']for v in ref['cases'][group]]
   for metric in METRICS:
    x,y=recomputed['summary'][group][metric]['mean'],ref['summary'][group][metric]['mean'];assert x==y or(x is not None and y is not None and abs(x-y)<1e-12)
    diffs=[x[metric]-y[metric]for x,y in zip(new['cases'][group],ref['cases'][group])if x[metric]is not None and y[metric]is not None]
    paired.append(dict(candidate=new['id'],reference=oldid,group=group,metric=metric,**interval(diffs)))
 oldselected=[oldrows[r['id'].replace('23342','12000')]for r in rows];allrows=oldselected+rows
 for group in GROUPS:
  for key in ['coverage_hold','coverage_endpoint_pose','physical_oracle_pose_cost','physical_oracle_hold_success','physical_oracle_joint_pose_pass']:
   ref=[c[key]for c in allrows[0]['cases'][group]];assert all([c[key]for c in r['cases'][group]]==ref for r in allrows)
 totals={k:sum(r['actual_calls'][k]for r in reports)for k in reports[0]['actual_calls']};assert totals==protocol['totals']
 proof=dict(all4_actual_exit0=True,completion_record=dict(path=args.completion_record,sha256=sha(args.completion_record),record=completion),actual_exit_records=exits,all4_weights_buffers_RNG_unchanged=True,all200_case_pool_hashes_match=True,all200_action_and_physical_truth_arrays_exact_vs12000=True,old_formula_reproduced=True,actual_calls=totals,simulator_calls=0,training_steps=0,sum_timed_seconds=sum(r['seconds']for r in reports),max_allocated_gib=max(r['peak_gib']for r in reports),helper_commit=protocol['helper_source']['source_commit'],helper_sha256=protocol['helper_sha256'],protocol_sha256=sha(root/'protocol.json'),old_results=dict(path=str(oldpath),sha256=sha(oldpath)),collector_sha256=sha(__file__))
 limitations=['Fixed old50x39 bank, not current CEM proposal coverage or complete50-step planning success.','First-plan H1/T5; no warm3+5/fresh/second-segment outcomes.','22 nonoracle is primary;17 recorded-action-neighborhood and all39 reported separately.','Physical labels are diagnostic only. Endpoint joint and anywhere-during-trajectory hold3 are distinct.','Metric plus signed-dynamics regret is an algebraic selector identity, not a unique causal decomposition; signed term can be negative.','Bootstrap resamples50 paired cases, not candidates or independent training seeds.','Old reset-vs-dataset RGB mismatch and one-realization simulator variability remain; no new simulation.','Raw native BN and original raw commands retained; no common-coordinate raw latent MSE comparison.']
 full=dict(proof=proof,rows=allrows,paired=paired,limitations=limitations);write(out/'results.json',full)
 small={**full,'rows':[{k:v for k,v in r.items()if k not in ['cases','zero']}for r in allrows]};write(out/'summary.json',small)
 def pair(kind,g,metric,scale=1):return '/'.join(f"{next(r for r in allrows if r['id']==kind+'-seed'+str(s))['summary'][g][metric]['mean']*scale:.3f}"for s in [1,2])
 lines=['# Mature Cube: identical old candidate-bank comparison','', 'All cells are seed1/seed2;50 paired cases. Original frozen scorer ea11036; no new simulator, fitting, renderer or training.','']
 for group,label in [('nonoracle','Primary:22 nonoracle'),('oracle_neighborhood','Appendix:17 recorded-action-neighborhood'),('all','Appendix:all39')]:
  lines += [f'## {label}','', '|Model|Physical↔encoded rank|Encoded↔predicted rank|Encoded hold %|Predicted hold %|Encoded joint %|Predicted joint %|Encoded pose cost|Predicted pose cost|Metric regret|Signed dynamics regret|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
  for kind in ['native12000','native23342','tf12000','tf23342']:
   keys=[('physical_encoded_rank',1),('encoded_predicted_rank',1),('encoded_oracle_hold_success',100),('predicted_hold_success',100),('encoded_oracle_joint_pose_pass',100),('predicted_joint_pose_pass',100),('encoded_oracle_pose_cost',1),('predicted_pose_cost',1),('metric_regret',1),('signed_dynamics_regret',1)]
   lines.append('|'+kind+'|'+'|'.join(pair(kind,group,k,f)for k,f in keys)+'|')
  z=allrows[0]['summary'][group];lines += ['',f"Shared coverage: hold {100*z['coverage_hold']['mean']:.1f}%, endpoint joint {100*z['coverage_endpoint_pose']['mean']:.1f}%; physical endpoint-cost selector pose {z['physical_oracle_pose_cost']['mean']:.4f}, hold {100*z['physical_oracle_hold_success']['mean']:.1f}%. This selector is not a hold-success ceiling.",'']
 lines += ['## Paired changes (23342−12000)','', '|Model seed|Group|Metric|Mean delta|95% conditional case interval|','|---|---|---|---:|---:|']
 for v in paired:
  if v['group']=='nonoracle'and v['metric']in ['encoded_oracle_pose_cost','predicted_pose_cost','encoded_oracle_hold_success','predicted_hold_success','encoded_predicted_rank','metric_regret','signed_dynamics_regret']:
   lines.append(f"|{v['candidate']}|{v['group']}|{v['metric']}|{v['mean']:.4f}|[{v['ci95'][0]:.4f},{v['ci95'][1]:.4f}]|")
 lines+=['','## Scope','']+['- '+v for v in limitations]+['', 'Full per-case arrays and file provenance remain in EFS results.json. The small summary retains all8 model/budget rows, all3 groups and all paired intervals.']
 (out/'TABLES.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(proof=proof,nonoracle=[dict(id=r['id'],**{k:r['summary']['nonoracle'][k]['mean']for k in ['physical_encoded_rank','encoded_predicted_rank','encoded_oracle_hold_success','predicted_hold_success','encoded_oracle_pose_cost','predicted_pose_cost','metric_regret','signed_dynamics_regret']})for r in allrows]),indent=2))
if __name__=='__main__':main()
