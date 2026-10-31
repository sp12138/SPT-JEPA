"""Prepare four held specifications for the unchanged ea11036 scoring runner. No model imports."""
from pathlib import Path
import argparse,copy,hashlib,json
B=Path('/efs/prescottsun/SPT-JEPA');PY='/efs/prescottsun/project_jepa/envs/lewm/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def verify_source(s):
 for k in ['manifest','runtime_manifest']:
  m=read(s[k])
  for p,h in m['files'].items():assert sha(Path(m['source'])/p)==h,p

def main():
 a=argparse.ArgumentParser();a.add_argument('--root',required=True);args=a.parse_args();root=Path(args.root);assert not root.exists(),'Preserve existing output';old=B/'runs/cube_pool_reuse2000_20261031';mature=B/'runs/cube_maturity23342_20261031';plan=read(mature/'plan.json');specs=[];jobs=[];proof=[];verified=set()
 inv=read(B/'reports/cube-pool-reuse-review-20261031/inventory.json');assert inv['cases']==50 and inv['candidates_per_case']==39
 for c in inv['cases_provenance']:assert sha(c['path'])==c['sha256']
 for family,kind,slug,dim in [('native','native','native-lewm',192),('tf','tf','temporal-forecast',64)]:
  for seed in [1,2]:
   oldspecpath=old/'specs'/f'{kind}12000-seed{seed}.json';s=read(oldspecpath);original=copy.deepcopy(s);name=f'{kind}23342-seed{seed}';runname=f'cube-{slug}-seed{seed}';run=mature/runname
   prior=next(j for j in plan['jobs'] if j['id']==f'original-primary-{runname}');assert prior['source']==s['model_source'] and prior['manifest_sha256']==inv['manifest_sha256']
   manifest=Path(prior['command'][prior['command'].index('--manifest')+1]);assert sha(manifest)==inv['manifest_sha256'];assert read(manifest)['pairs']==[c['pair'] for c in inv['cases_provenance']]
   cfg=run/'config.json';weights=run/'model-step0023342/weights.pt';state=read(mature/f'transition-{runname}/state-audit.json');assert state['step']==23342 and state['state_dim']==dim
   assert read(cfg)['model']==read(s['config'])['model'];assert read(cfg)['seed']==seed
   assert sha(cfg)==state['config_sha256'] and sha(weights)==state['weights_sha256']
   for source in [s['helper_source'],s['model_source']]:
    if source['source_commit'] not in verified:verify_source(source);verified.add(source['source_commit'])
   runner=Path(s['helper_source']['source'])/'experiments/cube_pool_reuse_20261031/assay.py';common=runner.parents[1]/'planner_context_20261031/assay.py'
   assert sha(runner)==s['helper_sha256'] and sha(common)==s['common_sha256'];assert sha(s['inventory'])==s['inventory_sha256']
   assert s['encoder_owner'] and s['encoder_geometry_sha256'] is None and s['dimension']==dim
   s.update(id=name,kind=f'{kind}23342',config=str(cfg),config_sha256=sha(cfg),weights=str(weights),weights_sha256=sha(weights),encoded_root=str(root/'encoded'/name),owner_id=name,output=str(root/name))
   allowed={'id','kind','config','config_sha256','weights','weights_sha256','encoded_root','owner_id','output'};changed={k for k in s if s[k]!=original[k]};assert changed<=allowed
   sp=root/'specs'/f'{name}.json';write(sp,s);specs.append(s)
   cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=15s','420s',PY,'-B','-u',str(runner),'run','--spec',str(sp)]
   jobs.append(dict(id=name,host='dev13',gpu=seed-1,hold=True,output=s['output'],source=s['helper_source'],depends_on=[] if family=='native' else [f'native23342-seed{seed}'],command=cmd))
   proof.append(dict(id=name,spec=str(sp),spec_sha256=sha(sp),changed_fields=sorted(changed),old_spec=str(oldspecpath),old_spec_sha256=sha(oldspecpath),transition=str(mature/f'transition-{runname}/state-audit.json'),transition_sha256=sha(mature/f'transition-{runname}/state-audit.json'),config_sha256=s['config_sha256'],weights_sha256=s['weights_sha256'],original_model_config_exact=True,model_source_exact=True))
 totals={k:sum(s['budget'][k] for s in specs) for k in specs[0]['budget']};assert totals==dict(rgb_frames=8200,encoder=1200,predict=1000,action_encoder=200,renderer=0)
 protocol=dict(root=str(root),helper_source=specs[0]['helper_source'],helper_sha256=specs[0]['helper_sha256'],common_sha256=specs[0]['common_sha256'],specs=specs,inventory=specs[0]['inventory'],inventory_sha256=specs[0]['inventory_sha256'],totals=totals,simulator_calls=0,fit_steps=0,training_steps=0,actual_model_calls_during_preparation=0,allocated_slots=[['dev13',0],['dev13',1]],hard_wall_seconds=900,per_job_timeout_seconds=420,timeout_kill_grace_seconds=15,activation_required='Root review and actual-free verification. Do not activate here. Launcher must enforce 900s from actual batch activation, stopping only owned exact processes.',semantics='Same full old50x39 bank; native first plan H1/T5 with configured max history3; unchanged raw float32 action normalization; raw BN; all four models own new encoded banks. No current-CEM coverage or full50-step success claim.',analysis='Compare each seed 12000 to23342 on identical candidates, grouping nonoracle22/oracle-neighborhood17/all39; predicted-vs-encoded ranks and signed physical selection regret. Preserve all50 cases and two seeds; no outcome selection.',prior_resource_evidence=dict(path='experiments/cube_pool_reuse_20261031/results/gate2000/review.json',max_allocated_gib=.27028751373291016,sum12_timed_seconds=106.88331580162048,limitation='Historical measurement, not a guaranteed bound; excludes all process-start/source-hash overhead. No new GPU query/measurement in preparation.'))
 write(root/'protocol.json',protocol);write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment={'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},jobs=jobs));(root/'HOLD_DISPATCH').write_text('Prepared only. Root review/launch authorization and actual-free check required.\n')
 review=dict(passed=True,root=str(root),helper_commit=specs[0]['helper_source']['source_commit'],helper_sha256=specs[0]['helper_sha256'],common_sha256=specs[0]['common_sha256'],inventory_sha256=specs[0]['inventory_sha256'],old50_pool_files_sha_exact=True,primary50_pairs_exact=True,specs=proof,totals=totals,model_source_commits=sorted(verified),protocol_sha256=sha(root/'protocol.json'),candidate_plan_sha256=sha(root/'candidate-plan.json'),preparation_script_sha256=sha(__file__),actual_model_calls=0,simulator_calls=0,launches=0,old_roots_unchanged=True)
 write(root/'review.json',review);print(json.dumps({k:review[k]for k in ['passed','root','helper_commit','totals','protocol_sha256','candidate_plan_sha256']},indent=2))
if __name__=='__main__':main()
