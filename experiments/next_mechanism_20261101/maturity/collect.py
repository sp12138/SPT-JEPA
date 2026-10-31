"""CPU/read-only extraction of completed maturity diagnostics; never loads a model."""
from pathlib import Path
import argparse,hashlib,json,math

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p);return json.loads(p.read_text()),{'path':str(p),'sha256':sha(p)}
def transition(p):
 d,src=read(p);c=d['conditions'];m=lambda k:c[k]['mse'];clean=m('clean_rollout');teacher=m('teacher_forced_prefix')
 row={'source':src,**{k:d[k] for k in ['step','model_class','config_sha256','weights_sha256','manifest_sha256','cases','window','state_dim','observed_state_std_mean','observed_state_rms']},'mse':{k:v['mse'] for k,v in c.items()},'horizon_mse':{k:v['per_horizon_mse'] for k,v in c.items()},'clean_over_copy':clean/m('initial_carry'),'clean_over_zero':clean/m('zero_prediction'),'teacher_over_shifted_copy':teacher/m('shifted_observed_carry'),'clean_over_teacher':clean/teacher,'excess_clean_over_teacher_fraction':(clean-teacher)/clean,'intervention_error_over_clean':{k:m(k)/clean for k in ['other_initial','zero_initial','wrong_action_order','other_episode_actions']},'intervention_effect':{k:v['intervention_effect_mse'] for k,v in c.items() if k in ['other_initial','zero_initial','wrong_action_order','other_episode_actions']},'access_contract':d['access_contract']}
 row['horizon_clean_over_copy']=[x/y for x,y in zip(c['clean_rollout']['per_horizon_mse'],c['initial_carry']['per_horizon_mse'])]
 row['horizon_clean_over_zero']=[x/y for x,y in zip(c['clean_rollout']['per_horizon_mse'],c['zero_prediction']['per_horizon_mse'])]
 wp=Path(p).parent/'windows.json';w,ws=read(wp);row['windows_source']=ws;row['windows']=w
 return row

def health(p):
 d,src=read(p);s=d['summary'];return {'source':src,'feature_std':d['feature_std'],'summary':s,'high_noise_forecast_over_posterior':s['fm_t0.95_forecast']/s['fm_t0.95_posterior'],'high_noise_wrong_action_over_forecast':s['fm_t0.95_wrong_actions']/s['fm_t0.95_forecast'],'scope':d['scope']}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',default='/efs/prescottsun/SPT-JEPA');ap.add_argument('--output',required=True);a=ap.parse_args();B=Path(a.base);R=B/'runs';new=R/'cube_maturity23342_20261031';out={'scope':'Completed JSON-only CPU analysis; no model, GPU, simulation, fitting, or new cohort. Raw MSE is not compared across representations.','script_sha256':sha(__file__),'runs':[],'missing_historical_transition':[]}
 plan,plan_src=read(new/'plan.json');out['plan_source']=plan_src;out['frozen_model_sources']={j['family']:j['source'] for j in plan['jobs']}
 for family,slug,oldroot,oldstep in [('local','spatial-local','local_isotropic6000_20261031',6000),('native','native-lewm','native_cube12000_20261031',12000),('tf','temporal-forecast','cube_tf12000_20261031',12000)]:
  for seed in [1,2]:
   name=f'cube-{slug}-seed{seed}';n=transition(new/f'transition-{name}/state-audit.json');row={'family':family,'seed':seed,'current':n,'history':{}};cfg,cs=read(new/name/'config.json');row['config_source']=cs;row['model_config']={k:v for k,v in cfg['model'].items() if not isinstance(v,(list,dict))};row['large_config_fields_omitted_but_hash_bound']=[k for k,v in cfg['model'].items() if isinstance(v,(list,dict))]
   previous=[(oldroot,oldstep)]
   if family=='native':previous.insert(0,('native_lewm6000_20261031',6000))
   if family=='tf':previous.insert(0,('carrier_learn6000_20261031',6000))
   for old,budget in previous:
    h={};tp=R/old/f'transition-{name}/state-audit.json';hp=R/old/f'health-{name}/health.json'
    if tp.exists():
     h['transition']=transition(tp);h['exact_windows_json_equal']=h['transition']['windows']==n['windows'];h['manifest_sha_equal']=h['transition']['manifest_sha256']==n['manifest_sha256'];h['clean_over_copy_change']=n['clean_over_copy']-h['transition']['clean_over_copy'];h['teacher_over_shifted_copy_change']=n['teacher_over_shifted_copy']-h['transition']['teacher_over_shifted_copy']
    else:out['missing_historical_transition'].append(str(tp))
    if hp.exists():h['health']=health(hp)
    row['history'][str(budget)]=h
   hp=new/f'health-{name}/health.json'
   if hp.exists():
    row['current_health']=health(hp)
    for h in row['history'].values():
     if 'health' in h:h['forecast_health_relative_change']=row['current_health']['summary']['fm_t0.95_forecast']/h['health']['summary']['fm_t0.95_forecast']-1
   out['runs'].append(row)
 for row in out['runs']:
  row['current'].pop('windows')
  for h in row['history'].values():
   if 'transition' in h:h['transition'].pop('windows')
 out['limitations']=['Wrong-action interventions measure dependence, not accurate counterfactual predictions.','Observed targets are model coordinates; low normalized error does not establish retention of task-relevant physical information.','Clean rollout starts at one real frame; native planning has history3 and horizon5, so this is not identical to a complete CEM episode.','Teacher-forcing gap is an input/history intervention, not an additive causal attribution of error.','Old TF6000/12000 roots contain no matched common64-window transition audit; renderer health is not substituted for clean latent forecasting.','23342 is two loader lengths and 18.69% of the original LR schedule, not a proof of convergence.']
 path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 for r in out['runs']:
  n=r['current'];print(r['family'],r['seed'],'copy',round(n['clean_over_copy'],5),'zero',round(n['clean_over_zero'],5),'teacher',round(n['teacher_over_shifted_copy'],5),'clean/teacher',round(n['clean_over_teacher'],3),'samecases',{k:v.get('exact_windows_json_equal') for k,v in r['history'].items()})
if __name__=='__main__':main()
