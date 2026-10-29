from pathlib import Path
import json,math,time,torch
torch.set_num_threads(2);b=Path('/efs/prescottsun/SPT-JEPA');r=b/'runs/promising_epoch1_20261029';p=json.loads((r/'plan.json').read_text());rows=[];initials={};started=time.time()
def finite(v):
 if torch.is_tensor(v):assert torch.isfinite(v).all()
 elif isinstance(v,dict):
  for x in v.values():finite(x)
 elif isinstance(v,(list,tuple)):
  for x in v:finite(x)
 elif isinstance(v,float):assert math.isfinite(v)
for j in p['jobs']:
 if j['phase']!='first-epoch':continue
 folder=Path(j['output']);assert json.loads((r/'exits'/(j['id']+'.json')).read_text())['returncode']==0
 stage=json.loads((folder/'stage-complete.json').read_text());assert stage['initial_step']==j['initial_step'] and stage['stop_step']==j['stop_step'] and stage['stop_step']==12491
 weights=torch.load(folder/f"model-step{j['stop_step']:07d}/weights.pt",map_location='cpu',weights_only=True);finite(weights)
 state=torch.load(folder/'last.ckpt',map_location='cpu',weights_only=False);assert state['global_step']==j['stop_step'] and state['campaign_samples']==128*j['stop_step'] and state['campaign_rng'] and state['optimizer_states'] and state['lr_schedulers']
 assert set(state['state_dict'])=={'model.'+k for k in weights} and all(torch.equal(v,state['state_dict']['model.'+k]) for k,v in weights.items())
 finite(state['optimizer_states']);finite(state['lr_schedulers'])
 trace=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()];assert all(math.isfinite(v['loss']) for v in trace)
 for prefix in ['encoder','predictor','renderer']:assert any(v.get('grad_norm_'+prefix,0)>0 and math.isfinite(v['grad_norm_'+prefix]) for v in trace)
 health=json.loads((r/('health-'+j['id'])/'metrics.json').read_text());assert not health['collapsed']
 rows.append({'id':j['id'],'all_finite':True,'global_step':j['stop_step'],'health':health,'all3gradient_groups_nonzero_finite':True})
 del state,weights

out=r/'reports/EPOCH1-AUDIT.json';assert not out.exists()
record={'all16pass':len(rows)==16,'new_seed_initial_tensors_verified_in_prior600audit':True,'seconds':time.time()-started,'rows':rows,'decision':'Onlyto2epochsaftercompletedfirstepochtaskreview;preserveallseedsandoriginalmatchedcontrols.Firstepochhealthisnotmatureperformance.'}
assert record['all16pass'];out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'all16pass':True,'seconds':record['seconds']}))
