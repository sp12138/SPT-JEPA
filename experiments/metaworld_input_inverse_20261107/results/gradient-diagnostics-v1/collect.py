import json,pathlib,time,statistics,datetime
out=pathlib.Path('/efs/prescottsun/SPT-JEPA/runs/metaworld_input_inverse_20261107_v1/gradient-diagnostics-v1')
plan=json.loads((out/'plan.json').read_text())
def read(path):
 try:return json.loads(path.read_text())
 except (OSError,json.JSONDecodeError):return None
while time.time()<plan['deadline_unix']:
 records=[];errors=[]
 for r in plan['recipes']:
  x=read(out/(r['name']+'.json'));e=read(out/(r['name']+'-error.json'))
  if x is not None:records.append(x)
  elif e is not None:errors.append({'name':r['name'],**e})
 if len(records)+len(errors)==len(plan['recipes']):break
 time.sleep(5)
else:
 (out/'DEADLINE.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Finite15minute diagnostic cap'})+'\n')
 raise SystemExit(1)
groups={}
for r in records:
 assert r['passed']and r['model_unchanged']and r['model_updates']==0
 family=pathlib.Path(r['run']).name.split('-')[0]
 for module,z in r['groups'].items():
  groups.setdefault(family,{}).setdefault(module,[]).append(z)
summary={f:{m:{'models':len(values),'median_inverse_over_main':statistics.median([v['inverse_over_main']for v in values if v['inverse_over_main']is not None])if any(v['inverse_over_main']is not None for v in values)else None,'median_cosine':statistics.median([v['cosine']for v in values if v['cosine']is not None])if any(v['cosine']is not None for v in values)else None}for m,values in modules.items()}for f,modules in groups.items()}
windows=set()
for r in records:
 for ep,rows in zip(r['heldout_windows']['episode'],r['heldout_windows']['rows']):windows.add((ep,tuple(rows)))
result={'passed':not errors,'models':len(records),'errors':errors,'unique_observed_windows':len(windows),'family_groups':summary,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Frozen pergroup main/inverse gradient decomposition,0SGD/0simulation/0baseline execution. Tinysharedheldoutwindows,NOT control performance or causal mechanism proof.'}
tmp=out/'summary.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out/'summary.json')
