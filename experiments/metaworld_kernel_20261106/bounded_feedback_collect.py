
"""Read-only finite synthesis; explicit plan file permits audited collector adoption."""
import pathlib,json,time,sys,subprocess,hashlib
root=pathlib.Path(sys.argv[1]);plan_path=pathlib.Path(sys.argv[2])if len(sys.argv)>2 else root/'plan.json'
plan=json.loads(plan_path.read_text());b=pathlib.Path('/efs/prescottsun/SPT-JEPA/runs')
def read(path):
 for i in range(3):
  try:return json.loads(path.read_text())
  except OSError:
   if i==2:raise
   time.sleep(.2)
def save(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
while True:
 summaries={}
 for name in plan['roots']:
  r=b/name;training=(r/'training-plan.json').exists();p=r/('training-plan.json'if training else'plan.json');root_plan=read(p);recipes=root_plan['recipes'];expected=len(recipes);f=r/'results/summary.json'
  d=read(f)if f.exists()else dict(complete=0,errors=0,scores=[])
  terminal={};sgd_complete=0
  for recipe in recipes:
   out=pathlib.Path(recipe['run']if training else recipe['output']);done=out/'complete.json'
   if done.exists():terminal[out.name]=read(done)
   if training:
    cap=root_plan['stages'][-1];audit=out/f'audit-{cap}.json';archive=out/f'archive-step{cap:07d}/manifest.json'
    if audit.exists()and archive.exists():
     proof=read(audit);saved=read(archive)
     sgd_complete+=bool(proof['passed']and proof['step']==cap and saved['step']==cap)
  summaries[name]=dict(expected_runs=expected,complete=d['complete'],errors=d['errors'],disposed=d['complete']+d['errors']==expected,scores=d['scores'],
   training=training,completed_native_SGD_runs=sgd_complete,
   completed_end_to_end_training_runs=sum(bool(v.get('trained',v.get('step',0)>0))for v in terminal.values())if training else 0,
   skipped_without_SGD=sum(v.get('trained')is False and v.get('step')==0 for v in terminal.values())if training else 0,terminal_details=terminal)
 all_done=all(x['disposed']for x in summaries.values());timed_out=time.time()>plan['deadline_unix']
 save(root/'results/progress.json',dict(all_done=all_done,timed_out=timed_out,roots=summaries,new_updates=0,new_evaluations=0))
 if all_done or timed_out:break
 time.sleep(30)
assert hashlib.sha256(pathlib.Path(plan['compare']).read_bytes()).hexdigest()==plan['compare_sha256']
subprocess.run([sys.executable,plan['compare'],'--output',str(root/'results/paired-comparisons.json')],check=True)
save(root/'results/final-summary.json',dict(all_done=all_done,timed_out=timed_out,roots=summaries,new_updates=0,new_evaluations=0,scope=plan['scope'],public_SOTA_claim=False))
