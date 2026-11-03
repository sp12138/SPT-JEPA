"""Single owned-startup snapshot;actual state/gradient evidence,not performance."""
import argparse
import datetime
import math
from pathlib import Path
import subprocess

import psutil
from experiments.metaworld_eval_20261103.worker import read,save,digest


def main(root,host,output):
    recipes=[r for r in read(root/'plan.json')['recipes'] if r['host']==host]
    ownership=read(root/('launch-eio-v2-dev15.json' if host=='dev15' else 'launch-dev13.json'))['jobs']
    rows=[]
    for r in recipes:
        run=Path(r['run']);proof=read(run/'FORK.json');entry=proof['entry_step'];first=r.get('stages',[8242])[0]
        a=run/f'audit-step{entry}.json';b=run/f'audit-step{first}.json';p=run/'planning-preflight/metrics.json'
        status=read(run/'status.json') if (run/'status.json').exists() else {}
        audit=read(a) if a.exists() else {};two=read(b) if b.exists() else {};planning=read(p) if p.exists() else {}
        groups=list(two.get('gradients',{}));periodic={}
        if (run/'metrics.jsonl').exists():
            for line in (run/'metrics.jsonl').read_text().splitlines():
                item=__import__('json').loads(line)
                if groups and all('grad_norm_'+k in item for k in groups):periodic=item
        gradients={k:periodic.get('grad_norm_'+k) for k in groups}
        runtime_paths=sorted(run.glob('stage-runtime-source-*.json'))
        runtime=read(runtime_paths[-1]) if runtime_paths else {}
        owner=next(o for o in ownership if o['gpu']==r['gpu'])
        try:
            process=psutil.Process(owner['pid']);alive=process.is_running() and process.status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:alive=False
        checks=dict(entry_full_restore=audit.get('passed',False) and all(audit.get('actual_cpu_restore',{}).values()),
                    optimizer_associations=audit.get('optimizer_parameter_names_equal',False),
                    native_two_updates=two.get('passed',False),planning_interface=planning.get('passed',False),
                    planning_state_unchanged=bool(planning) and planning.get('model_state_before')==planning.get('model_state_after'),
                    actual_progress100=status.get('step',0)>=entry+100,
                    finite_loss='loss' in status and math.isfinite(status['loss']),
                    positive_finite_native_gradients=bool(gradients) and all(v is not None and math.isfinite(v) and v>0 for v in gradients.values()),
                    actual_GPU=runtime.get('CUDA_VISIBLE_DEVICES')==str(r['gpu']),
                    actual_source=bool(runtime) and all(digest(path)==runtime['sha256'][key] for key,path in runtime['paths'].items()),
                    parent_alive=alive,no_effective_failure=not (run/'worker-failed.json').exists())
        rows.append(dict(name=r['name'],run=str(run),gpu=r['gpu'],owner_pid=owner['pid'],step=status.get('step'),
                         state=status.get('state'),loss=status.get('loss'),native_gradient_step=periodic.get('step'),
                         gradients=gradients,runtime=runtime,checks=checks,passed=all(checks.values())))
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    report=dict(passed=len(rows)==8 and all(r['passed'] for r in rows),host=host,
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,gpu_state=gpu,
        scope='Single startup validation. Earlier8 EIO failures retained separately;positive gradients are not control success.')
    save(output,report)
    print(dict(passed=report['passed'],host=host,steps=[r['step'] for r in rows],
               failing=[(r['name'],[k for k,v in r['checks'].items() if not v]) for r in rows if not r['passed']],gpu=gpu),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--host',choices=['dev13','dev15'],required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args();main(Path(args.root),args.host,Path(args.output))
