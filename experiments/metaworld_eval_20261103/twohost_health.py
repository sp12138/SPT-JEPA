"""One startup snapshot of owned processes and native gradients; no monitor."""
import argparse
import datetime
import math
from pathlib import Path
import subprocess

import psutil
from experiments.metaworld_eval_20261103.worker import digest,read,save


def main(root,host,output):
    assert host in ['dev13','dev15']
    activation=read(root/f'activation-{host}.json')
    diagnostic=read(root/f'diagnostic-activation-v2-{host}.json')
    source=root.parent.parent/'reports/metaworld_eval_20261103/runtime-60c887e'
    rows=[]
    for row in activation['jobs']:
        if row['mode']!='train':continue
        run=Path(row['run']);status=read(run/'status.json');audit=read(run/'audit-step2472.json')
        runtime=read(run/'stage-runtime-source-0008240.json');config=read(run/'config.json')
        process=psutil.Process(status['pid']);environment=process.environ()
        gradients={name:status.get('grad_norm_'+name) for name in config['audit_expectations']['gradients']}
        checks=dict(entry_restore=audit['passed'] and all(audit['actual_cpu_restore'].values()),
                    progressed100=status['step']>=2572 and status['state']=='running',
                    finite_loss=math.isfinite(status['loss']),
                    positive_gradients=all(value is not None and math.isfinite(value) and value>0 for value in gradients.values()),
                    parent_alive=psutil.Process(row['pid']).is_running(),
                    actual_native_command=any(x.endswith('/safe_train_stage.py') for x in process.cmdline())
                                          and '--resume' in process.cmdline() and str(run/'config.json') in process.cmdline(),
                    actual_gpu=environment['CUDA_VISIBLE_DEVICES']==str(row['gpu']),
                    actual_cwd=process.cwd()==str(source),entry_step=runtime['step']==2473,
                    source_hashes=all(digest(path)==runtime['sha256'][key] for key,path in runtime['paths'].items()),
                    native_restored='Restored all states' in (run/'train-stage8240.log').read_text(),
                    no_failure=not (run/'worker-failed.json').exists())
        rows.append(dict(run=str(run),gpu=row['gpu'],pid=process.pid,step=status['step'],loss=status['loss'],
                         gradients=gradients,checks=checks,passed=all(checks.values()),
                         entry_checkpoint_sha256=audit['checkpoint_sha256'],runtime=runtime))
    controllers=[]
    for row in diagnostic['jobs']:
        outer=psutil.Process(row['pid']);children=outer.children()
        controllers.append(dict(gpu=row['gpu'],launcher=row['pid'],alive=outer.is_running(),
                           children=[dict(pid=p.pid,command=p.cmdline()) for p in children]))
    jobs=read(root/'diagnostic-plan.json')['jobs'];diagnostics=[]
    for job in jobs:
        path=Path(job['work']);log=path/'diagnose.log'
        completed_tasks=sum(line.startswith('{"task_id"') for line in log.read_text().splitlines()) if log.exists() else 0
        diagnostics.append(dict(name=job['name'],complete=(path/'complete.json').exists(),
                            failed=(path/'failed.json').exists(),claimed=(path/'claim').exists(),completed_tasks=completed_tasks))
    gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits'],text=True)
    report=dict(passed=len(rows)==2 and all(r['passed'] for r in rows) and all(r['alive'] for r in controllers),
                host=host,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                training=rows,evaluation_controllers=controllers,diagnostics=diagnostics,gpu_state=gpu_state,
                scope='Single startup check;no recurring monitor,model update,retry,extension or performance claim.')
    save(output,report)
    print(dict(passed=report['passed'],host=host,steps=[r['step'] for r in rows],
               failing=[(r['run'],[k for k,v in r['checks'].items() if not v]) for r in rows if not r['passed']],
               diagnostic_failures=sum(r['failed'] for r in diagnostics),gpu_state=gpu_state),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--host',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    main(Path(args.root),args.host,Path(args.output))
