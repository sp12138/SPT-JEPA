"""Completed-only three-task results; keep all requested tasks/arms/seeds."""
import argparse
import json
from pathlib import Path

import numpy as np

from common import read, sha, write


def interval(delta):
    delta=np.asarray(delta,dtype=float)
    ids=np.random.default_rng(20261101).integers(len(delta),size=(10000,len(delta)))
    return dict(mean_pp=float(delta.mean()),ci95_pp=np.quantile(delta[ids].mean(1),[.025,.975]).tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--tasks',nargs='+',default=['pusht','tworoom','cube']);p.add_argument('--step',type=int,choices=[600,1200,2000],default=600)
    a=p.parse_args();root=Path(a.root);protocol=read(root/'protocol.json');rows=[];diagnostics=[];bindings={}
    arms=['flat','field','tf'] if a.step==600 else ['flat','field']
    cases=['primary','fresh'] if a.step==2000 else ['primary']
    for task in a.tasks:
        for arm in arms:
            for seed in [1,2]:
                ident=f'{task}-{arm}-seed{seed}';run=root/ident
                terminal=root/'worker-records'/f'{ident}-{"gate" if a.step==600 else "maturity2000"}.json'
                if a.step==600 or a.step==2000:
                    t=read(terminal);assert t['passed'] and all(x['returncode']==0 for x in t['results'])
                    bindings[str(terminal)]=sha(terminal)
                audit_path=root/'audits'/f'{ident}-step{a.step}'/'audit.json';audit=read(audit_path);assert audit['passed']
                assert audit['state_before']==audit['state_after'] and audit['native_cpu_model_optimizer_scheduler_restore']
                archive=run/'full-archives'/f'step{a.step:07d}.ckpt';assert sha(archive)==audit['full_checkpoint_sha256']
                dpath=audit_path.with_name('state-audit.json');d=read(dpath)
                assert d['step']==a.step and len(d['cohort'])==64
                diagnostics.append(dict(task=task,arm=arm,seed=seed,step=a.step,
                    conditions=d['results']['conditions'],audit_sha256=sha(audit_path),diagnostic_sha256=sha(dpath)))
                for case in cases:
                    for mode in (['l2'] if a.step==600 else ['l2','ngc']):
                        suffix='' if case=='primary' else '-fresh'
                        out=(root/'evaluations'/ident) if a.step==600 else root/'graph-evaluations'/f'{ident}-step{a.step}-{mode}{suffix}'
                        metrics=read(out/'metrics.json');values=metrics['metrics']['episode_successes'];assert len(values)==50
                        assert all(isinstance(x,bool) for x in values)
                        if a.step!=600:
                            g=read(out/'graph-audit.json');assert g['passed'] and g['model_audit']['state_before']==g['state_after']
                        pair_key=metrics['pairs'] if 'pairs' in metrics else dict(episodes=metrics['episodes'],start_steps=metrics['start_steps'])
                        rows.append(dict(task=task,arm=arm,seed=seed,case_set=case,mode=mode,step=a.step,
                            success_percent=100*sum(values)/50,success_vector=values,pairs=pair_key,
                            metrics_sha256=sha(out/'metrics.json')))
    contrasts=[]
    for task in a.tasks:
        for case in cases:
            for mode in (['l2'] if a.step==600 else ['l2','ngc']):
                diffs=[]
                for seed in [1,2]:
                    find=lambda arm:next(x for x in rows if (x['task'],x['case_set'],x['mode'],x['seed'],x['arm'])==(task,case,mode,seed,arm))
                    flat,field=find('flat'),find('field');assert flat['pairs']==field['pairs']
                    delta=100*(np.asarray(field['success_vector'],float)-np.asarray(flat['success_vector'],float));diffs.append(delta)
                    contrasts.append(dict(task=task,case_set=case,mode=mode,seed=seed,**interval(delta)))
                contrasts.append(dict(task=task,case_set=case,mode=mode,seed='mean_of_two_fixed_models',**interval(np.mean(diffs,axis=0))))
    write(a.output,dict(passed=True,step=a.step,tasks=a.tasks,task_endpoints=len(rows),rows=rows,diagnostics=diagnostics,
          field_minus_flat=contrasts,terminal_bindings=bindings,
          scope='All requested tasks/arms/seeds complete;case CIs condition on two trained weights,exclude simulator repeats and seed-population uncertainty. Task protocols differ;not canonical CLEAR or a three-task success claim.'))
    print(json.dumps(dict(step=a.step,rows=[{k:x[k] for k in ['task','arm','seed','case_set','mode','success_percent']} for x in rows],contrasts=contrasts)))


if __name__=='__main__':main()
