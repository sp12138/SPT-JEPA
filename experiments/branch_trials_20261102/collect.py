"""Collect the entire bounded stage; never choose from partially finished arms."""
import argparse
from pathlib import Path
import numpy as np

from common import read,write,sha,source_check


def interval(values):
    values=np.asarray(values,float);rng=np.random.default_rng(716)
    boot=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
    return dict(mean_pp=float(values.mean()),conditional_case_ci95_pp=np.quantile(boot,[.025,.975]).tolist())


def main(root,output=None):
    root=Path(root);protocol=read(root/'protocol.json');source_check(protocol['source'])
    for host in protocol['host_families']:
        final=read(root/'dispatchers'/f'{host}-gate-finished.json');assert final['passed']
    rows=[];audits=[]
    for run in protocol['runs']:
        ident=run['id'];terminal=read(root/'worker-records'/f'{ident}-gate.json')
        assert terminal['passed'] and terminal['step']==2000
        assert [r['returncode'] for r in terminal['results']]==[0,0,0,0]
        audit_path=root/'audits'/f'{ident}-step2000/audit.json';audit=read(audit_path)
        assert audit['passed'] and audit['step']==2000 and audit['state_before']==audit['state_after']
        assert audit['native_cpu_model_optimizer_scheduler_restore'] and audit['optimizer_steps_added_by_audit']==0
        assert audit['config_sha256']==run['config_sha256']==sha(Path(run['run'])/'config.json')
        assert audit['full_checkpoint_sha256']==sha(Path(run['run'])/'full-archives/step0002000.ckpt')
        assert audit['portable_sha256']==sha(Path(run['run'])/'model-step0002000/weights.pt')
        audits.append(dict(id=ident,sha256=sha(audit_path),diagnostics=read(audit_path.parent/'diagnostics.json')))
        for case in ['primary','fresh']:
            out=root/'evaluations'/f'{ident}-step2000-{case}';mp=out/'metrics.json';report=read(mp)
            values=report['metrics']['episode_successes'];assert len(values)==50 and set(values)<=set([0,1,False,True])
            if run['task']=='pusht':
                pairs=list(map(list,zip(report['episodes'],report['start_steps'])))
                assert report['seed']==(42 if case=='primary' else 1042) and report['split']=='heldout'
            else:
                pairs=[[p['episode_id'],p['start_step']] for p in report['pairs']]
                manifest=protocol['manifests'][run['task']][case]
                assert report['manifest_sha256']==sha(manifest)==protocol['manifest_sha256'][manifest]
            if run['family']=='query' and run['arm']=='function':
                proof=read(out/'query-objective-proof.json')
                assert proof['objective']=='GoalQueryObjective' and proof['finite'] and proof['cost_shape'][-1]==300
                assert proof['source_sha256']==sha(Path(protocol['source']['source'])/'as_jepa_lewm/goal_query.py')
            rows.append(dict(id=ident,task=run['task'],family=run['family'],arm=run['arm'],seed=run['seed'],case_set=case,
                             success_percent=2*sum(values),success_vector=values,pairs=pairs,metrics_sha256=sha(mp)))
    assert len(rows)==56 and len(audits)==28
    for task in ['pusht','tworoom','cube']:
        for case in ['primary','fresh']:
            selected=[r for r in rows if r['task']==task and r['case_set']==case]
            assert all(r['pairs']==selected[0]['pairs'] for r in selected)
    contrasts=[]
    def contrast(name,task,case,family,terms):
        deltas=[]
        for seed in [1,2]:
            delta=np.zeros(50)
            for arm,weight in terms:
                row=next(r for r in rows if (r['task'],r['case_set'],r['family'],r['seed'],r['arm'])==(task,case,family,seed,arm))
                delta+=100*weight*np.asarray(row['success_vector'],float)
            deltas.append(delta)
            contrasts.append(dict(name=name,task=task,case_set=case,seed=seed,**interval(delta)))
        contrasts.append(dict(name=name,task=task,case_set=case,seed='mean_of_two_fixed_models',**interval(np.mean(deltas,0))))
    for case in ['primary','fresh']:
        for task in ['pusht','tworoom','cube']:
            contrast('query_minus_direct_forecast',task,case,'query',[('function',1),('forecast',-1)])
        contrast('window_minus_pooled_batch_control','tworoom',case,'decoder',[('window',1),('pooled',-1)])
        contrast('scale_main_effect','tworoom',case,'round2',[(s+'-'+a,.5 if s=='calibrated' else -.5) for s in ['calibrated','original'] for a in ['near','past']])
        contrast('anchor_main_effect','tworoom',case,'round2',[(s+'-'+a,.5 if a=='past' else -.5) for s in ['calibrated','original'] for a in ['near','past']])
        contrast('scale_anchor_interaction','tworoom',case,'round2',[(s+'-'+a,1 if (s=='calibrated')==(a=='past') else -1) for s in ['calibrated','original'] for a in ['near','past']])
    out=Path(output) if output else root/'results';out.mkdir(exist_ok=False,parents=True)
    result=dict(passed=True,complete_runs=28,task_endpoints=56,stop_step=2000,source=protocol['source'],
                protocol_sha256=sha(root/'protocol.json'),rows=rows,audits=audits,contrasts=contrasts,
                limitations=['Bounded2000 screen;not convergence,novelty or three-task success.',
                    'CIs condition on two fixed trained models;exclude seed-population uncertainty,simulator repeats and multiplicity.',
                    'Query references add training images/encoder exposure;not matched train FLOPs versus direct forecast.',
                    'Pooledlogdet is a batch-coupled diagnostic;sample-level candidate is logdet_window.',
                    'Calibration changes pool initialization and initialization-loader RNG;not identical full training RNG across scale arms.',
                    'Training losses and forecast MSE in different learned coordinates are not directly comparable across families.',
                    'Custom heldout cases,pinned nativeCEM;not canonical CLEAR or complete uploaded specs.'])
    write(out/'summary.json',result)
    print(dict(passed=True,runs=28,task_endpoints=56,result=str(out/'summary.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output');a=p.parse_args();main(a.root,a.output)
