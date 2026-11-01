"""Complete matched repair results; preserve original negative evidence."""
import argparse
from pathlib import Path
import numpy as np

from common import read,write,sha,source_check
from experiments.branch_trials_20261102.collect import interval


def main(root):
    root=Path(root);p=read(root/'protocol.json');source_check(p['source'])
    parent_path=Path(p['parent'])/'results/summary.json';assert sha(parent_path)==p['parent_summary_sha256'];parent=read(parent_path)
    for host in p['host_families']:assert read(root/'dispatchers'/f'{host}-gate-finished.json')['passed']
    rows=[];audits=[]
    for run in p['runs']:
        ident=run['id'];step=run['stop_step'];r=Path(run['run']);terminal=read(root/'worker-records'/f'{ident}-gate.json')
        assert terminal['passed'] and terminal['step']==step and [x['returncode'] for x in terminal['results']]==[0,0,0,0]
        path=root/'audits'/f'{ident}-step{step}/audit.json';audit=read(path)
        assert audit['passed'] and audit['native_cpu_model_optimizer_scheduler_restore'] and audit['state_before']==audit['state_after']
        assert audit['config_sha256']==run['config_sha256']==sha(r/'config.json')
        assert audit['full_checkpoint_sha256']==sha(r/'full-archives'/f'step{step:07d}.ckpt')
        assert audit['portable_sha256']==sha(r/f'model-step{step:07d}/weights.pt')
        audits.append(dict(id=ident,audit_sha256=sha(path),diagnostics=read(path.parent/'diagnostics.json')))
        for case in ['primary','fresh']:
            out=root/'evaluations'/f'{ident}-step{step}-{case}';mp=out/'metrics.json';report=read(mp)
            vector=report['metrics']['episode_successes'];assert len(vector)==50 and set(vector)<=set([0,1,False,True])
            if run['task']=='pusht':
                pairs=list(map(list,zip(report['episodes'],report['start_steps'])))
                assert report['seed']==(42 if case=='primary' else 1042) and report['split']=='heldout'
            else:
                pairs=[[v['episode_id'],v['start_step']] for v in report['pairs']]
                manifest=p['manifests'][run['task']][case];assert report['manifest_sha256']==sha(manifest)==p['manifest_sha256'][manifest]
            reference=next(x for x in parent['rows'] if x['id']==run['parent_id'] and x['case_set']==case)
            assert pairs==reference['pairs']
            if run['family']=='query' and run['arm']!='forecast':
                proof=read(out/'query-objective-proof.json')
                assert proof['objective']=='GoalQueryObjective' and proof['finite'] and proof['cost_shape'][-1]==300
                assert proof['source_sha256']==sha(Path(p['source']['source'])/'as_jepa_lewm/goal_query.py')
            rows.append(dict(id=ident,task=run['task'],family=run['family'],arm=run['arm'],seed=run['seed'],step=step,
                             case_set=case,success_percent=2*sum(vector),success_vector=vector,pairs=pairs,metrics_sha256=sha(mp)))
    assert len(rows)==68 and len(audits)==34
    contrasts=[]
    def compare(name,task,family,case,terms):
        deltas=[]
        for seed in [1,2]:
            delta=np.zeros(50)
            for era,arm,weight in terms:
                data=rows if era=='new' else parent['rows']
                row=next(x for x in data if (x['task'],x['family'],x['arm'],x['seed'],x['case_set'])==(task,family,arm,seed,case))
                delta+=100*weight*np.asarray(row['success_vector'],float)
            deltas.append(delta);contrasts.append(dict(name=name,task=task,family=family,case_set=case,seed=seed,**interval(delta)))
        contrasts.append(dict(name=name,task=task,family=family,case_set=case,seed='mean_of_two_fixed_models',**interval(np.mean(deltas,0))))
    for case in ['primary','fresh']:
        for task in ['pusht','tworoom','cube']:
            compare('episode_minus_global',task,'query',case,[('new','episode',1),('new','global',-1)])
            for arm in ['global','episode']:
                compare(arm+'_minus_forecast4k',task,'query',case,[('new',arm,1),('new','forecast',-1)])
            compare('global_4k_minus_2k',task,'query',case,[('new','global',1),('old','function',-1)])
            compare('forecast_4k_minus_2k',task,'query',case,[('new','forecast',1),('old','forecast',-1)])
            compare('fp32_pool_window_minus_original',task,'decoder',case,[('new','window',1),('old','window',-1)])
        compare('fp32_pool_pooled_control_minus_original','tworoom','decoder',case,[('new','pooled',1),('old','pooled',-1)])
        for arm in ['original-near','original-past','calibrated-near','calibrated-past']:
            compare('fp32_pool_'+arm+'_minus_original','tworoom','round2',case,[('new',arm,1),('old',arm,-1)])
    out=root/'results';out.mkdir(exist_ok=False)
    write(out/'summary.json',dict(passed=True,complete_runs=34,task_endpoints=68,source=p['source'],helper=p['helper'],
          parent_summary_sha256=p['parent_summary_sha256'],rows=rows,original_rows=parent['rows'],audits=audits,contrasts=contrasts,
          limitations=['Query continuation intervention2000→4000,both reference arms share first2k global training;not a fresh episode-trained algorithm.',
            'Precision repair fresh2000 versus original2000 changes state/noise rounding and may change random draws;not common full training RNG.',
            'Both seeds/case sets preserved.Case bootstrap conditions on these two trained models;no simulator-repeat/seed-population/multiplicity coverage.',
            'Query versus forecast has additional reference-image/encoder exposure;global versus episode has equal image count.',
            'Cases were already examined for research selection,not untouched confirmatory cases.No optimal-action/physical/success training labels.',
            'Pooled decoder control is batch-coupled;window and query candidates are sample-level.',
            'Latent MSE across separately learned states is not directly comparable.No novelty/convergence/broad superiority follows.']))
    print(dict(passed=True,runs=34,task_endpoints=68,result=str(out/'summary.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args();main(args.root)
