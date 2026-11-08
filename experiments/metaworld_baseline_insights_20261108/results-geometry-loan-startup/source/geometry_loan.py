"""Independent frozen encoder-geometry cost test;no privileged fitted heads."""
import argparse
import concurrent.futures
from pathlib import Path

import support_loan as loan
from experiments.metaworld_baselines_20261107.campaign import BASE,PYTHON,env,read,save,sha,now

ROOT=BASE/'runs/metaworld_geometry_cost_20261108'
loan.ROOT=ROOT


def prepare(helper,evaluator):
    import numpy as np
    ROOT.mkdir(exist_ok=False)
    old=read(BASE/'runs/metaworld_control_support_20261108/plan.json');jobs=[]
    for label in ['technical','formal']:
        for i in [0,1]:
            save(ROOT/f'manifests/{label}-{i}.json',read(BASE/f'runs/metaworld_control_support_20261108/manifests/{label}-{i}.json'))
    for original in old['jobs']:
        j=dict(original);metric=BASE/f'runs/metaworld_goal_geometry_20261108/{j["name"]}/encoder_geometry_metric.npz'
        proof=read(metric.parent/'summary.json');assert proof['passed'] and proof['frozen_weights_sha256']==j['weights_sha256']
        with np.load(metric) as f:
            assert f.files==['linear'];matrix=f['linear'];assert matrix.ndim==2 and np.isfinite(matrix).all()
        j.update(metric=str(metric),metric_sha256=sha(metric),metric_coefficients=int(matrix.size))
        jobs.append(j)
    save(ROOT/'plan.json',dict(helper=str(helper),evaluator=str(evaluator),evaluator_sha256=sha(evaluator),jobs=jobs,
        formal_episodes=24,technical_episodes=8,unique_physical_cases=12,max_seconds=7200,deadline=old['deadline'],created_utc=now(),
        scope='Independentcostintervention,2frozen100epochmodels x12samehardcases. Original100CEMcontrolsinheritedafteractual20stepstandardparitygates. '
            'OnlyGoalMSEcomparespredictedandgoallatentdifferencesafterafrozenlinearmapfittoSAMEencoderbackbonefeatures. '
            'No state/task/reward/action/pixel/strongteacher targets;notprivilegedstateprobes. 0worldmodelSGD. '
            'NativeNNparentshavebatchregs,soNOTentirelysamplelevelcandidate. Notnovel/SOTA/fresh/full50claim. '
            'CEM300x15/H6/execute3/context2/f5/500nativeanysuccessunchanged;sameNNrolloutcount,extraBmatrixcostdisclosed. '
            'Borrowsecondmetadatapaironlyafteritsowncurrentchildfinishes;exactparentresumes. Supportpilotindependent/unchanged.'))


def pilot(runner,host,pair):
    plan=read(ROOT/'plan.json');helper=Path(plan['helper']);assert sha(plan['evaluator'])==plan['evaluator_sha256'];results=[]
    for j in plan['jobs']:
        assert sha(j['metric'])==j['metric_sha256']
        for arm,label,budget in [('standard','technical',20),('geometry','technical',20),('geometry','formal',500)]:
            def one(i,gpu):
                out=ROOT/j['name']/arm/f'{label}-{i}'
                cmd=[PYTHON,plan['evaluator'],'--checkpoint',j['checkpoint'],'--data',j['data'],
                    '--manifest',str(ROOT/f'manifests/{label}-{i}.json'),'--output',str(out),'--budget',str(budget)]
                if arm=='geometry':cmd+=['--geometry',j['metric']]
                runner.run(cmd,out.parent/f'{label}-{i}.log',env(helper,[gpu]),helper)
                report=read(out/'metrics.json');loan.verify(report,j,arm,label)
                if arm=='geometry':assert report['goal_geometry_metric_sha256']==j['metric_sha256']
                return report
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(one,i,2*pair+i) for i in [0,1]]
                reports=[f.result() for f in futures]
            if label=='formal':
                rows=[r for d in reports for r in d['episodes']]
                results.append(dict(name=j['name'],episodes=12,geometry_successes=sum(r['success'] for r in rows),
                    reference_cem_successes=sum(r['success'] for r in j['reference_rows']),
                    successes=[dict(task=r['task'],seed=r['seed']) for r in rows if r['success']],metric_coefficients=j['metric_coefficients']))
                save(ROOT/'progress.json',dict(models=results,utc=now()))
    target=next(r for r in results if r['name']=='smwm-seed1')
    save(ROOT/'summary.json',dict(passed=True,models=results,formal_episodes=24,technical_episodes=8,
        unique_physical_cases=12,worldmodel_updates=0,utc=now(),eligible_replication=target['geometry_successes']>=3,
        scope=plan['scope']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','selftest','work'],required=True)
    p.add_argument('--helper',type=Path);p.add_argument('--evaluator',type=Path);p.add_argument('--host');p.add_argument('--pair',type=int);p.add_argument('--parent',type=int)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.helper,a.evaluator)
    elif a.mode=='selftest':loan.selftest()
    else:loan.work(a.host,a.pair,a.parent,pilot_fn=pilot)
