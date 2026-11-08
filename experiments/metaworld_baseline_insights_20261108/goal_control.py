"""Finite goal-rank head control test,reusing audited single-GPU machinery.

Published T2G control runs separately with exactly the same models and cases.
No training happens here. Only an eligible native metadata worker is borrowed;
all its current children finish naturally and the exact worker then resumes.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import time
import traceback

BASE=Path('/efs/prescottsun/SPT-JEPA')
ROOT=BASE/'runs/metaworld_goal_rank_control_20261108'
PUBLIC_CONTROL=BASE/'runs/metaworld_t2g_control_20261108'
SOURCE=BASE/'reports/metaworld_baseline_insights_20261108/t2g-pilot-v1'
sys.path.insert(0,str(SOURCE))
import t2g_pilot as controller
from experiments.metaworld_baselines_20261107.campaign import read,save,sha,now


def prepare():
    ROOT.mkdir(exist_ok=False);plan=read(PUBLIC_CONTROL/'plan.json');jobs=[]
    pair=read(BASE/'runs/metaworld_goal_rank_20261108/PAIRED_HEAD_PROOF.json');assert pair['passed']
    for name in ['formal','technical']:
        save(ROOT/f'manifests/{name}.json',read(PUBLIC_CONTROL/f'manifests/{name}.json'))
    for old in plan['jobs']:
        j=dict(old);metric=BASE/f'runs/metaworld_goal_rank_20261108/goal/{j["name"]}/metric-state.pt'
        proof=read(metric.with_name('summary.json'))
        assert proof['passed'] and proof['objective_direction']=='goal' and proof['reference_goal_initial_tensors_exact']
        assert proof['feature_proof']['weights_sha256']==j['weights_sha256'] and proof['metric_state_sha256']==sha(metric)
        j.update(metric=str(metric),metric_sha256=sha(metric),metric_training_summary_sha256=sha(metric.with_name('summary.json')),
            public_reference_metric=old['metric'],public_reference_metric_sha256=old['metric_sha256'])
        jobs.append(j)
    plan.update(jobs=jobs,created_utc=now(),max_seconds=10800,public_reference_control_root=str(PUBLIC_CONTROL),
        wrapper_sha256=sha(__file__),underlying_controller_sha256=sha(SOURCE/'t2g_pilot.py'),
        scope='Future-goalRGB-rankobjectivealignmentablation,NOTpublishedT2G/novelalgorithm/wholesamplelevelcandidate. '
            'SamepublicTemporalMetricarchitecture/exactinitialheadtensors/same1085optimizersteps/features/config/samplerRNGasreference;ONLYRGB-timeorientationofrankingchanges. '
            'Nativebodyfrozen100epochs/hasbatchregs. No action/state/reward/tasktargets/strongteacher/externalweights. '
            '32formal+8technicalepisodes/16sharedphysicalcases(12hard+4positive),2samefrozenmodelsaspublicT2Gcontrol. '
            'OnlyterminalcostchangesfromMSE;CEM300x15/H6/execute3/context2/f5/500anysuccessunchanged. '
            'Actualstandard20stepaction/success/firstbankSHAparitygatesandfrozenworld/headproofsrequired. '
            'PublicT2Gcontrolrunsindependently;no duplicateformalstandard/publishedheadcases. '
            'All18scalingclaimsmustexist,targetpaircompletebeforeborrowingONE nativeMETADATAworker. '
            'Itscurrentnativechildfinishesnaturally;other23GPUjobsunchanged. No SGD/SGDrepeat/oldchildkill/budgetextension.')
    save(ROOT/'plan.json',plan)


def audit():
    import torch
    torch.set_num_threads(8);assert not torch.cuda.is_initialized()
    spec=importlib.util.spec_from_file_location('goal_cost_binding',SOURCE/'evaluate_t2g.py')
    evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
    from stable_worldmodel.planning import GoalMSE
    checks=[]
    for j in read(ROOT/'plan.json')['jobs']:
        before_rng=torch.get_rng_state().clone();metric,proof=evaluator.load_temporal_metric(j['metric'],'cpu')
        assert torch.equal(before_rng,torch.get_rng_state()) and proof['objective_direction']=='goal'
        pred=torch.randn(2,7,8,metric.dim);goal=torch.randn(2,1,metric.dim);before=evaluator.model_sha(metric)
        cost=evaluator.make_temporal_cost(GoalMSE,metric)(dict(predicted_emb=pred,goal_emb=goal))
        assert cost.shape==(2,7) and torch.equal(cost,metric(pred[:,:,-1],goal[:,None,-1])) and torch.isfinite(cost).all()
        assert before==evaluator.model_sha(metric)
        checks.append(dict(name=j['name'],metric_sha256=sha(j['metric']),terminal_cost_equation_exact=True,
            metric_unchanged=True,cpu_rng_preserved_on_load=True))
    assert not torch.cuda.is_initialized()
    save(ROOT/'SOURCE_AUDIT.json',dict(passed=True,checks=checks,wrapper_sha256=sha(__file__),
        evaluator_sha256=sha(SOURCE/'evaluate_t2g.py'),underlying_controller_sha256=sha(SOURCE/'t2g_pilot.py'),
        worldmodel_updates=0,head_updates=0,physics_episodes=0,gpu_contexts=0))


def work(host,gpu,parent):
    plan=read(ROOT/'plan.json');assert sha(__file__)==plan['wrapper_sha256'] and read(ROOT/'SOURCE_AUDIT.json')['passed']
    assert sha(SOURCE/'t2g_pilot.py')==plan['underlying_controller_sha256']
    # Wait only for the current scaling pair's completion. No process is paused
    # during this short wait;the ordinary baseline work retains priority.
    deadline=time.time()+300
    scaling=BASE/'runs/metaworld_baseline_scaling_20261108'
    while True:
        ready=True
        for j in read(scaling/'plan.json')['jobs']:
            owner=read(scaling/'claims'/f'{j["name"]}.json')
            if owner['host']==host and owner['pair']==gpu//2:
                if (Path(j['output'])/'error.json').exists():raise RuntimeError('Upstream scaling error;do not borrow')
                ready=ready and (Path(j['output'])/'complete.json').exists()
        if ready:break
        if time.time()>deadline:raise TimeoutError('Scaling pair still unfinished;no metadata paused')
        time.sleep(.2)
    controller.ROOT=ROOT
    controller.work(host,gpu,parent)
    if (ROOT/'summary.json').exists():
        d=read(ROOT/'summary.json');d.update(metric_training_direction='future-goal',public_reference_control_root=str(PUBLIC_CONTROL),
            provisional_gain_over_raw_cem=d['eligible_replication'],eligible_replication=False,
            promotion='No automatic long promotion;need matched public-T2G control and independent confirmation.')
        for m in d['models']:
            for group in m['groups'].values():group['goal_rank_successes']=group.pop('t2g_successes')
        save(ROOT/'summary.json',d)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','audit','work'],required=True)
    p.add_argument('--host');p.add_argument('--gpu',type=int);p.add_argument('--parent',type=int);a=p.parse_args()
    try:
        if a.mode=='prepare':prepare()
        elif a.mode=='audit':audit()
        else:work(a.host,a.gpu,a.parent)
    except BaseException:
        save(ROOT/'WRAPPER_ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise
