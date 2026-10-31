"""Prepare exactly48 held original/correlated evaluations. Never activate or launch."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
B=Path('/efs/prescottsun/SPT-JEPA');PY='/efs/prescottsun/project_jepa/envs/lewm/bin/python'
CAL=B/'reports/action-covariance-20261031/calibration'
MODELS={
 'tworoom':[('F64free','carrier-unconstrained',6000,'carrier_learn6000_20261031'),('native-raw','native-lewm',6000,'native_lewm6000_20261031'),('TF64','temporal-forecast',6000,'carrier_learn6000_20261031')],
 'cube':[('local1024','spatial-local',6000,'local_isotropic6000_20261031'),('native-raw','native-lewm',12000,'native_cube12000_20261031'),('TF64','temporal-forecast',12000,'cube_tf12000_20261031')]}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def arg(j,k):return j['command'][j['command'].index('--'+k)+1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--source-record');a=ap.parse_args()
    root=Path(a.root);assert not root.exists();(root/'specs').mkdir(parents=True)
    helper=read(a.source_record)if a.source_record else None
    helperdir=Path(helper['source'])/'experiments/action_covariance_20261031'if helper else Path(__file__).parent
    helpers={n:sha(helperdir/n)for n in ['adapter.py','evaluate.py']}
    cpu=read(B/'reports/action-covariance-20261031/cpu-contracts.json');assert cpu['passed']
    assert cpu['helper_files'][str(B/'code/experiments/action_covariance_20261031/adapter.py')]==helpers['adapter.py']
    slots=[(h,g)for h in ['dev2','dev13','dev15']for g in range(8)];jobs=[];refs=[];cache={};casehash={};checked=set()
    def digest(p):
        p=str(p)
        if p not in cache:cache[p]=sha(p)
        return cache[p]
    def source_check(s):
        if s['source_commit']in checked:return
        for mk in ['manifest','runtime_manifest']:
            m=read(s[mk])
            for name,h in m['files'].items():assert digest(Path(m['source'])/name)==h,name
        checked.add(s['source_commit'])
    if helper:source_check(helper)
    for task,models in MODELS.items():
        cal=read(CAL/task/'calibration.json');assert sha(CAL/task/'matrix.npz')==cal['matrix_sha256']and cal['zero_variance_coordinates']==0
        with np.load(CAL/task/'matrix.npz')as f:
            np.testing.assert_array_equal(np.linalg.cholesky(f['correlation']),f['cholesky'])
        dest=root/'calibration'/task;dest.mkdir(parents=True);shutil.copy2(CAL/task/'calibration.json',dest/'calibration.json');shutil.copy2(CAL/task/'matrix.npz',dest/'matrix.npz')
        for family,suffix,step,parent_name in models:
            for seed in [1,2]:
                name=f'{task}-{suffix}-seed{seed}'
                for case in ['primary','fresh']:
                    parent=B/'runs'/parent_name;phase='planning'if case=='primary'else'fresh-planning'
                    if task=='tworoom'and family=='native-raw'and case=='fresh':parent=B/'runs/native_rawfresh6000_20261031';phase='fresh-planning-raw'
                    plan=read(parent/'plan.json');matches=[j for j in plan['jobs']if j['id']==phase+'-'+name];assert len(matches)==1,(parent,phase,name)
                    old=matches[0];exitrec=read(parent/'exits'/(old['id']+'.json'));assert exitrec['returncode']==0
                    ref=Path(old['output'])/'metrics.json';metrics=read(ref);source=old['source'];source_check(source)
                    checkpoint=Path(arg(old,'checkpoint'));manifest=Path(arg(old,'manifest'));data=Path(arg(old,'data'))
                    assert metrics['checkpoint']==str(checkpoint)and metrics['manifest_sha256']==digest(manifest)
                    assert metrics['normalization']=='train episodes'and metrics['checkpoint_audit']['strict_equal']
                    cfg=read(checkpoint/'config.json');assert cfg['_target_'].startswith('as_jepa_lewm.')
                    assert len(metrics['pairs'])==50 and metrics['pairs']==read(manifest)['pairs']
                    key=(task,case)
                    if key not in casehash:casehash[key]=digest(manifest)
                    assert casehash[key]==digest(manifest)
                    card=read(data/'dataset_card.json');assert digest(data/'dataset_card.json')==cal['dataset_card_sha256']
                    assert card['action_mean']==cal['action_mean']and card['action_std']==cal['action_std']
                    native=Path(source['source'])/'as_jepa_tools/benchmark_evaluate.py'
                    # All selected exports use the evaluator's train-card scaler branch; runtime set_env rechecks actual arrays.
                    assert "startswith('as_jepa_lewm.')"in native.read_text().replace(chr(34),chr(39))
                    paths=[ref,checkpoint/'weights.pt',checkpoint/'config.json',manifest,data/'dataset_card.json',native,dest/'calibration.json',dest/'matrix.npz']
                    immutable={str(p):digest(p)for p in paths};host,gpu=slots[len(refs)]
                    previous=None
                    for mode in ['original','correlated']:
                        jid=f'{mode}-{case}-{task}-{family}-seed{seed}';out=root/jid
                        spec=dict(id=jid,task=task,family=family,step=step,training_seed=seed,case_set=case,mode=mode,source=source,
                            checkpoint=str(checkpoint),manifest=str(manifest),data=str(data),output=str(out),reference_metrics=str(ref),native_evaluator=str(native),
                            calibration=str(dest/'calibration.json'),helper_files=helpers,immutable_inputs=immutable)
                        sp=root/'specs'/f'{jid}.json';write(sp,spec)
                        jobs.append(dict(id=jid,host=host,gpu=gpu,hold=True,phase='action-correlation-controller',task=task,family=family,training_seed=seed,case_set=case,mode=mode,
                            source=helper,output=str(out),spec=str(sp),spec_sha256=sha(sp),depends_on=[]if previous is None else[previous],
                            command=[PY,'-B','-u',str(helperdir/'evaluate.py')if helper else'<FROZEN_HELPER_REQUIRED>/evaluate.py','--spec',str(sp)],environment={'MUJOCO_EGL_DEVICE_ID':str(gpu)}));previous=jid
                    refs.append(dict(task=task,family=family,step=step,seed=seed,case_set=case,checkpoint=str(checkpoint),source_commit=source['source_commit'],
                        reference_metrics=str(ref),parameters=metrics['checkpoint_audit']['parameters'],checkpoint_sha256=digest(checkpoint/'weights.pt'),manifest_sha256=digest(manifest),slot=[host,gpu],
                        actual_policy_scaler_assertion='train-card arrays checked before first actual cost call'))
    assert len(jobs)==48 and len(refs)==24 and len(casehash)==4
    environment=read(B/'runs/carrier_learn6000_20261031/plan.json')['environment']
    write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment=environment,jobs=jobs))
    write(root/'protocol.json',dict(prepared_only=True,all_held=True,source=helper,models=refs,jobs=48,checkpoints=12,cases_per_job=50,modes=['original','correlated'],
        allocation='All24 slots dev2/dev13/dev15 g0-7, exactly original then correlated on each same slot; actual-free verification required before activation',
        budget='300 samples x30 iterations/top30; horizon5,receding5,actionblock5,history3,50 raw steps. No extra model calls, no new simulator cases beyond48x50 evaluations.',
        hypothesis='Common frozen training-action correlation coordinates, not a new learning algorithm. TR F64free and Cube local are different candidate families. Capacities/exposures differ; compare within checkpoint/controller first.',
        fitting='Original train128 cohorts only; Cholesky of unit-diagonal automatic LedoitWolf covariance, no mean injection/clip/floor/tuning; all empirical variances nonzero.',
        replay='Original native path preserved; record actual history replay discrepancies without expecting outcome-bitwise reproducibility, and never select best historical/replay baseline.',
        contracts=str(B/'reports/action-covariance-20261031/cpu-contracts.json'),contracts_sha256=digest(B/'reports/action-covariance-20261031/cpu-contracts.json'),
        complete_only='Read all48 completed results before outcome comparison. Failure HOLD; no fallback or parameter changes. No automatic follow-up.'))
    (root/'HOLD_DISPATCH').write_text('All48 candidates held. Original holds preserved; explicit activation after freeze and actual availability checks.\n')
    print(json.dumps(dict(root=str(root),endpoints=len(jobs),original=24,correlated=24,all_held=True,source_commits=sorted(checked))))
if __name__=='__main__':main()
