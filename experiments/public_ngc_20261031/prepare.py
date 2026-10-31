"""Prepare12 frozen graph builds and48 held objective-only evaluations."""
import argparse
from pathlib import Path
import numpy as np
from common import *

MODELS={
 'tworoom':[('local1024','spatial-local',6000,'local_isotropic6000_20261031',1024),
            ('native-raw','native-lewm',6000,'native_lewm6000_20261031',192),
            ('TF64','temporal-forecast',6000,'carrier_learn6000_20261031',64)],
 'cube':[('local1024','spatial-local',6000,'local_isotropic6000_20261031',1024),
         ('native-raw','native-lewm',12000,'native_cube12000_20261031',192),
         ('TF64','temporal-forecast',12000,'cube_tf12000_20261031',64)]}
SLOTS=[('dev2',g)for g in range(3,8)]+[('dev13',g)for g in range(3,8)]+[('dev15',g)for g in range(2,8)]


def argument(job,name):return job['command'][job['command'].index('--'+name)+1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--calibration-root',required=True)
    p.add_argument('--source-record');a=p.parse_args();root,cal=Path(a.root),Path(a.calibration_root)
    assert root!=cal and not root.exists()and not cal.exists()
    helper=read(a.source_record)if a.source_record else None
    here=Path(__file__).parent;helperdir=Path(helper['source'])/'experiments/public_ngc_20261031'if helper else here
    hashes={f:sha(helperdir/f)for f in FILES};checked=set();cache={}
    if helper:source_check(helper);assert all(sha(here/f)==h for f,h in hashes.items())
    def digest(p):
        p=str(p)
        if p not in cache:cache[p]=sha(p)
        return cache[p]
    def check_source(s):
        if s['source_commit']not in checked:source_check(s);checked.add(s['source_commit'])
    contracts=read(here/'cpu-contracts.json');assert contracts['passed']
    assert all(contracts['files'][f]==hashes[f]for f in ['common.py','objective.py','vendor/cost.py','vendor/graph.py'])
    root.mkdir(parents=True);cal.mkdir(parents=True);caljobs=[];evaljobs=[];models=[];casehash={};previous={}
    for task,entries in MODELS.items():
        data=B/'data/hardtasks-v1'/f'{task}-metadata'
        arrays={f:np.load(data/(f+'.npy'))for f in ['episode_ids','offsets','lengths','train_episodes','validation_episodes','test_episodes']}
        assert len(arrays['train_episodes'])==9000
        nodes=select_rows(arrays['episode_ids'],arrays['offsets'],arrays['lengths'],arrays['train_episodes'],np.r_[arrays['validation_episodes'],arrays['test_episodes']])
        cohort=cal/f'nodes-{task}.json';write(cohort,dict(count=NODES,seed=SEED,split='train',frameskip=5,window=8,
            dataset_card_sha256=digest(data/'dataset_card.json'),metadata_sha256={f+'.npy':digest(data/(f+'.npy'))for f in arrays},
            sampling='Uniform distinct train episode, uniform valid8-frame window start, uniform frame within window; sorted row order.',nodes=nodes))
        verify_cohort(cohort,data)
        for family,suffix,step,parent_name,dimension in entries:
            for seed in [1,2]:
                model_id=f'{task}-{family}-seed{seed}';name=f'{task}-{suffix}-seed{seed}';primary=None;refs=[]
                for case in ['primary','fresh']:
                    parent=B/'runs'/parent_name;phase='planning'if case=='primary'else'fresh-planning'
                    if task=='tworoom'and family=='native-raw'and case=='fresh':parent=B/'runs/native_rawfresh6000_20261031';phase='fresh-planning-raw'
                    matches=[j for j in read(parent/'plan.json')['jobs']if j['id']==phase+'-'+name];assert len(matches)==1,(parent,phase,name)
                    old=matches[0];assert read(parent/'exits'/(old['id']+'.json'))['returncode']==0
                    source=old['source'];check_source(source);cp=Path(argument(old,'checkpoint'));manifest=Path(argument(old,'manifest'))
                    assert Path(argument(old,'data'))==data;ref=Path(old['output'])/'metrics.json';r=read(ref);cfg=read(cp/'config.json')
                    assert r['checkpoint']==str(cp)and r['manifest_sha256']==digest(manifest)and r['normalization']=='train episodes'
                    assert r['pairs']==read(manifest)['pairs']and len(r['pairs'])==50 and r['checkpoint_audit']['strict_equal']
                    assert cfg['_target_'].startswith('as_jepa_lewm.')and cp.name==f'model-step{step:07d}'
                    if family=='local1024':assert cfg['spatial_mode']=='local'and cfg.get('spatial_basis','identity')=='identity'and cfg.get('spatial_chart','none')=='none'
                    key=(task,case)
                    if key in casehash:assert casehash[key]==digest(manifest)
                    casehash[key]=digest(manifest)
                    native=Path(source['source'])/'as_jepa_tools/benchmark_evaluate.py'
                    immutable={str(f):digest(f)for f in [cp/'weights.pt',cp/'config.json',manifest,data/'dataset_card.json',ref,native,cohort]}
                    base=dict(model_id=model_id,task=task,family=family,step=step,seed=seed,dimension=dimension,model_source=source,
                        helper_source=helper,helper_files=hashes,checkpoint=str(cp),data=str(data),cohort=str(cohort),immutable_inputs=immutable,
                        native_evaluator=str(native),manifest=str(manifest),reference_metrics=str(ref),graph_gate=str(cal/'graph-gate.json'))
                    if primary is None:primary=base
                    else:assert base['checkpoint']==primary['checkpoint']and base['model_source']==primary['model_source']
                    for mode in ['l2','ngc']:
                        jid=f'{mode}-{case}-{model_id}';s=dict(base,id=jid,case_set=case,mode=mode,output=str(root/jid))
                        sp=root/'specs'/f'{jid}.json';write(sp,s);host,gpu=SLOTS[len(evaljobs)%len(SLOTS)];slot=(host,gpu)
                        deps=[]if slot not in previous else[previous[slot]];previous[slot]=jid
                        evaljobs.append(dict(id=jid,host=host,gpu=gpu,hold=True,source=helper,output=s['output'],spec=str(sp),spec_sha256=sha(sp),
                            depends_on=deps,requires=[str(cal/'graph-gate.json')],environment={'MUJOCO_EGL_DEVICE_ID':str(gpu)},
                            command=[PY,'-B','-u',str(helperdir/'evaluate.py')if helper else'<FROZEN_HELPER_REQUIRED>','--spec',str(sp)]))
                    refs.append(dict(case_set=case,path=str(ref),manifest_sha256=digest(manifest)))
                s=dict(primary,id=model_id,output=str(cal/model_id));sp=cal/'specs'/f'{model_id}.json';write(sp,s)
                host,gpu=SLOTS[len(caljobs)];caljobs.append(dict(id=model_id,host=host,gpu=gpu,hold=True,source=helper,output=s['output'],spec=str(sp),spec_sha256=sha(sp),
                    command=[PY,'-B','-u',str(helperdir/'build.py')if helper else'<FROZEN_HELPER_REQUIRED>','--spec',str(sp)]))
                models.append(dict(id=model_id,task=task,family=family,seed=seed,step=step,dimension=dimension,checkpoint=s['checkpoint'],
                    parameters=r['checkpoint_audit']['parameters'],weights_sha256=digest(cp/'weights.pt'),source_commit=source['source_commit'],references=refs))
    assert len(caljobs)==12 and len(evaljobs)==48 and len(casehash)==4
    environment=read(B/'runs/carrier_learn6000_20261031/plan.json')['environment']
    for target,jobs in [(cal,caljobs),(root,evaljobs)]:
        write(target/'candidate-plan.json',dict(root=str(target),deadline=None,environment=environment,jobs=jobs))
        (target/'HOLD_DISPATCH').write_text('All candidate jobs held. Preserve this pre-activation record; separate reviewed activation only.\n')
    write(root/'protocol.json',dict(models=models,slots=SLOTS,calibration_root=str(cal),helper_source=helper,
        graph=dict(nodes=5000,neighbors=15,anchors=4,rule='mutual',bridging='Unmodified public minimum inter-component edges',node_seed=SEED),
        totals=dict(graph_jobs=12,images=60000,encoder_calls=1884,eval_jobs=48,cases_per_eval=50),
        method='Public NGC terminal-cost replacement only. Local1024 on BOTH tasks; raw native, no BN recalibration. No correlation/projection action intervention.',
        budgets='Original300x30/top30,H5,frame_skip5,history3,receding5,50primitive steps; primary/fresh manifests unchanged.',
        failures='Any nonfinite/collapse/radius/provenance failure holds its chain; no floor, fallback, parameter search or graph resampling.',
        activation='Graph launch only after freeze/actual-free check. All12 graph integrity reports required; report gate to root before any48eval activation.',
        outcome_review='Retain all contemporaneous L2 replays; no partial winner review or historical best selection. All48 complete before aggregate.',
        node_distribution_adaptation='Our common episode-level9000train split and8-frame training population, not public clip-random split/4-frame sampling.',
        contracts=contracts))
    print(json.dumps(dict(graph_jobs=12,eval_jobs=48,all_held=True,calibration_root=str(cal),evaluation_root=str(root))))


if __name__=='__main__':main()
