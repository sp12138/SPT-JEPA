"""Prepare32 bounded-controller evaluations +4 original replays; never launch."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
B=Path('/efs/prescottsun/SPT-JEPA')
PY='/efs/prescottsun/project_jepa/envs/lewm/bin/python'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def arg(job,name):return job['command'][job['command'].index('--'+name)+1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(B/'runs/action_projection6000_20261031'))
    p.add_argument('--helper-source',default=str(B/'code'));a=p.parse_args();root=Path(a.root)
    assert not (root/'plan.pending.json').exists();(root/'specs').mkdir(parents=True,exist_ok=True)
    helper=Path(a.helper_source)/'experiments/action_projection_20261031'
    helper_files={name:sha(helper/name)for name in ['adapter.py','evaluate.py']}
    cpu=read(B/'reports/action-projection-20261031/cpu-contracts.json');assert cpu['passed']
    assert cpu['helper_files'][str(B/'code/experiments/action_projection_20261031/adapter.py')]==helper_files['adapter.py']
    support=read(B/'reports/action-projection-20261031/cube-support.json');assert support['counts']['outside_coordinates']==0
    roots={'native-cal':B/'runs/native_lewm6000_20261031','TF64':B/'runs/carrier_learn6000_20261031',
           'F64':B/'runs/carrier_learn6000_20261031','local1024':B/'runs/local_isotropic6000_20261031'}
    plans={key:read(path/'plan.json')['jobs']for key,path in roots.items()}
    jobs=[];models=[];cache={};source_records={}
    def digest(path):
        path=str(path)
        if path not in cache:cache[path]=sha(path)
        return cache[path]
    for ti,task in enumerate(['tworoom','cube']):
        for fi,family in enumerate(roots):
            for seed in [1,2]:
                suffix={'native-cal':f'{task}-native-lewm-seed{seed}','TF64':f'{task}-temporal-forecast-seed{seed}',
                        'F64':f'{task}-carrier-identity-seed{seed}','local1024':f'{task}-spatial-local-seed{seed}'}[family]
                slot=fi*2+seed-1
                for case in ['primary','fresh']:
                    phase=('planning-calibrated'if case=='primary'else'fresh-planning-calibrated')if family=='native-cal'else('planning'if case=='primary'else'fresh-planning')
                    candidates=[j for j in plans[family]if j['id']==phase+'-'+suffix]
                    assert len(candidates)==1,(family,suffix,phase)
                    original=candidates[0];prior=Path(original['output'])/'metrics.json';metrics=read(prior)
                    exitrec=read(roots[family]/'exits'/(original['id']+'.json'));assert exitrec['returncode']==0
                    source=original['source'];smanifest=read(source['manifest'])
                    for name,d in smanifest['files'].items():assert digest(Path(source['source'])/name)==d
                    for name,d in read(source['runtime_manifest'])['files'].items():assert digest(Path(source['runtime_source'])/name)==d
                    source_records[family]=source
                    checkpoint=arg(original,'checkpoint');manifest=arg(original,'manifest');data=arg(original,'data')
                    assert metrics['checkpoint']==checkpoint and metrics['manifest_sha256']==digest(manifest)
                    assert metrics['pairs']==read(manifest)['pairs'] and len(metrics['pairs'])==50
                    assert metrics['normalization']=='train episodes' and metrics['checkpoint_audit']['strict_equal']
                    assert read(Path(checkpoint)/'config.json')['_target_'].startswith('as_jepa_lewm.')
                    native=Path(source['source'])/'as_jepa_tools/benchmark_evaluate.py'
                    paths=[prior,Path(checkpoint)/'weights.pt',Path(checkpoint)/'config.json',Path(data)/'dataset_card.json',Path(manifest),native,Path(source['manifest']),Path(source['runtime_manifest'])]
                    immutable={str(path):digest(path)for path in paths}
                    immutable.update(cpu['native_files']);immutable.update(support['source_files'])
                    modes=['bounded']+(['original']if seed==1 and case=='primary'and family in ['native-cal','local1024']else[])
                    for mode in modes:
                        name=f'{mode}-{case}-{task}-{family}-seed{seed}';out=root/name
                        spec=dict(id=name,task=task,family=family,training_seed=seed,step=6000,case_set=case,mode=mode,
                            source=source,checkpoint=checkpoint,manifest=manifest,data=data,output=str(out),native_evaluator=str(native),
                            reference_metrics=str(prior),reference_success=metrics['metrics']['success_rate'],
                            immutable_inputs=immutable,helper_files=helper_files)
                        specpath=root/'specs'/f'{name}.json';write(specpath,spec)
                        jobs.append(dict(id=name,phase='planning-bounded'if mode=='bounded'else'planning-original-replay',
                            host='dev15',gpu=slot,task=task,family=family,training_seed=seed,case_set=case,mode=mode,
                            source=source,output=str(out),spec=str(specpath),spec_sha256=sha(specpath),
                            command=[PY,'-B','-u',str(helper/'evaluate.py'),'--spec',str(specpath)],
                            environment={'MUJOCO_EGL_DEVICE_ID':str(slot)},depends_on=[]))
                    models.append(dict(task=task,family=family,seed=seed,case_set=case,checkpoint=checkpoint,
                        checkpoint_sha256=immutable[str(Path(checkpoint)/'weights.pt')],original_success=metrics['metrics']['success_rate'],original_metrics=str(prior),source_commit=source['source_commit']))
    assert len(jobs)==36 and sum(j['mode']=='bounded'for j in jobs)==32
    jobs.sort(key=lambda j:(j['gpu'],j['mode']!='original',j['task'],j['case_set']))
    # Preserve sequential execution per GPU and replay before that slot's treatments.
    for gpu in range(8):
        slotjobs=[j for j in jobs if j['gpu']==gpu]
        for prev,j in zip(slotjobs,slotjobs[1:]):j['depends_on']=[prev['id']]
    protocol=dict(prepared_only=True,root=str(root),jobs_count=36,bounded_evaluations=32,original_replays=4,
        source_records=source_records,helper_source=str(helper),helper_files=helper_files,models=models,
        cpu_contracts=dict(path=str(B/'reports/action-projection-20261031/cpu-contracts.json'),sha256=sha(B/'reports/action-projection-20261031/cpu-contracts.json')),
        cube_support=dict(path=str(B/'reports/action-projection-20261031/cube-support.json'),sha256=sha(B/'reports/action-projection-20261031/cube-support.json')),
        scope='One common Box-bounded controller for4families,2tasks,2seeds,primary/fresh50. Frozen6000 checkpoints; no learning/calibration/metric changes. Compare within-checkpoint original/projection paired cases, then compare all models under the same controller. Cube is an explicit bounded-controller change, not an execution-equivalent rewrite. Native calibrated is a train-only BN variant with existing extra exposure; F/local/TF/native capacities differ. Two seed means do not establish population stability or novelty.',
        treatment='P(z)=scaler.transform(clip(scaler.inverse_transform(z), actual single_action_space.low, high)); returned physical commands clip to the same bounds. CEM proposals/moments/warm-start retain native code; past command history follows actual bounded returned commands.',
        launch_block='Root must review CPU contracts, freeze helpers, repoint all36 commands to immutable source, verify actual GPU idle and explicit launch authorization. This script does not launch anything.')
    write(root/'protocol.pending.json',protocol);write(root/'plan.pending.json',dict(prepared_only=True,root=str(root),jobs=jobs))
    out=B/'reports/action-projection-20261031/preparation.json';out.write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(root=str(root),jobs=len(jobs),bounded=32,original_replays=4,original_results=models)),flush=True)

if __name__=='__main__':main()
