"""Frozen future-chunk baselines;never candidates,teachers,or initializations."""
import argparse
import json
from pathlib import Path
import time
from experiments.metaworld_contact_20261106.launch import environment,PYTHON,wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command
from experiments.metaworld_kernel_20261106.campaign import save,digest


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=False);recipes=[]
    for seed in[1,2]:
        for index,noise_policy in enumerate(['fresh','fixed']):
            run=a.parent/'bridge'/f'local-flow-simple-seed{seed}';cfg=json.loads((run/'config.json').read_text())
            assert cfg['model']['chunk']==6 and not cfg['model']['temporal_aux']
            for k,execute in enumerate([1,3]):
                gpu=(seed-1)*4+index*2+k
                recipes.append(dict(host='dev6',gpu=gpu,seed=seed,arm=f'flow-execute{execute}-{noise_policy}',execute=execute,noise_policy=noise_policy,run=str(run),
                    output=str(a.root/f'flow-execute{execute}-{noise_policy}-seed{seed}'),weights_sha256=digest(run/'model-step0040000/weights.pt')))
    save(a.root/'plan.json',dict(source=str(a.source),manifest='/efs/prescottsun/SPT-JEPA/runs/metaworld_eval_20261103/cases-v2/manifest.json',
        recipes=recipes,deadline_unix=time.time()+6*3600,new_updates=0,scope='2OLDNNsx2feedbackintervalsx2ownlatentnoisepolicies;futurechunk6supervisionDISCLOSED.'
              'Readonlycomparatorsonly.Nocandidateancestry,distillation,ortrainingdata.Nofuture demonstrated commandsat inference.'))


def work(a):
    plan=json.loads((a.root/'plan.json').read_text());r=next(x for x in plan['recipes']if x['gpu']==a.gpu);source=Path(plan['source']);out=Path(r['output']);out.mkdir()
    try:
        wait_gpu_release(a.gpu);env=environment(source,a.gpu);assert digest(Path(r['run'])/'model-step0040000/weights.pt')==r['weights_sha256']
        base=[PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/inverse_evaluate.py'),'--run',r['run'],'--step','40000',
            '--execute',str(r['execute']),'--noise-policy',r['noise_policy'],'--manifest',plan['manifest']]
        command(base+['--output',str(out/'entry20'),'--budget','20','--tasks','bin-picking-v3','--seeds','42'],out/'entry20.log',env,source,plan['deadline_unix'])
        command(base+['--output',str(out/'pilot')],out/'pilot.log',env,source,plan['deadline_unix'])
        proof=json.loads((out/'pilot/summary.json').read_text());assert proof['passed']and proof['model_unchanged']
        save(out/'complete.json',dict(passed=True,episodes=len(proof['rows']),new_updates=0,comparator_only=True))
    except BaseException as error:save(out/'execution-error.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True)
    for key in['root','source','parent']:p.add_argument('--'+key,type=Path)
    p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    else:work(a)
