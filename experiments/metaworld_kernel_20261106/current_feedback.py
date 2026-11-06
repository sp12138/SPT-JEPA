"""Full shared-case evaluation of persistent self-generated flow latent noise."""
import argparse
import json
from pathlib import Path
import time
from experiments.metaworld_contact_20261106.launch import environment,PYTHON,wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command
from experiments.metaworld_kernel_20261106.campaign import save,digest


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=False);recipes=[]
    for j,arm in enumerate(['episodes','native']):
        for seed in[1,2]:
            run=a.parent/'training'/f'{arm}-seed{seed}';cfg=json.loads((run/'config.json').read_text());assert cfg['model']['chunk']==1
            assert json.loads((run/'complete.json').read_text())['passed']
            recipes.append(dict(arm=arm+'-fixed',seed=seed,host='dev6',gpu=4+j*2+seed-1,run=str(run),
                output=str(a.root/f'{arm}-seed{seed}'),weights_sha256=digest(run/'model-step0040000/weights.pt')))
    save(a.root/'plan.json',dict(source=str(a.source),recipes=recipes,manifest='/efs/prescottsun/SPT-JEPA/runs/metaworld_eval_20261103/cases-v2/manifest.json',
        deadline_unix=time.time()+12*3600,new_model_updates=0,scope='FourfrozenCURRENT-onlyflowmodels;full20sharedphysicalcases/task.TheonlychangeisreuseofowninitialGaussianlatentateachfeedbackstep.'
            'Matchedfresh-latentcontrolscompletedinparentroot.No newteacher/actionreference/clock/history/taskrule.'))


def work(a):
    plan=json.loads((a.root/'plan.json').read_text());r=next(x for x in plan['recipes']if x['gpu']==a.gpu);out=Path(r['output']);out.mkdir();source=Path(plan['source'])
    try:
        wait_gpu_release(a.gpu);assert digest(Path(r['run'])/'model-step0040000/weights.pt')==r['weights_sha256'];env=environment(source,a.gpu)
        base=[PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/inverse_evaluate.py'),'--run',r['run'],'--step','40000','--execute','1','--noise-policy','fixed','--manifest',plan['manifest']]
        command(base+['--output',str(out/'entry20'),'--budget','20','--tasks','bin-picking-v3','--seeds','42'],out/'entry20.log',env,source,plan['deadline_unix'])
        command(base+['--output',str(out/'pilot'),'--seeds']+list(map(str,list(range(42,52))+list(range(1042,1052)))),out/'pilot.log',env,source,plan['deadline_unix'])
        result=json.loads((out/'pilot/summary.json').read_text());assert result['passed']and result['model_unchanged']and len(result['rows'])==160
        save(out/'complete.json',dict(passed=True,episodes=160,new_updates=0,current_only=True))
    except BaseException as error:save(out/'execution-error.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True)
    for k in['root','source','parent']:p.add_argument('--'+k,type=Path)
    p.add_argument('--gpu',type=int);a=p.parse_args();prepare(a)if a.mode=='prepare'else work(a)
