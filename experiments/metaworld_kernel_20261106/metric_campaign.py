"""Eight matched frozen original-data model/geometry controls on dev6."""
import argparse
import json
from pathlib import Path
import time
from experiments.metaworld_contact_20261106.launch import environment,PYTHON,wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command,TASKS
from experiments.metaworld_kernel_20261106.campaign import save,digest


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=False);recipes=[]
    for seed in[1,2]:
        run=a.parent/'training'/f'rollout-seed{seed}';assert json.loads((run/'audit-2500.json').read_text())['passed']
        for index,(metric,prior)in enumerate([('rgb','increment'),('rgb','uniform'),('reachable','increment'),('reachable','uniform')]):
            gpu=(seed-1)*4+index;output=a.root/f'{metric}-{prior}-seed{seed}'
            recipes.append(dict(host='dev6',gpu=gpu,seed=seed,run=str(run),output=str(output),metric=metric,prior=prior,
                weights_sha256=digest(run/'model-step0002500/weights.pt')))
    old=json.loads((a.parent/'training-plan.json').read_text());save(a.root/'plan.json',dict(source=str(a.source),manifest=old['manifest'],recipes=recipes,
        deadline_unix=time.time()+6*3600,updates=0,scope='Originaldatamodelsfrozenat2500;2trainedparents x2metrics x2controlpriors.Not8independentpretrainings.'
              'Same8tasks/2physicalcases/CEM32x4/H15/own3x5controlwords/replan5/sensor5/500nativeanysuccess.'
              'ReachablemetrichasextraSVDalgebra(costmeasured),sameNNrolloutbudget.Noextra trainingdata,oracle,actionheadorcommandlibrary.'))


def work(a):
    plan=json.loads((a.root/'plan.json').read_text());r=next(x for x in plan['recipes']if x['gpu']==a.gpu);source=Path(plan['source']);output=Path(r['output'])
    output.mkdir();deadline=plan['deadline_unix'];env=environment(source,a.gpu);wait_gpu_release(a.gpu)
    try:
        assert json.loads((a.root/'initial-contract.json').read_text())['passed']
        run=Path(r['run']);assert digest(run/'model-step0002500/weights.pt')==r['weights_sha256']
        common=[PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/native_evaluate.py'),'--run',str(run),'--step','2500','--manifest',plan['manifest'],
            '--planner','path','--replan-native','5','--path-metric',r['metric'],'--control-prior',r['prior']]
        command(common+['--output',str(output/'entry20'),'--path-particles','128','--path-iterations','8','--budget','20','--tasks','bin-picking-v3','--seeds','42'],output/'entry20.log',env,source,deadline)
        assert json.loads((output/'entry20/summary.json').read_text())['passed']
        command(common+['--output',str(output/'pilot'),'--path-particles','32','--path-iterations','4','--budget','500','--tasks']+TASKS+['--seeds','42','1042'],output/'pilot.log',env,source,deadline)
        result=json.loads((output/'pilot/summary.json').read_text());assert result['passed']and result['model_unchanged']
        save(output/'complete.json',dict(passed=True,episodes=len(result['rows']),new_updates=0,model_unchanged=True))
    except BaseException as error:save(output/'execution-error.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True)
    for key in['root','source','parent']:p.add_argument('--'+key,type=Path)
    p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    else:work(a)
