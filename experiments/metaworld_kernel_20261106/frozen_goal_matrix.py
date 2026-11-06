"""Reuse completed goal-model slots for the same full50-task frozen comparison."""
import argparse
import json
from pathlib import Path
import time
from experiments.metaworld_contact_20261106.launch import environment,PYTHON,wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command
from experiments.metaworld_kernel_20261106.campaign import save,digest


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=False);(a.root/'results').mkdir();parent=json.loads((a.parent/'training-plan.json').read_text());cases=json.loads((a.cases/'plan.json').read_text());recipes=[]
    for r in parent['recipes']:
        recipes.append(dict(arm=r['arm'],seed=r['seed'],host=r['host'],gpu=r['gpu'],run=r['run'],output=str(a.root/f'{r["arm"]}-seed{r["seed"]}')))
    save(a.root/'plan.json',dict(source=str(a.source),recipes=recipes,manifest=cases['manifest'],tasks=cases['tasks'],case_seeds=cases['case_seeds'],deadline_unix=time.time()+24*3600,new_model_updates=0,
        scope='All12completed40kgoal-cells,frozenonSAME200NEWphysicalcases/50tasksasoriginal40kcontrols.2trainedseeds/cell,4physicalcases/task.'
              'Originaldataonly,oneCURRENT5native20Dloss.SameRGBgoal/sensor5/500anysuccess/ownfreshnoise/12flowsteps.No publicMT50/SOTAclaim.'
              'Checkpointselectionand50-taskscreenareNOTuntouchedconfirmation;newbroadercasesrequiredforfinalperformanceclaim.'))


def work(a):
    plan=json.loads((a.root/'plan.json').read_text());r=next(r for r in plan['recipes']if r['host']==a.host and r['gpu']==a.gpu);source=Path(plan['source']);out=Path(r['output']);out.mkdir();run=Path(r['run'])
    try:
        while not(run/'complete.json').exists():
            assert not(run/'execution-error.json').exists();assert time.time()<plan['deadline_unix'];time.sleep(5)
        assert json.loads((run/'audit-40000.json').read_text())['passed'];wait_gpu_release(a.gpu);env=environment(source,a.gpu)
        base=[PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/inverse_evaluate.py'),'--run',r['run'],'--step','40000','--manifest',plan['manifest'],'--execute','1']
        command(base+['--output',str(out/'entry20'),'--budget','20','--tasks','bin-picking-v3','--seeds',str(plan['case_seeds'][0])],out/'entry20.log',env,source,plan['deadline_unix'])
        command(base+['--output',str(out/'pilot'),'--seeds']+list(map(str,plan['case_seeds']))+['--tasks']+plan['tasks'],out/'pilot.log',env,source,plan['deadline_unix'])
        value=json.loads((out/'pilot/summary.json').read_text());assert value['passed']and value['model_unchanged']and len(value['rows'])==200
        save(out/'complete.json',dict(passed=True,new_model_updates=0,episodes=200,original_data_only=True,weights_sha256=digest(run/'model-step0040000/weights.pt')))
    except BaseException as e:save(out/'execution-error.json',dict(error=repr(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True)
    for k in['root','source','parent','cases']:p.add_argument('--'+k,type=Path)
    p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args();prepare(a)if a.mode=='prepare'else work(a)
