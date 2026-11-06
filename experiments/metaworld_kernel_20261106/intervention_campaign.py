"""Finite, separately labelled own-intervention corpus collection."""
import argparse
import json
from pathlib import Path
import time

from experiments.metaworld_contact_20261106.launch import environment,PYTHON,wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command
from experiments.metaworld_kernel_20261106.campaign import save,digest


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=False);recipes=[]
    for host,offset,seeds in [('dev13',0,[1,3]),('dev15',1,[2,4])]:
        owners=json.loads((a.parent/f'launch-generator-v3-{host}.json').read_text())['workers']
        for index,(gpu,seed)in enumerate(zip([0,4],seeds)):
            owner=next(r for r in owners if r['gpu']==gpu)
            recipes.append(dict(host=host,gpu=gpu,seed=seed,tile=offset+2*index,tasks=list(range(offset+2*index,50,4)),
                policy=str(a.parent/'training'/f'point1-seed{seed}'),resource_pid=owner['pid'],output=str(a.root/f'tile{offset+2*index}')))
    save(a.root/'plan.json',dict(source=str(a.source),bank=str(a.bank),recipes=recipes,deadline_unix=time.time()+12*3600,
        manifest='/efs/prescottsun/SPT-JEPA/runs/metaworld_eval_20261103/cases-v2/manifest.json',
        scope='SEPARATE interventional-data track,0 learner updates.No expert controls.No benchmark states in training.'
              'All future comparisons must share the identical extra corpus and count collection cost.No public-setting/SOTA equivalence.'))


def work(a):
    plan=json.loads((a.root/'plan.json').read_text());r=next(x for x in plan['recipes']if x['host']==a.host and x['gpu']==a.gpu)
    source=Path(plan['source']);output=Path(r['output']);deadline=plan['deadline_unix']
    try:
        owner=Path(f'/proc/{r["resource_pid"]}/cmdline')
        while owner.exists()and b'goal_campaign.py'in owner.read_bytes():assert time.time()<deadline;time.sleep(10)
        assert json.loads((Path(r['policy'])/'complete.json').read_text())['passed'];wait_gpu_release(r['gpu'])
        assert json.loads((Path(plan['bank'])/'manifest.json').read_text())['no_actions_read']
        env=environment(source,r['gpu']);helper=source/'experiments/metaworld_kernel_20261106/intervention_data.py'
        common=[PYTHON,'-B','-u',str(helper),'--mode','collect','--bank',plan['bank'],'--policy',r['policy'],'--eval-manifest',plan['manifest']]
        # First verify real branch/reset/RGB invariance on the assigned GPU.
        pilot=a.root/f'pilot-tile{r["tile"]}'
        command(common+['--output',str(pilot),'--tasks',str(r['tasks'][0]),'--parents','1','--native-steps','40','--context-steps','40'],
            a.root/f'pilot-tile{r["tile"]}.log',env,source,deadline)
        proof=json.loads((pilot/'manifest.json').read_text());assert proof['passed']and proof['policy_unchanged']and len(proof['records'])==29
        # Pilot is retained separately,never silently duplicated in the corpus.
        command(common+['--output',str(output),'--tasks']+list(map(str,r['tasks'])),a.root/f'tile{r["tile"]}.log',env,source,deadline)
        result=json.loads((output/'manifest.json').read_text());assert result['passed']and result['policy_unchanged']
        save(a.root/f'complete-tile{r["tile"]}.json',dict(passed=True,branches=len(result['records']),manifest_sha256=digest(output/'manifest.json'),
            pilot_separate=True,learner_updates=0,extra_data_not_used_in_primary_models=True))
    except BaseException as error:save(a.root/f'error-tile{r["tile"]}.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True)
    for key in ['root','source','parent','bank']:p.add_argument('--'+key,type=Path)
    p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    else:work(a)
