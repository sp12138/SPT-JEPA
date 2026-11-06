"""Use released slots for frozen causal diagnostics,then a matched H15 test."""
import argparse
import json
from pathlib import Path
import time
from experiments.metaworld_contact_20261106.launch import environment,PYTHON,wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command
from experiments.metaworld_kernel_20261106.campaign import save


def work(a):
    plan=json.loads((a.root/'training-plan.json').read_text());r=next(x for x in plan['recipes']if x['host']==a.host and x['gpu']==a.gpu)
    source=Path(plan['source']);run=Path(r['run']);deadline=plan['deadline_unix'];env=environment(source,a.gpu)
    try:
        owner=Path(f'/proc/{r["resource_pid"]}/cmdline')
        while owner.exists()and b'goal_campaign.py'in owner.read_bytes():assert time.time()<deadline;time.sleep(10)
        wait_gpu_release(a.gpu)
        while not a.manifest.exists():assert time.time()<deadline;time.sleep(10)
        assert json.loads(a.manifest.read_text())['passed']and json.loads((a.root/'initial-contract.json').read_text())['passed']
        original=a.root.parent/'metaworld_query_recurrent_20261106/training'/f'{r["arm"]}-seed{r["seed"]}'
        proof=a.root/'original-model-branch-audits'/f'{r["arm"]}-seed{r["seed"]}.json'
        command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/branch_audit.py'),'--mode','audit','--manifest',str(a.manifest),
            '--run',str(original),'--step','2500','--output',str(proof)],proof.with_suffix('.log'),env,source,deadline)
        assert json.loads(proof.read_text())['passed'];wait_gpu_release(a.gpu)
        from experiments.metaworld_kernel_20261106.recurrent_campaign import work as train
        train(a.root,source,a.host,a.gpu)
    except BaseException as error:save(run/'execution-error.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--host',required=True);p.add_argument('--gpu',type=int,required=True);work(p.parse_args())
