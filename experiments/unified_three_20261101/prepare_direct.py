"""One evidence-motivated direct-action variant; exactly6 two-step gates."""
import argparse
import copy
from pathlib import Path

from common import read, sha, source_check, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--root',required=True)
    p.add_argument('--source',required=True);p.add_argument('--cpu',required=True);a=p.parse_args()
    parent,root=Path(a.parent),Path(a.root);assert not root.exists()
    old=read(parent/'protocol.json');source=read(a.source);source_check(source);cpu=read(a.cpu)
    assert cpu['passed'] and cpu['no_CUDA_reseed_called'] and cpu['action_route_probe']['direct_single_output_perturbation_mse']>0
    assert len(cpu['actual_preflight_initials_exact'])==6
    runs=[]
    for original in old['runs']:
        if original['arm']!='field':continue
        row=copy.deepcopy(original);cfg=read(Path(row['run'])/'config.json')
        assert read(parent/'worker-records'/(row['id']+'-gate.json'))['passed']
        cfg=copy.deepcopy(cfg);cfg['output']=str(root/row['id']);cfg['model']['local_readout_condition']=True
        row.update(run=cfg['output'],gpu=2 if row['seed']==1 else 5,arm='field',
                   parent_indirect_config=str(parent/row['id']/'config.json'),parent_indirect_config_sha256=sha(parent/row['id']/'config.json'))
        path=Path(row['run'])/'config.json';write(path,cfg);row['config_sha256']=sha(path);runs.append(row)
    assert len(runs)==6
    for task in ['pusht','tworoom','cube']:
        write(root/'cohorts'/f'{task}.json',read(parent/'cohorts'/f'{task}.json'))
    write(root/'source-record.json',source)
    write(root/'protocol.json',dict(old,source=source,runs=runs,preflight_stop=2,short_gate_stop=600,
        parent=str(parent),cpu_contract_sha256=sha(a.cpu),
        note='One shared direct-action readout;same512hidden units,98304extra parameters versus indirect field,about0.2%trainable. No loss/data/sigma/beta/optimizer or task-specific changes.',
        review_required_before600=True,automatic_long_training=False))
    write(root/'candidate-basis.json',dict(
        evidence='Indirect field600 PushT action-swap MSE changes only0.5%/0.09% despite better factual error;native AdaLN modulation starts exactlyzero. Structural CPU witness verifies a direct condition exposes actions after one output perturbation.',
        limitation='This does not prove the optimization barrier causes poor control. Direct six-run600 test only after actual native technical review. Same method family,not another qualified original idea.'))
    print('Prepared6 direct-condition native2-update gates',root)


if __name__=='__main__':main()
