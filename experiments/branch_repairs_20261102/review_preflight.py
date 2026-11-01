"""Require all technical endpoints,original/fork initial states and no failures."""
import argparse
from pathlib import Path
import torch

from common import read,write,sha,source_check


def main(root):
    root=Path(root);p=read(root/'protocol.json');source_check(p['source']);assert read(root/'copy-restores.json')['passed']
    rows=[];torch.set_num_threads(2)
    for row in p['runs']:
        ident=row['id'];step=row['preflight_step'];terminal=root/'worker-records'/f'{ident}-preflight.json';record=read(terminal)
        assert record['passed'] and [x['returncode'] for x in record['results']]==[0,0,0]
        ap=root/'audits'/f'{ident}-step{step}/audit.json';a=read(ap)
        assert a['passed'] and a['step']==step and a['native_cpu_model_optimizer_scheduler_restore'] and a['state_before']==a['state_after']
        assert a['config_sha256']==row['config_sha256']==sha(Path(row['run'])/'config.json')
        left=torch.load(Path(row['run'])/'initial-model/weights.pt',map_location='cpu',weights_only=True,mmap=True)
        right=torch.load(Path(row['parent_run'])/'initial-model/weights.pt',map_location='cpu',weights_only=True,mmap=True)
        assert set(left)==set(right) and all(torch.equal(left[k],right[k]) for k in left),ident
        if row['start_step']:
            log=(root/'logs'/f'{ident}-preflight-train.log').read_text();assert 'Restored all states' in log
        rows.append(dict(id=ident,preflight_step=step,audit_sha256=sha(ap),terminal_sha256=sha(terminal),initial_tensors_exact=len(left)))
    assert len(rows)==34
    write(root/'gate-approval.json',dict(passed=True,run_ids=[r['id'] for r in rows],rows=rows,source=p['source'],helper=p['helper'],
          protocol_sha256=sha(root/'protocol.json'),automatic_extension=False))
    print(dict(passed=True,runs=len(rows),native_copied_restores=18,initial_pairs_exact=34))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
