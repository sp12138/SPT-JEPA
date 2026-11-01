"""Review complete preflights and actual paired initialization before training."""
import argparse
from pathlib import Path
import torch

from common import read,write,sha,source_check


def main(root):
    root=Path(root);protocol=read(root/'protocol.json');source_check(protocol['source'])
    torch.set_num_threads(2)
    records={};pairs=[]
    for row in protocol['runs']:
        ident=row['id'];cfg=Path(row['run'])/'config.json';assert sha(cfg)==row['config_sha256']
        terminal=root/'worker-records'/f'{ident}-preflight.json';status=read(terminal)
        audit=root/'audits'/f'{ident}-step2/audit.json';a=read(audit)
        assert a['passed'] and a['native_cpu_model_optimizer_scheduler_restore'] and a['step']==2
        assert a['config_sha256']==row['config_sha256'] and a['state_before']==a['state_after']
        assert sha(Path(row['run'])/'full-archives/step0000002.ckpt')==a['full_checkpoint_sha256']
        record=dict(terminal_sha256=sha(terminal),audit_sha256=sha(audit))
        if not status['passed']:
            repair=root/'repairs'/f'{ident}-preflight.json';r=read(repair)
            assert r['passed'] and r['original_failure_sha256']==sha(terminal)
            record['repair_sha256']=sha(repair)
        records[ident]=record
    def compare(left,right,excluded=(),shared=False):
        rows={r['id']:r for r in protocol['runs']}
        a=torch.load(Path(rows[left]['run'])/'initial-model/weights.pt',map_location='cpu',weights_only=True,mmap=True)
        b=torch.load(Path(rows[right]['run'])/'initial-model/weights.pt',map_location='cpu',weights_only=True,mmap=True)
        keys=(set(a)&set(b)) if shared else set(a)
        if not shared:assert set(a)==set(b)
        keys=sorted(k for k in keys if not any(k.startswith(x) for x in excluded))
        assert keys and all(torch.equal(a[k],b[k]) for k in keys),(left,right)
        pairs.append(dict(left=left,right=right,exact_tensors=len(keys),excluded=list(excluded),common_keys_only=shared))
    for task in ['pusht','tworoom','cube']:
        for seed in [1,2]:compare(f'query-{task}-function-seed{seed}',f'query-{task}-forecast-seed{seed}',shared=True)
    for seed in [1,2]:
        compare(f'decoder-tworoom-window-seed{seed}',f'decoder-tworoom-pooled-seed{seed}')
        for scale in ['original','calibrated']:
            compare(f'round2-tworoom-{scale}-near-seed{seed}',f'round2-tworoom-{scale}-past-seed{seed}')
        for anchor in ['near','past']:
            compare(f'round2-tworoom-original-{anchor}-seed{seed}',f'round2-tworoom-calibrated-{anchor}-seed{seed}',excluded=('pool.head.',))
    report=dict(passed=True,run_ids=list(records),preflights=records,paired_initializations=pairs,
                source=protocol['source'],protocol_sha256=sha(root/'protocol.json'),stop_step=2000,
                original_failures_preserved=True,automatic_long_extension=False,
                calibration_rng_scope='Pool calibration consumes initialization-loader RNG;not common full training RNG across scale arms.')
    write(root/'gate-approval.json',report)
    print(dict(passed=True,runs=len(records),paired_initializations=len(pairs),cap=2000))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
