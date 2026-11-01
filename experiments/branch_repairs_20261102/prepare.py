"""One uniform precision repair plus matched query-reference continuations."""
import argparse
import copy
import os
from pathlib import Path
import shutil

from common import B,read,write,sha,source_check


def main(root,source,helper):
    root=Path(root);assert not root.exists();source=read(source);helper=read(helper);source_check(source)
    parent=B/'runs/branch_trials_20261102';old=read(parent/'protocol.json');summary=read(parent/'results/summary.json')
    assert summary['passed'] and (parent/'HOLD').exists()
    assert read(B/'reports/branch-repair-20261102/reference-contracts-v1.json')['passed']
    root.mkdir();runs=[]
    def add(row,arm,model=None,reference=None):
        cfg=copy.deepcopy(read(Path(row['run'])/'config.json'));query=row['family']=='query'
        ident=f'{row["family"]}-{row["task"]}-{arm}-seed{row["seed"]}';run=root/ident
        start=2000 if query else 0;preflight=start+2;stop=4000 if query else 2000
        cfg.update(output=str(run),retain_steps=[preflight,stop],checkpoint_every=1000,record_runtime_source=True)
        if model:cfg['model'].update(model)
        if reference:cfg['goal_query_reference_mode']=reference
        write(run/'config.json',cfg)
        binding=None
        if query:
            proof=read(parent/'audits'/f'{row["id"]}-step2000/audit.json');assert proof['passed']
            src=Path(row['run'])/'full-archives/step0002000.ckpt';assert sha(src)==proof['full_checkpoint_sha256']
            os.link(src,run/'last.ckpt')
            (run/'full-archives').mkdir();os.link(src,run/'full-archives/step0002000.ckpt')
            (run/'initial-model').mkdir()
            os.link(Path(row['run'])/'initial-model/weights.pt',run/'initial-model/weights.pt')
            shutil.copy2(Path(row['run'])/'initial-model/config.json',run/'initial-model/config.json')
            binding=dict(parent_checkpoint=str(src),parent_sha256=proof['full_checkpoint_sha256'],copied_sha256=sha(run/'last.ckpt'))
            assert binding['parent_sha256']==binding['copied_sha256']
            write(run/'copy-binding.json',binding)
        runs.append(dict(row,id=ident,arm=arm,run=str(run),parent_id=row['id'],parent_run=row['run'],
                         config_sha256=sha(run/'config.json'),start_step=start,preflight_step=preflight,stop_step=stop,copy_binding=binding))
    # Reference intervention and unchanged controls start from identical2k full states.
    for row in [r for r in old['runs'] if r['family']=='query']:
        if row['arm']=='function':
            add(row,'global',reference='global');add(row,'episode',reference='episode')
        else:add(row,'forecast')
    # Same repair across every uploaded cell,no outcome-selected hyperparameters.
    for row in [r for r in old['runs'] if r['family']!='query']:
        add(row,row['arm'],model=dict(fp32_pool=True))
    assert len(runs)==34
    for task in ['pusht','tworoom','cube']:write(root/'cohorts'/f'{task}.json',read(parent/'cohorts'/f'{task}.json'))
    env=dict(old['environment']);env['TMPDIR']='/tmp/spt-jepa-branch-repairs-20261102'
    protocol=dict(source=source,helper=helper,parent=str(parent),parent_summary_sha256=sha(parent/'results/summary.json'),
                  runs=runs,host_families=old['host_families'],environment=env,manifests=old['manifests'],manifest_sha256=old['manifest_sha256'],
                  task_endpoints=68,continuous_monitor=False,automatic_long_training=False,
                  scope='16 uniform fp32_pool fresh2000 repairs versus original2000.18 query/forecast full-state continuations2000→4000;'
                        'global/episode fork the exact same query state;forecast is matched4k baseline.'
                        'No new architecture/optimal-action/physical/success labels.Independent episode references add no images over global.'
                        'Mixed-precision repair changes state/noise rounding and can change RNG draws;not an identical full training RNG claim.'
                        'Query first2k were global for both reference arms;this is a continuation intervention,not a fresh-episode algorithm result.'
                        'Case sets were already inspected for research selection;not new unseen confirmatory cases.')
    write(root/'protocol.json',protocol)
    print(dict(runs=len(runs),host_counts={h:sum(x['host']==h for x in runs) for h in protocol['host_families']},task_endpoints=68))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--helper',required=True);a=p.parse_args();main(a.root,a.source,a.helper)
