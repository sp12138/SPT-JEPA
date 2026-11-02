"""Inventory all existing prescribed reference exports,without reading outcomes."""
import argparse
from pathlib import Path

from common import B,read,write,sha,source_check


def main(root,helper):
    root=Path(root);assert not root.exists();root.mkdir()
    report=B/'reports/lingjing_fullspec_20261102';inv=read(report/'reference-inventory-v1.json')
    source=read(B/'reports/branch-repair-20261102/source-record-v1.json');source_check(source)
    helper=read(helper);groups={};seen={}
    for row in inv['rows']:
        if row['family']=='L0':continue
        if row['seed'] not in (range(9) if row['family']=='original' else range(1,9)):continue
        if row['family'] in ['TN','FO'] and row['seed'] not in [1,2]:continue
        for step in [24982,37473]:
            record=row['portable'].get(str(step))
            if not record:continue
            cp=Path(record['path']);cfg=record['config']
            if cfg.get('mode')!='L2' or cfg.get('sigma0',.1)!=.1 or cfg.get('window',8)!=8:continue
            if cfg.get('domain','standard')!='standard':continue
            if any(cfg.get(k,v)!=v for k,v in dict(renderer_width=384,renderer_depth=12,renderer_heads=6,state_slots=4,state_width=16,carry=True).items()):continue
            if str(cp) not in seen:seen[str(cp)]=sha(cp/'weights.pt')
            digest=seen[str(cp)]
            key=(row['family'],row['seed'],step,digest)
            groups.setdefault(key,[]).append(dict(checkpoint=str(cp),source_config=row['config_path'],config_sha256=sha(cp/'config.json')))
    jobs=[];references=[]
    for index,(key,aliases) in enumerate(sorted(groups.items())):
        family,seed,step,digest=key;aliases.sort(key=lambda x:x['checkpoint']);cp=aliases[0]['checkpoint']
        reference=dict(family=family,seed=seed,step=step,weights_sha256=digest,checkpoint=cp,aliases=aliases)
        references.append(reference)
        for fp32 in [True,False]:
            ident=f'{family}-s{seed}-n{step}-{digest[:8]}-{"fp32" if fp32 else "bf16"}'
            jobs.append(dict(reference,id=ident,fp32=fp32,host=['dev2','dev13','dev15'][len(jobs)%3]))
    assert references
    protocol=dict(source=source,helper=helper,jobs=jobs,references=references,data=str(B/'data/pusht-lance-full-v1/phase0'),
                  cases=128,window=16,context=3,model_updates=0,simulation_calls=0,
                  scope='All existing requested2/3epoch exports;identical weight files deduplicated,all aliases preserved.No outcome selection.')
    write(root/'protocol.json',protocol)
    print(dict(references=len(references),jobs=len(jobs),host_counts={h:sum(j['host']==h for j in jobs) for h in ['dev2','dev13','dev15']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--helper',required=True);a=p.parse_args();main(a.root,a.helper)
