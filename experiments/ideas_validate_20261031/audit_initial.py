"""Read-only exact initialization pairing, including all baseline buffers."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--seed',type=int)
    p.add_argument('--expect-count',required=True,type=int);args=p.parse_args()
    root=Path(args.root);protocol=read(args.protocol)
    records=[r for r in protocol['runs'] if args.seed is None or r['seed']==args.seed]
    assert len(records)==args.expect_count
    torch.set_num_threads(2);references={};paired={};rows=[]
    for r in records:
        ref=Path(r['original_control']);name=r['id']
        if str(ref) not in references:
            references[str(ref)]=torch.load(ref/'initial-model/weights.pt',map_location='cpu',weights_only=True)
        original=references[str(ref)]
        weights=torch.load(root/name/'initial-model/weights.pt',map_location='cpu',weights_only=True)
        missing=sorted(set(original)-set(weights))
        changed=[k for k in original if k in weights and not torch.equal(original[k],weights[k])]
        extra=sorted(set(weights)-set(original));assert len(original)==550 and not missing and not changed,(name,missing,changed)
        group=(r['task'],r['seed'],r['family'])
        if group not in paired:
            paired[group]={k:v.clone()for k,v in weights.items()}
        else:
            assert set(weights)==set(paired[group])
            assert all(torch.equal(v,paired[group][k])for k,v in weights.items()),group
        size=read(root/name/'model_size.json');old_size=read(ref/'model_size.json')
        rows.append(dict(id=name,task=r['task'],seed=r['seed'],family=r['family'],arm=r['arm'],
            reference=str(ref),common_tensors=len(original),all_original_tensors_equal=True,
            extra_tensors=extra,extra_state_elements=sum(weights[k].numel()for k in extra),
            extra_parameters=size['total']-old_size['total'],parameter_counts=size,passed=True))
        print(name,'550 original tensors equal',flush=True)
        del weights
    for task,seed,family in paired:
        assert len([r for r in rows if(r['task'],r['seed'],r['family'])==(task,seed,family)])==3
    result=dict(passed=True,runs=rows,paired_family_tensors_equal=True,
        paired_groups=[dict(task=t,seed=s,family=f,arms=3,all_initial_tensors_equal=True)for t,s,f in paired],
        source_commit=protocol['source_commit'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='All550reference state tensors, including non-parameter buffers. Extra buffer elements are not counted as learned parameters. Complete initial state_dict equal across three arms within each task/seed/family; trainable flags intentionally differ for identity carrier.')
    (root/'initial-tensor-identity.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
