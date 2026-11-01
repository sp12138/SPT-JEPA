"""Real-data episode/split and backward-compatibility contracts for references."""
import argparse
import importlib.util
from pathlib import Path
import torch

from common import B,dataset,read,write
from as_jepa_tools.goal_query_data import QueryReferenceWindows


def main(old_source,output):
    spec=importlib.util.spec_from_file_location('original_query_reference',Path(old_source)/'as_jepa_tools/goal_query_data.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);rows=[]
    for task in ['pusht','tworoom','cube']:
        cfg=read(B/'runs/branch_trials_20261102'/f'query-{task}-function-seed1/config.json')
        for split in ['train','validation']:
            base=dataset(dict(cfg,goal_query_reference=False),split)
            old=module.QueryReferenceWindows(base,1);global_reference=QueryReferenceWindows(base,1)
            episode=QueryReferenceWindows(base,1,mode='episode')
            n=base.natural_length
            indices=[0,int(base.counts[0])-1,int(base.counts[0]),n-1,n+11]
            indices+=list(range(0,n,max(1,n//113)))
            for i in indices:
                assert old.reference_index(i)==global_reference.reference_index(i)
                assert base.locate(episode.reference_index(i))[0]==base.locate(i)[0]
                assert episode.reference_index(i)//n==i//n
            rng=torch.get_rng_state()
            for i in [0,n-1]:
                item=episode[i];ref=base[episode.reference_index(i)];factual=base[i]
                assert item['episode']==item['reference_episode']
                assert torch.equal(item['reference_pixels'],ref['pixels'][3:])
                assert torch.equal(item['pixels'],factual['pixels']) and torch.equal(item['action'],factual['action'])
            assert torch.equal(rng,torch.get_rng_state())
            rows.append(dict(task=task,split=split,indices=len(indices),global_indices_byte_compatible=True,
                             references_stay_in_episode_and_split=True,epoch_encoding_preserved=True,
                             factual_pixels_actions_unchanged=True,global_rng_unchanged=True))
    write(output,dict(passed=True,rows=rows,scope='Sampler contracts,not model/task improvement.'))
    print(dict(passed=True,datasets=len(rows)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old-source',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.old_source,a.output)
