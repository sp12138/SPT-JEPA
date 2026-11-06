"""Validate real branch alignment and paired observation-only fine-tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from as_jepa_tools.query_branch_mix import QueryBranchMix
from as_jepa_lewm.factory import build_model
from experiments.metaworld_kernel_20261106.audit import equal


def run(a):
    torch.set_num_threads(2);cfg=json.loads(a.config.read_text());data=QueryBranchMix(cfg);branch=data.branches
    selected={0:[],1:[]}
    for i in np.linspace(0,len(branch)-1,256,dtype=int):
        row=branch[int(i)];role=int(row['boundary'])
        assert row['future_pixels'].shape==(5,3,128,128)
        if role:assert(row['executed_sequence']==0).all()and(row['previous_control_sequence']==0).all()and row['target_row']<row['goal_row']
        if len(selected[role])<2:selected[role].append(int(i))
    assert all(len(x)==2 for x in selected.values())
    batch=next(iter(DataLoader(Subset(branch,selected[0]+selected[1]),batch_size=4)))
    model=build_model(cfg).eval();parent=torch.load(cfg['model']['warm_start_checkpoint'],map_location='cpu',weights_only=True);assert equal(parent,model.state_dict())
    loss,_=model.loss(batch);assert torch.isfinite(loss);loss.backward()
    with torch.no_grad():
        separate=torch.stack([model.loss({k:v[j:j+1]if torch.is_tensor(v)else v for k,v in batch.items()})[0]for j in range(4)]).mean()
        relative=float((separate-loss.detach()).abs()/(1+loss.detach().abs()));assert relative<2e-5
    # Check actual stored physical suffix bytes/command alignment on64 starts.
    for i in np.linspace(0,len(branch)-1,64,dtype=int):
        ep=int(np.searchsorted(branch.ends,int(i),side='right'));record=branch.records[ep];shard=branch.shards[record['shard']]
        start=int(i)-(int(branch.ends[ep-1])if ep else 0);value=branch[int(i)]
        future=np.asarray(shard['pixels'][record['offset']+start:record['offset']+start+5])
        assert np.array_equal(value['future_pixels'].permute(0,2,3,1).numpy(),future)
        if not value['boundary']:
            assert np.array_equal(value['executed_sequence'].numpy(),shard['inputs'][record['offset']+start:record['offset']+start+5])
    for i in np.linspace(0,len(data.original)-1,16,dtype=int):assert equal(data.original[int(i)],data[int(i)])
    report=dict(passed=True,parent_weights_exact=True,original_train_windows=len(data.original),new_train_windows=len(branch),
        natural_union_length=len(data),same_observation_label_per_branch=True,actual_suffix_alignment_checks=64,original_prefix_checks=16,
        sample_loss_independence_relative=relative,loss=float(loss.detach()),updates=0,
        scope='Extracorpusonly;actualowncontrolsforwardINPUTonly.GgoalRGBstrictlyafterallfivefuturelabels;allGcontrols/pastcontrolszero.'
              'Noprivilegedarrays/reward/success/taskIDmodelinput;parentgroupssplit.Dataaugmentationstudy,notsame-corpusmethodsuperiority.'))
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
