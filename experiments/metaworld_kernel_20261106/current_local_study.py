"""Exact native-state continuations,with explicitly separate own-intervention data."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from as_jepa_tools.local_inverse_mix import LocalInverseMix
from as_jepa_lewm.factory import build_model
from experiments.metaworld_kernel_20261106.campaign import save,digest
from experiments.metaworld_kernel_20261106.audit import equal


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=False);recipes=[]
    for objective,parent_root in[('flow',a.flow_parent),('mse',a.mse_parent)]:
        for seed in[1,2]:
            gpu=(0 if objective=='flow'else 2)+seed-1;parent=parent_root/'training'/f'episodes-seed{seed}'
            assert json.loads((parent/'audit-20000.json').read_text())['passed']
            run=a.root/'training'/f'{objective}-seed{seed}';run.mkdir(parents=True);cfg=json.loads((parent/'config.json').read_text())
            cfg.update(output=str(run),dataset_kind='local_inverse_mix',local_inverse_cache='/tmp/prescottsun-intervention-rgb224-20261107-v1',
                retain_steps=[20000,20002,24000,30000,40000])
            assert cfg['model']['chunk']==1 and cfg['max_steps']==40000
            save(run/'config.json',cfg);native=parent/'archive-step0020000/native.ckpt';shutil.copy2(native,run/'last.ckpt')
            shutil.copytree(parent/'model-step0020000',run/'model-step0020000')
            save(run/'ORIGIN.json',dict(parent=str(parent),parent_step=20000,native_parent_sha256=digest(native),
                parent_model_sha256=digest(parent/'model-step0020000/weights.pt'),full_model_optimizer_scheduler_RNG_cursor_inherited=True,
                origin='Ownscratchcurrent-only20k;oldfuture-chunkweightsNEVERloaded.TotaleligibleSGD40000,new20000.Searchcostinseparateoriginaltrackdisclosed.',
                data='Original+own11600branches/400contexts/108000newphysicalsteps;naturalwindowunion,no hand-tunedmixture.',
                loss='Singlecurrent-inverseloss;newlocalgoalistheACTUALendofcurrent5commands.No optimalityassertion/futureactionlabel.',
                comparison='Original20k->40kcontinuationsalreadycompleted,neverreplaythem.Augmentationchangesdatadistributionbutpreservesinitialfulltrainingstate.',
                limitation='Extra-dataresult,notoriginal-corpus/publicSOTAalgorithmclaim;identicalcorpusmustbeavailabletocomparators.'))
            recipes.append(dict(arm=objective,seed=seed,host='dev6',gpu=gpu,run=str(run),initial_step=20000,resource_pid=0,resource_marker='unallocated',config_sha256=digest(run/'config.json')))
    old=json.loads((a.flow_parent/'training-plan.json').read_text())
    save(a.root/'training-plan.json',dict(source=str(a.source),recipes=recipes,stages=[20000,20002,24000,30000,40000],manifest=old['manifest'],deadline_unix=time.time()+36*3600,
        separate_extra_data_track=True,scope='Fourcontinuationsfromtwoseedsxflow/MSE;current-onlyeligibleancestry.One sample-levelinverseobjective.Noextrafuture-actionreferences.'))


def contract(a):
    torch.set_num_threads(2);plan=json.loads((a.root/'training-plan.json').read_text());reports=[]
    for r in plan['recipes']:
        run=Path(r['run']);cfg=json.loads((run/'config.json').read_text());data=LocalInverseMix(cfg);test=LocalInverseMix(cfg,'test')
        assert not{v['episode']for v in data.records}&{v['episode']for v in test.records}
        groups=lambda d:{(t,rec['group'])for t in range(4)for rec in d.local.records if rec['shard']==t}
        assert not groups(data)&groups(test)
        for i in np.linspace(0,len(test.local)-1,64,dtype=int):
            v=test.local[int(i)];rec=next(x for x in test.local.records if x['episode']*5==v['episode']);shard=test.local.shards[rec['shard']];start=v['current_interval_start'];stop=v['current_interval_stop'];old=shard['inputs']
            class Guard:
                def __getitem__(self,key):
                    assert isinstance(key,slice)and key.start==start and key.stop==stop
                    return old[key]
            shard['inputs']=Guard()
            try:assert equal(v,test.local[int(i)])
            finally:shard['inputs']=old
            raw=np.load(rec['source']);t=v['start'];expected=(raw['controls'][t:t+5].reshape(20)-test.local.mean)/test.local.std
            assert np.array_equal(expected,v['action'][0].numpy())
            assert np.array_equal(raw['pixels'][t+4],v['pixels'][1].permute(1,2,0).numpy())
            if t:assert np.array_equal(raw['pixels'][t-1],v['pixels'][0].permute(1,2,0).numpy())
        ckpt=torch.load(run/'last.ckpt',map_location='cpu',weights_only=False)
        assert ckpt['global_step']==20000 and ckpt['campaign_samples']==20000*128
        assert digest(run/'last.ckpt')==json.loads((run/'ORIGIN.json').read_text())['native_parent_sha256']
        model=build_model(cfg).eval();model.load_state_dict({k.removeprefix('model.'):v for k,v in ckpt['state_dict'].items()},strict=True)
        before={k:v.clone()for k,v in model.state_dict().items()};batch=next(iter(DataLoader(Subset(test,[0,len(test.original),len(test)-1]),batch_size=3)))
        loss,_=model.loss(batch);assert torch.isfinite(loss);loss.backward();assert equal(before,model.state_dict())
        reports.append(dict(arm=r['arm'],seed=r['seed'],original_train=len(data.original),local_train=len(data.local),original_test=len(test.original),local_test=len(test.local),
            local_current_target_guards=64,rawRGB_and_current_control_parity=True,parent_state_exact=True))
    save(a.root/'initial-contract.json',dict(passed=True,reports=reports,current_interval_only=True,grouped_parent_split=True,new_model_updates=0,
        scope='Newraw224pixelscopiedfromowninterventionswithoutinterpolation.Originalandaugmentedrowsshareoneceterisparibusinverseobjective;no hiddenstate/reward/taskID.'))
    print('PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','contract'],required=True)
    for k in['root','source','flow-parent','mse-parent']:p.add_argument('--'+k,type=Path)
    a=p.parse_args();prepare(a)if a.mode=='prepare'else contract(a)
