"""Execute native CPU restoration on each copied parent state before updates."""
import argparse
import gc
from pathlib import Path
import torch

from common import read,write,sha
from as_jepa_lewm.train import TrainModule
from experiments.unified_three_20261101.common import same


def main(root):
    root=Path(root);p=read(root/'protocol.json');rows=[];torch.set_num_threads(2)
    for row in p['runs']:
        if not row['start_step']:continue
        run=Path(row['run']);cfg=read(run/'config.json');binding=read(run/'copy-binding.json')
        assert sha(run/'last.ckpt')==binding['copied_sha256']==binding['parent_sha256']
        cp=torch.load(run/'last.ckpt',map_location='cpu',weights_only=False)
        torch.manual_seed(cfg['seed']);module=TrainModule(cfg);module.load_state_dict(cp['state_dict'],strict=True)
        opt=module.configure_optimizers();optimizer=opt['optimizer'];scheduler=opt['lr_scheduler']['scheduler']
        optimizer.load_state_dict(cp['optimizer_states'][0]);scheduler.load_state_dict(cp['lr_schedulers'][0])
        assert same(module.state_dict(),cp['state_dict']) and same(optimizer.state_dict(),cp['optimizer_states'][0]) and same(scheduler.state_dict(),cp['lr_schedulers'][0])
        assert cp['global_step']==2000 and cp['campaign_samples']==2000*128 and cp['campaign_rng']
        rows.append(dict(id=row['id'],passed=True,step=2000,sha256=binding['copied_sha256'],native_cpu_model_optimizer_scheduler_restore=True,saved_rng_fields_checked=True,rng_replay_claim=False))
        del cp,module,opt,optimizer,scheduler;gc.collect()
    assert len(rows)==18;write(root/'copy-restores.json',dict(passed=True,rows=rows));print(dict(passed=True,copies=len(rows)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
