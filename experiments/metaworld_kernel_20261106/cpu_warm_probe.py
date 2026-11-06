"""Small matched native CPU training ablation;never an action learner."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset


def run(a):
    from as_jepa_lewm.train import TrainModule
    from as_jepa_tools.goal_rgb_paths import GoalRGBPaths
    from experiments.metaworld_kernel_20261106.audit import equal
    torch.set_num_threads(2);cfg=json.loads(a.config.read_text());output=Path(cfg['output'])
    cmd=[a.python,'-B','-u',str(a.source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'),
        '--config',str(a.config),'--stop-step','100']
    with(output/'training.log').open('x')as log:subprocess.run(cmd,cwd=a.source,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    saved=torch.load(output/'last.ckpt',map_location='cpu',weights_only=False);module=TrainModule(cfg).cpu();module.load_state_dict(saved['state_dict'],strict=True)
    options=module.configure_optimizers();options['optimizer'].load_state_dict(saved['optimizer_states'][0]);options['lr_scheduler']['scheduler'].load_state_dict(saved['lr_schedulers'][0])
    assert saved['global_step']==100 and saved['campaign_samples']==800
    restore=dict(model=equal(module.state_dict(),saved['state_dict']),optimizer=equal(options['optimizer'].state_dict(),saved['optimizer_states'][0]),
        scheduler=equal(options['lr_scheduler']['scheduler'].state_dict(),saved['lr_schedulers'][0]))
    assert all(restore.values())
    assert equal(module.model.world.state_dict(),torch.load(cfg['model']['world_checkpoint'],map_location='cpu',weights_only=True))
    data=GoalRGBPaths(cfg,'test');indices=np.linspace(0,len(data)-1,256,dtype=int).tolist();loader=DataLoader(Subset(data,indices),batch_size=16)
    measurements={}
    for name,path in [('initial',output/'initial-model/weights.pt'),('trained100',output/'model-step0000100/weights.pt')]:
        module.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));module.model.eval();errors=[];copy=[]
        with torch.inference_mode():
            for batch in loader:
                prediction=module.model.distribution(batch);target=module.model.encoder(batch['next_pixels']);current=module.model.encoder(batch['current_pixels'])
                errors.extend((prediction-target).square().mean((1,2,3)).tolist());copy.extend((current-target).square().mean((1,2,3)).tolist())
        measurements[name]=dict(mse=float(np.mean(errors)),copy_mse=float(np.mean(copy)),beats_copy=sum(x<y for x,y in zip(errors,copy)))
    report=dict(passed=True,native_training_updates=100,training_examples=800,heldout_windows=256,actual_cpu_restore=restore,world_unchanged=True,
        warm=cfg['model']['init_from_world'],measurements=measurements,
        scope='Own same-data20kforwardparent+100nativeCPUupdates.Only futureRGBsupervision.Noexternal/inversepretraining/optimalactionlabels.'
              'Training-efficiency diagnostic only,not task-success evidence.PrimaryGPUlearners unchanged;theseCPUweights are not their parents.')
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--python',required=True);run(p.parse_args())
