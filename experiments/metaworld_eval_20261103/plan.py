"""Freeze a small, paired comparison before any MetaWorld model outcomes."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def main(root, data):
    root = Path(root); destination = root / 'training'; destination.mkdir(exist_ok=False)
    recipes=[]
    for arm in ['temporal-forecast','as-jepa','forecast-render','native-lewm']:
        for seed in [1,2]:
            name = f'{arm}-seed{seed}'; run = destination / name; run.mkdir()
            model = dict(mode='L2',beta=.25,sigma0=.1,window=8,anchor_gap=8,
                         renderer_width=384,renderer_depth=12,renderer_heads=6,
                         action_dim=20,rho=.8 if arm=='temporal-forecast' else 0.,
                         forecast_fraction=0. if arm=='as-jepa' else .5,
                         forecast_seed=20261029 + seed*100003)
            if arm=='native-lewm':model=dict(window=8,action_dim=20)
            cfg=dict(output=str(run),data=str(data),evaluation_data=str(data),
                     dataset_kind='benchmark_hdf5',model_class='native_lewm' if arm=='native-lewm' else 'robust',
                     seed=seed,batch_size=128,workers=4,cpu_threads=2,max_steps=41200,max_seconds=172800,
                     checkpoint_every=824,retain_steps=[2,600,2472,8240],precision='bf16-mixed',
                     record_initial_weights=True,record_runtime_source=True,model=model,
                     audit_expectations=dict(gradients={} if arm=='native-lewm' else
                                             {k:'active' for k in ['encoder','pool','predictor','action_encoder','renderer']}))
            path=run/'config.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
            recipes.append(dict(name=name,arm=arm,seed=seed,run=str(run),
                                config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                host='dev2' if arm in ['temporal-forecast','as-jepa'] else 'dev15',
                                gpu=(0 if arm in ['temporal-forecast','forecast-render'] else 2)+seed-1))
    proof=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),recipes=recipes,
               natural_train_windows=105486,updates_per_loader_epoch=824,batch_size=128,
               scheduler_horizon=41200,stages=[2,600,2472,8240],finite_training_cap=8240,
               scope='RGB-only independent LeRobot data; no pretrained weights. Three AS arms exact paired initializations required before continuation. Native LeWM capacity/optimizer differ. Same dataset/split/action normalization/evaluation cases/controller. No task-specific hyperparameters. Old campaigns and goal remain stopped.')
    (root/'plan.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--data',required=True)
    args=parser.parse_args();main(args.root,args.data)
