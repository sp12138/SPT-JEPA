"""Launch the pinned upstream PushT planner with campaign-owned outputs."""
import argparse,os,subprocess,json,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--upstream',required=True);p.add_argument('--data-root',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--episodes',type=int,default=50);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);checkpoint=Path(a.checkpoint).resolve();cfg=json.loads((checkpoint/'config.json').read_text())
 # The evaluator writes beside policy. Copy only the tiny config and link read-only weights.
 policy=out/'policy';policy.mkdir(exist_ok=True);(policy/'config.json').write_text(json.dumps(cfg,indent=2))
 link=policy/'weights.pt'
 if not link.exists():link.symlink_to(checkpoint/'weights.pt')
 cache=Path(a.data_root);datasets=cache/'datasets';datasets.mkdir(parents=True,exist_ok=True)
 dataset=datasets/'pusht_expert_train.h5'
 if not dataset.exists():dataset.symlink_to('/efs/prescottsun/SPT/data/pusht_hjepa/assets/pusht_expert_train.h5')
 command=[os.sys.executable,str(Path(a.upstream)/'scripts/plan/eval_wm.py'),f'policy={policy}',f'cache_dir={cache}',f'eval.num_eval={a.episodes}',f'seed={a.seed}','+plan_config.history_len=3','bf16=true','compile=false',f'hydra.run.dir={out}/hydra']
 # cache_dir is not present in the upstream config; declare it for Hydra's strict parser.
 command=[('+cache_dir='+str(cache)) if item.startswith('cache_dir=') else item for item in command]
 import hdf5plugin
 started=time.time();env={**os.environ,'HDF5_PLUGIN_PATH':hdf5plugin.PLUGINS_PATH,'STABLEWM_HOME':str(cache),'MUJOCO_GL':'egl','SDL_VIDEODRIVER':'dummy','PYGLET_HEADLESS':'true'}
 with (out/'planner.log').open('w') as log:r=subprocess.run(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
 result=dict(returncode=r.returncode,seconds=time.time()-started,seed=a.seed,episodes=a.episodes,command=command,scope='pinned upstream PushT P0: CEM300x30, horizon5, history3, offset25, budget50')
 (out/'execution.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 if r.returncode:raise SystemExit(r.returncode)

if __name__=='__main__':main()
