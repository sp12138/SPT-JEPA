"""Launch the pinned upstream PushT planner with campaign-owned outputs."""
import argparse,os,subprocess,json,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True)
    p.add_argument("--upstream",required=True);p.add_argument("--data-root",required=True)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--episodes",type=int,default=50)
    p.add_argument("--split",choices=["all","heldout"],default="heldout")
    p.add_argument("--test-episodes");p.add_argument("--bf16",action="store_true")
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);checkpoint=Path(a.checkpoint).resolve()
    cfg=json.loads((checkpoint/"config.json").read_text())
    if cfg.get("_target_") not in {"as_jepa_lewm.model.ASJepaLeWM","as_jepa_lewm.research.ResearchLeWM","as_jepa_lewm.robust.RobustLeWM"}:
        raise ValueError("This planner entry requires the LeWM rollout interface")
    policy=out/"policy";policy.mkdir(exist_ok=True);(policy/"config.json").write_text(json.dumps(cfg,indent=2))
    if not (policy/"weights.pt").exists():os.link(checkpoint/"weights.pt",policy/"weights.pt")
    cache=Path(a.data_root);datasets=cache/"datasets";datasets.mkdir(parents=True,exist_ok=True)
    dataset=datasets/"pusht_expert_train.h5"
    if not dataset.exists():dataset.symlink_to("/efs/prescottsun/SPT/data/pusht_hjepa/assets/pusht_expert_train.h5")
    command=[os.sys.executable,"-u","-m","as_jepa_tools.planner_entry",
             f"policy={policy}",f"+cache_dir={cache}",f"eval.num_eval={a.episodes}",f"seed={a.seed}",
             "+plan_config.history_len=3",f"bf16={str(a.bf16).lower()}","compile=false",f"hydra.run.dir={out}/hydra",f"world.num_envs={a.episodes}","--config-path",str(Path(a.upstream).resolve()/"scripts/plan/config")]
    import hdf5plugin
    env={**os.environ,"HDF5_PLUGIN_PATH":hdf5plugin.PLUGINS_PATH,"STABLEWM_HOME":str(cache),
         "MUJOCO_GL":"egl","SDL_VIDEODRIVER":"dummy","PYGLET_HEADLESS":"true",
         "ASJEPA_UPSTREAM":str(Path(a.upstream).resolve()),"ASJEPA_PLANNER_OUTPUT":str(out.resolve()),
         "ASJEPA_PLANNER_SPLIT":a.split,"ASJEPA_PLANNER_SEED":str(a.seed)}
    if a.split=="heldout":
        if not a.test_episodes:p.error("--test-episodes is required for heldout selection")
        env["ASJEPA_TEST_EPISODES"]=str(Path(a.test_episodes).resolve())
    started=time.time()
    with (out/"planner.log").open("w") as log:
        r=subprocess.run(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
    result=dict(returncode=r.returncode,seconds=time.time()-started,seed=a.seed,episodes=a.episodes,
                split=a.split,bf16=a.bf16,command=command,checkpoint=str(checkpoint),
                scope="pinned upstream P0: CEM300x30, horizon5, history3, offset25, budget50; split restricted before case selection")
    (out/"execution.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if r.returncode:raise SystemExit(r.returncode)


if __name__=="__main__":main()
