"""Existing SWM planner plus pinned CLEAR scoring, on frozen campaign manifests."""
import argparse,hashlib,json,os,random,time
from pathlib import Path
import numpy as np,torch

def json_safe(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [json_safe(v) for v in x]
    return x

def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--data",required=True);p.add_argument("--output",required=True);p.add_argument("--limit",type=int);a=p.parse_args()
    os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("SDL_VIDEODRIVER","dummy")
    from hydra import compose,initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf,open_dict
    from sklearn.preprocessing import StandardScaler
    import stable_worldmodel as swm
    from clear_lewm.datasets import metadata_fingerprint
    from clear_lewm.protocols import protocol_from_dict
    from clear_lewm.runner import _install_task_success,_image_transform
    from clear_lewm.tworoom_runtime import topology_audit_records
    from vforge.ops import OpsConfig,initialize_ops
    b=Path("/efs/prescottsun/SPT-JEPA");out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    mp=Path(a.manifest);manifest=json.loads(mp.read_text());pairs=manifest["pairs"][:a.limit] if a.limit else manifest["pairs"];spec=protocol_from_dict(manifest["protocol"])
    card=json.loads((Path(a.data)/"dataset_card.json").read_text());assert metadata_fingerprint(card["h5_path"])==manifest["dataset"]["fingerprint"]["value"]
    allowed=np.load(Path(a.data)/(manifest["campaign_split"]+"_episodes.npy"));assert set(x["episode_id"] for x in pairs).issubset(set(allowed.tolist()))
    seed=int(manifest["policy_seed"]);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2);torch.set_float32_matmul_precision("highest")
    with initialize_config_dir(config_dir=str(b/"sources/stable-worldmodel/scripts/plan/config"),version_base=None):cfg=compose(config_name=manifest["task"])
    with open_dict(cfg):
        cfg.world.num_envs=len(pairs);cfg.world.max_episode_steps=2*spec.eval_budget;cfg.plan_config.history_len=3;cfg.solver.batch_size=1
    ds=swm.data.load_dataset(card["h5_path"],keys_to_cache=list(cfg.dataset.keys_to_cache))
    own=a.checkpoint!="random" and json.loads((Path(a.checkpoint)/"config.json").read_text()).get("_target_","").startswith("as_jepa_lewm.")
    process={}
    for col in cfg.dataset.keys_to_cache:
        vals=ds.get_col_data(col);vals=vals[np.isfinite(vals).all(axis=1)];scaler=StandardScaler().fit(vals)
        if col=="action" and own:
            scaler.mean_=np.asarray(card["action_mean"]);scaler.scale_=np.asarray(card["action_std"]);scaler.var_=scaler.scale_**2
        process[col]=scaler
        if col!="action":process["goal_"+col]=scaler
    audit=None
    if a.checkpoint=="random":policy=swm.policy.RandomPolicy(seed=seed)
    else:
        initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda"),parallel_state=None)
        model=swm.wm.utils.load_pretrained(a.checkpoint).cuda().eval();model.requires_grad_(False)
        weights=torch.load(Path(a.checkpoint)/"weights.pt",map_location="cpu",weights_only=True)
        assert set(weights)==set(model.state_dict());assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in weights.items())
        audit=dict(tensors=len(weights),parameters=sum(p.numel() for p in model.parameters()),strict_equal=True);del weights
        cost=swm.planning.ShootingCostEvaluator(model,instantiate(cfg.objective));solver=instantiate(cfg.solver,cost=cost)
        transform={"pixels":_image_transform(224),"goal":_image_transform(224)}
        policy=swm.policy.WorldModelPolicy(solver=solver,config=swm.PlanConfig(**cfg.plan_config),process=process,transform=transform)
    world=swm.World(**cfg.world,image_shape=(224,224));_install_task_success(world,manifest["task"],spec);world.set_policy(policy)
    started=time.time()
    try:metrics=world.evaluate(dataset=ds,start_steps=[x["start_step"] for x in pairs],goal_offset=spec.goal_offset,eval_budget=spec.eval_budget,episodes_idx=[x["episode_id"] for x in pairs],callables=OmegaConf.to_container(cfg.eval.callables,resolve=True),video=None)
    finally:world.close()
    report=dict(metrics=json_safe(metrics),pairs=pairs,manifest=str(mp),manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),checkpoint=a.checkpoint,checkpoint_audit=audit,seconds=time.time()-started,solver=OmegaConf.to_container(cfg.solver,resolve=True),plan=OmegaConf.to_container(cfg.plan_config,resolve=True),normalization="train episodes" if own else "released reference full-data normalization",split=manifest["campaign_split"],protocol=manifest["protocol"],topology=json_safe(topology_audit_records()) if manifest["task"]=="tworoom" else [],scope="Custom SPT held-out manifest using pinned CLEAR task semantics and existing SWM planner; not canonical CLEAR benchmark. Released reference weights have different training/data exposure.")
    (out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(dict(metrics=report["metrics"],seconds=report["seconds"])),flush=True)
if __name__=="__main__":main()
