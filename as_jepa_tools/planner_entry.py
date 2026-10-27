"""Run the pinned upstream planner with reproducible solver RNG and optional split restriction."""
import importlib.util,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch


def to_json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:to_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [to_json(v) for v in value]
    return value


def main():
    upstream=Path(os.environ["ASJEPA_UPSTREAM"])
    output=Path(os.environ["ASJEPA_PLANNER_OUTPUT"])
    split=os.environ.get("ASJEPA_PLANNER_SPLIT","all")
    seed=int(os.environ.get("ASJEPA_PLANNER_SEED","42"))
    spec=importlib.util.spec_from_file_location("asjepa_pinned_planner",upstream/"scripts/plan/eval_wm.py")
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    if split=="heldout":
        allowed=np.load(os.environ["ASJEPA_TEST_EPISODES"])
        original_lengths=module.get_episodes_length
        def restricted_lengths(dataset,episodes):
            lengths=original_lengths(dataset,episodes)
            # Upstream excludes starts when length < goal_offset+1. Only selection
            # lengths change; source data, reset states and normalization stay identical.
            return np.where(np.isin(episodes,allowed),lengths,0)
        module.get_episodes_length=restricted_lengths
    import stable_worldmodel as swm
    original_evaluate=swm.World.evaluate
    def recorded_evaluate(world,*args,**kwargs):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        started=time.time()
        result=original_evaluate(world,*args,**kwargs)
        episodes=kwargs.get("episodes_idx",[])
        if split=="heldout" and not set(episodes).issubset(set(allowed.tolist())):
            raise AssertionError("Planner selected an episode outside the held-out split")
        report=dict(metrics=to_json(result),episodes=to_json(episodes),
                    start_steps=to_json(kwargs.get("start_steps")),split=split,
                    seed=seed,solver_rng_seed=seed,seconds=time.time()-started,
                    scope="Pinned upstream P0; deterministic posterior means; single trained seed")
        (output/"metrics.json").write_text(json.dumps(report,indent=2))
        return result
    swm.World.evaluate=recorded_evaluate
    module.run()


if __name__=="__main__":main()
