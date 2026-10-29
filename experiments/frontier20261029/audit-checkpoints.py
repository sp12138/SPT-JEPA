"""Read-only audit of staged full and portable model checkpoints."""
import argparse,json,math,time
from pathlib import Path
import torch


def check_tensors(value):
    if torch.is_tensor(value):
        assert torch.isfinite(value).all()
        return value.numel()
    if isinstance(value,dict):return sum(check_tensors(v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(check_tensors(v) for v in value)
    if isinstance(value,float):assert math.isfinite(value)
    return 0


def main():
    p=argparse.ArgumentParser();p.add_argument("--campaign",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    torch.set_num_threads(2);r=Path(a.campaign);plan=json.loads((r/"plan.json").read_text());started=time.time();rows=[];state_elements=None
    for j in plan["jobs"]:
        if j["phase"]!="adaptation600":continue
        e=json.loads((r/"exits"/(j["id"]+".json")).read_text());assert e["returncode"]==0
        folder=Path(j["output"]);size=json.loads((folder/"model_size.json").read_text());assert size["total"]==51097134
        stage=json.loads((folder/"stage-complete.json").read_text());assert stage["initial_step"]==24982 and stage["stop_step"]==25582
        weights=torch.load(folder/"model-step0025582/weights.pt",map_location="cpu",weights_only=True);count=check_tensors(weights)
        if state_elements is None:state_elements=count
        assert count==state_elements
        state=torch.load(folder/"last.ckpt",map_location="cpu",weights_only=False)
        assert state["global_step"]==25582 and state["campaign_samples"]==25582*128
        assert state["campaign_rng"] and state["optimizer_states"] and state["lr_schedulers"]
        assert set(state["state_dict"])=={"model."+k for k in weights}
        assert all(torch.equal(weights[k],state["state_dict"]["model."+k]) for k in weights)
        opt=check_tensors(state["optimizer_states"]);check_tensors(state["lr_schedulers"])
        rows.append(dict(id=j["id"],all_weights_and_fixed_buffers_finite=True,model_parameters=size["total"],
            state_dict_elements_including_fixed_buffers=count,global_step=state["global_step"],samples_consumed=state["campaign_samples"],
            full_and_portable_tensors_equal=True,optimizer_tensors_finite=True,optimizer_tensor_elements=opt))
        del state,weights
    result=dict(all20_passed=len(rows)==20,all_expected_passed=len(rows)==sum(j["phase"]=="adaptation600" for j in plan["jobs"]),checked_training_runs=len(rows),seconds=time.time()-started,rows=rows,
        scope="Exact update and sample cursor; finite model/optimizer/scheduler tensors; full and portable model tensors exactly equal; RNG exists. Fixed state buffers are distinct from model parameters.")
    assert result["all_expected_passed"];out=Path(a.output);assert not out.exists();out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}))
if __name__=="__main__":main()
