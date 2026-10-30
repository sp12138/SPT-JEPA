"""Explicit full-state and gradient audit before the trainer's 100-step log interval."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from as_jepa_tools.benchmark_data import BenchmarkWindows
from as_jepa_tools.benchmark_health import finite
from as_jepa_lewm.robust import RobustLeWM
from vforge.ops import OpsConfig, initialize_ops


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root, out = Path(a.run), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root / "config.json").read_text())
    state = torch.load(root / "last.ckpt", map_location="cpu", weights_only=False)
    step = int(state["global_step"])
    weights = torch.load(root / f"model-step{step:07d}/weights.pt", map_location="cpu", weights_only=True)
    checks = {key: finite(state[key]) for key in ["state_dict", "optimizer_states", "lr_schedulers"]}
    checks.update(rng_present=bool(state.get("campaign_rng")),
                  cursor=state["campaign_samples"] == step * cfg["batch_size"],
                  portable_equal=all(torch.equal(v, state["state_dict"]["model."+k]) for k, v in weights.items()))
    assert all(checks.values()), checks
    del state
    torch.set_num_threads(2)
    torch.manual_seed(1042)
    initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False), device=torch.device("cuda"), parallel_state=None)
    model = RobustLeWM(**cfg["model"]).cuda().train()
    model.load_state_dict(weights, strict=True)
    ds = BenchmarkWindows(cfg["data"], window=model.window, split="validation")
    batch = next(iter(DataLoader(Subset(ds, [0, 1]), batch_size=2, num_workers=0)))
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
    loss, metrics = model.loss(batch, step=step)
    assert finite(loss)
    loss.backward()
    norms = {}
    prefixes = ["encoder", "pool", "predictor", "renderer"]
    if getattr(model, "response_query_mode", "none") != "none":
        prefixes.append("response_query_head")
    for prefix in prefixes:
        grads = [p.grad for name, p in model.named_parameters() if name.startswith(prefix+".") and p.grad is not None]
        assert grads and all(finite(g) for g in grads), prefix
        norms[prefix] = float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt())
        assert norms[prefix] > 0, (prefix, norms)
    report = dict(run=str(root), step=step, checks=checks, loss=float(loss.detach()), gradient_norms=norms,
                  scope="No optimizer step. Explicit real-model backward because the existing trainer records gradient norms only every100 updates. Execution check, not a noncollapse result.")
    (out / "gradient-audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
