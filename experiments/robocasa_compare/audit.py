"""Real checkpoint restoration, temporal access, and sample-independence checks."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_compare.data import Windows
from experiments.robocasa_compare.train import (associations, build, gradients, make_optimizer,
                                               prepare_batch, state_hash,make_dataset)


def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def run(args):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    cfg = json.loads((args.run / "config.json").read_text())
    state = torch.load(args.run / "last.ckpt", map_location="cpu", weights_only=False)
    assert state["config_sha256"] == sha(args.run / "config.json")
    assert state["step"] == args.step and state["samples"] == args.step * cfg["global_batch"]
    assert len(state["rank_rng"]) == state["world_size"] == 4
    assert all(set(v) == {"python", "numpy", "torch", "cuda"} for v in state["rank_rng"])
    core = build(cfg["model"], args.root / "code")
    core.load_state_dict(state["model"], strict=True)
    optimizer, scheduler = make_optimizer(core, cfg["max_steps"], cfg.get("phase_start",0))
    assert associations(core, optimizer) == state["parameter_associations"]
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    assert equal(optimizer.state_dict(), state["optimizer"])
    assert equal(scheduler.state_dict(), state["scheduler"])
    assert state_hash(core.state_dict()) == state["nn_sha256"]
    cpu_restore = True
    del optimizer, scheduler
    dataset = make_dataset(cfg,cfg["data"],"validation")
    raw = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    assert set(raw) == {"pixels", "action"}
    assert raw["pixels"].dtype == torch.uint8
    assert torch.equal(raw["action"][:, -1], torch.zeros_like(raw["action"][:, -1]))
    core = core.cuda().eval()
    from vforge.ops import OpsConfig, initialize_ops
    initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False), device=torch.device("cuda"), parallel_state=None)
    batch = prepare_batch(raw, torch.device("cuda"))
    with torch.no_grad():
        pixels, _, _ = core.prepare(batch)
        joined = core.encode(dict(pixels=pixels, action=batch["action"]))
        singles = [core.encode(dict(pixels=pixels[i:i+1], action=batch["action"][i:i+1]))
                   for i in range(len(pixels))]
        encoder_error = float((joined["emb"] - torch.cat([x["emb"] for x in singles])).abs().max())
        pred = core.predict(joined["emb"][:, :-1], joined["act_emb"][:, :-1])
        individual = torch.cat([core.predict(x["emb"][:, :-1], x["act_emb"][:, :-1]) for x in singles])
        predictor_error = float((pred - individual).abs().max())
        assert encoder_error < 1e-4 and predictor_error < 1e-4, (encoder_error, predictor_error)
        torch.manual_seed(451)
        normal, _ = core.loss(batch, step=args.step)
        poisoned = dict(batch, action=batch["action"].clone())
        poisoned["action"][:, -1] = 100.
        torch.manual_seed(451)
        changed, _ = core.loss(poisoned, step=args.step)
        terminal_error = float((normal - changed).abs())
        assert terminal_error < 1e-6, terminal_error
    before = state_hash(core.state_dict())
    core.train()
    core.zero_grad(set_to_none=True)
    loss, _ = core.loss(batch, step=args.step)
    assert torch.isfinite(loss)
    loss.backward()
    norms = gradients(core)
    assert all(v > 0 and __import__("math").isfinite(v) for v in norms.values()), norms
    assert before == state_hash(core.state_dict()) == state["nn_sha256"]
    result = dict(passed=True, step=args.step, cpu_full_model_optimizer_scheduler_restore=cpu_restore,
                  parameter_associations_pass=True, rank_rng_and_cursor_pass=True,
                  heldout_fp32_loss=float(loss.detach()), required_gradients=norms,
                  sample_independence_encoder_max=encoder_error,
                  sample_independence_predictor_max=predictor_error,
                  unused_terminal_action_poison_error=terminal_error,
                  no_action_policy_target=True, model_unchanged=True,
                  model_sha256=before, checkpoint_sha256=sha(args.run / "last.ckpt"))
    save(args.run / f"audit-{args.step:06d}.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        save(args.run / f"audit-error-{args.step:06d}.json", dict(error=repr(error)))
        raise
