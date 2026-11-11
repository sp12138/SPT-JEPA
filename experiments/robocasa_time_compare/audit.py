"""Real-cache/backward and public K2 parity checks; no optimizer updates."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.robocasa_compare import train as common
from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_refine import encoder_ablation as spt
from experiments.robocasa_time_compare.train import build, prepare


def audit(root, run, local_data):
    torch.set_num_threads(2); torch.set_float32_matmul_precision("high")
    common.install_runtime(root/"code")
    from vforge.ops import OpsConfig, initialize_ops
    initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False),
                   device=torch.device("cuda"), parallel_state=None)
    cfg = json.loads((run/"config.json").read_text())
    state = torch.load(run/"last.ckpt", map_location="cpu", weights_only=False)
    assert state["config_sha256"] == sha(run/"config.json")
    assert state["step"] == 8 and state["samples"] == 256 and state["world_size"] == 8
    assert len(state["rank_rng"]) == 8
    model = build(root, cfg).cuda().eval()
    model.load_state_dict(state["model"], strict=True)
    before = common.state_hash(model.state_dict())
    assert before == state["nn_sha256"]
    data = spt.FeatureWindows(local_data, cfg)
    raw = torch.utils.data.default_collate([data[0]])
    batch = prepare(raw, cfg, "cuda")
    poisoned = dict(batch, action=batch["action"].clone())
    poisoned["action"][:, -1] = 1000.
    rng = common.rng_state()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a, _ = model.loss(batch, step=9)
        common.restore_rng(rng)
        b, _ = model.loss(poisoned, step=9)
    poison_error = float((a-b).abs())
    assert poison_error == 0., poison_error
    parity = None
    if cfg["method"] == "robojepa":
        # FP32 checks algebra independently of mixed FP16/BF16 concatenation
        # promotion in the two public wrappers (BF16 loss drift was1.55e-6).
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            precise = {k:v.float() for k,v in batch.items()}
            z = precise["features"].permute(0, 2, 1, 3, 4).reshape(1, 16, 2, 14, 14, 1664)
            actions = torch.nn.functional.pad(precise["action"], (0, 50))
            states = z.new_zeros(1, 16, 64)
            pred = model.predict(z, precise["action"])
            prefix = 3
            history = model.wm.select_history(z, actions, pred, states=states,
                predicted_states=None, camera_extrinsics=None, prefix_len=prefix)
            public, _ = model.wm.rollout(history, actions_suffix=actions[:, prefix:],
                features_suffix=z[:, prefix+1:], states_suffix=states[:, prefix+1:],
                padding_mask=z.new_ones(1, 15-prefix),
                view_map=torch.tensor([[0, 2]], device="cuda"),
                embodiment_ids=torch.tensor([[2]], device="cuda"), run_backward=False)
            direct, metrics = model.loss(precise, step=9, prefix=prefix)
            parity = float((public["loss"].mean()-metrics["ar"]).abs())
            assert parity < 1e-6, parity
            assert float((direct-(metrics["tf"]*.5+public["loss"].mean()*.25)).abs()) < 1e-6
    common.restore_rng(rng)
    model.train(); model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16): loss, _ = model.loss(batch, step=9)
    loss.backward()
    assert torch.isfinite(loss)
    gradients = {n:float(p.grad.float().norm()) for n,p in model.named_parameters() if p.grad is not None}
    assert gradients and all(np.isfinite(v) for v in gradients.values())
    assert common.state_hash(model.state_dict()) == before
    del model, state, batch, raw
    torch.cuda.empty_cache()
    # Re-encode ordinary training images using the actual cached preprocessing.
    card = json.loads((root/"data.json").read_text())
    record = next(r for r in card["records"] if r["split"] == "train")
    assert sha(Path(record["path"])) == record["sha256"]
    with np.load(record["path"]) as stream:
        images = torch.from_numpy(stream["pixels"][:4].copy()).flatten(0, 1).cuda()
    expected = torch.from_numpy(np.load(local_data/"features/vjepa"/(record["id"]+".npy"))[:4].copy()).flatten(0, 1).cuda()
    encoder = spt.load_encoder(root, "vjepa")
    observed = spt.encode(encoder, images, "vjepa")
    error = float((observed.float()-expected.float()).abs().max())
    assert error < .035, error
    save(run/"METHOD_PROOF.json", dict(passed=True, step=8, no_new_SGD=True,
        checkpoint_nn_sha256=before, checkpoint_model_hash_exact=True,
        real_training_backward_finite=True, terminal_action_poison_error=poison_error,
        public_K2_rollout_loss_max_error=parity, ordinary_RGB_cache_reencode_max_error=error,
        public_K2_parity_precision="FP32" if parity is not None else None,
        gradients_finite=True, no_model_weights_changed=True))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True); p.add_argument("--local-data", type=Path, required=True)
    a=p.parse_args(); audit(a.root, a.run, a.local_data)
