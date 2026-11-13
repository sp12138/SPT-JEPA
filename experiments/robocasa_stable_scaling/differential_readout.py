"""Finite detached inverse diagnostic: exact versus general differential forms.

Both heads vanish for identical images and reverse sign when the image pair is
reversed. The quadratic head is a potential difference; the bilinear head also
permits a non-integrable local action map. Labels remain the same factual a_t.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_action_probe import probe


def pair_features(previous, future, kind):
    delta = future-previous
    midpoint = (future+previous)*.5
    d = delta.shape[-1]
    if kind == "bilinear":
        context = (midpoint[..., :, None]*delta[..., None, :]).flatten(-2)
    elif kind == "antisymmetric":
        i, j = torch.triu_indices(d, d, offset=1, device=delta.device)
        context = (midpoint[..., i]*delta[..., j]-midpoint[..., j]*delta[..., i]) / 2**.5
    else:
        assert kind == "quadratic"
        i, j = torch.triu_indices(d, d, device=delta.device)
        context = midpoint[..., i]*delta[..., j]
        off = i != j
        context[..., off] = (context[..., off]+midpoint[..., j[off]]*delta[..., i[off]]) / 2**.5
    return torch.cat((delta, context), -1)


def fit(root, source, kind):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    directory = source / "heads/spt64-linear"
    rows, codes, actions, splits = [], [], [], []
    for shard in range(8):
        path = directory / "cache" / f"part-{shard}.npz"
        proof = json.loads(path.with_suffix(".json").read_text())
        assert proof["passed"] and proof["readout_precision"] == "fp32"
        assert sha(path) == proof["sha256"]
        with np.load(path) as data:
            rows.extend(data["indices"].tolist()); codes.extend(data["codes"])
            actions.extend(data["actions"]); splits.extend(data["splits"].tolist())
    assert sorted(rows) == list(range(1216)) and len(set(rows)) == len(rows)
    order = np.argsort(rows)
    z = torch.from_numpy(np.stack(codes)[order]).double().cuda()
    y = torch.from_numpy(np.stack(actions)[order]).double().cuda()
    mask = torch.as_tensor(np.asarray(splits)[order] == "train", device="cuda")
    x = pair_features(z[:, :-1], z[:, 1:], kind)
    xt, yt = x[mask].flatten(0, 1), y[mask].flatten(0, 1)
    xv, yv = x[~mask].flatten(0, 1), y[~mask].flatten(0, 1)
    assert len(xt) == 17920 and len(xv) == 1536
    zero = pair_features(z[:1, 0], z[:1, 0], kind)
    reverse = pair_features(z[:1, 1], z[:1, 0], kind)
    forward = pair_features(z[:1, 0], z[:1, 1], kind)
    assert bool((zero == 0).all()) and torch.equal(reverse, -forward)
    del x
    gram = xt.T @ xt / len(xt)
    rhs = xt.T @ yt / len(xt)
    energy = float(gram.diag().mean())
    assert energy > 0 and np.isfinite(energy)
    scores = []
    best = None
    for alpha in json.loads((root / "plan.json").read_text())["ridge_lambdas"]:
        regularized = gram.clone()
        regularized.diagonal().add_(alpha*energy)
        factor, info = torch.linalg.cholesky_ex(regularized)
        assert int(info) == 0, (kind, alpha, int(info))
        weight = torch.cholesky_solve(rhs, factor).T
        row = dict(alpha=alpha, train_RMSE=float((xt @ weight.T-yt).square().mean().sqrt()),
                   validation_RMSE=float((xv @ weight.T-yv).square().mean().sqrt()))
        assert np.isfinite(row["validation_RMSE"])
        scores.append(row)
        if best is None or row["validation_RMSE"] < best[0]:
            best = (row["validation_RMSE"], weight.float().cpu(), alpha)
        save(root / "heads/spt64-linear/progress.json", dict(phase="ridge", kind=kind, scores=scores))
        del regularized, factor, weight
    source_card = json.loads((root / "data.json").read_text())["records"]
    assert not {tuple(r["kitchen"]) for r in source_card if r["split"] == "train"} & {
        tuple(r["kitchen"]) for r in source_card if r["split"] == "validation"}
    frozen_input = xt[:4].float().detach().requires_grad_()
    head = best[1].cuda().detach().requires_grad_()
    loss = (frozen_input.detach() @ head.T-yt[:4].float()).square().mean()
    loss.backward()
    assert frozen_input.grad is None and head.grad is not None
    assert torch.isfinite(head.grad).all() and float(head.grad.norm()) > 0
    out = root / "heads/spt64-linear"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(weight=best[1], kind=kind, alpha=best[2],
                    target="same_transition_issued_translation_a_t", body_gradient=False), out / "head.pt")
    extra = {}
    if kind == "bilinear":
        matrix = best[1][:, z.shape[-1]:].reshape(3, z.shape[-1], z.shape[-1]).double()
        symmetric = (matrix+matrix.transpose(-1, -2))*.5
        antisymmetric = matrix-symmetric
        extra["antisymmetric_weight_norm_fraction"] = float(antisymmetric.norm()/matrix.norm())
    save(out / "ready.json", dict(passed=True, kind=kind, source=str(source),
        head_sha256=sha(out / "head.pt"), input_width=xt.shape[-1], base_width=z.shape[-1],
        scores=scores, selected_alpha=best[2], validation_RMSE=best[0],
        train_transitions=len(xt), validation_transitions=len(xv), readout_precision="fp32",
        zero_and_reversal_identity_pass=True, actual_stopgradient_backward_pass=True,
        no_body_updates=True, no_future_or_optimal_action_labels=True,
        target_case_feedback_used_for_selection=False, physical_kitchen_split_disjoint=True, **extra))


class Policy(probe.Policy):
    @torch.no_grad()
    def command(self, observation, goal):
        current = self.encode(observation)
        head = self.head_kind
        value = self.weight @ pair_features(current, goal, head)
        assert torch.isfinite(value).all()
        value = value*torch.clamp(.3/value.norm().clamp_min(1e-12), max=1.)
        command = np.zeros(7, np.float32)
        command[:3] = value.cpu().numpy()
        return command


def control(root, shard, shards):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    policy = Policy(root, "spt64-linear")
    policy.head_kind = torch.load(root / "heads/spt64-linear/head.pt", map_location="cpu", weights_only=True)["kind"]
    cases = json.loads((root / "plan.json").read_text())["cases"]
    for index in range(shard, len(cases), shards):
        from experiments.robocasa_compare.evaluate import case_name
        folder = root / "evaluation/spt64-linear" / case_name(**cases[index])
        previous = folder / "summary.json"
        if previous.exists():
            proof = json.loads(previous.read_text())
            assert proof["passed"] and proof["model_unchanged"] and proof["physical_replay_pass"]
            assert sha(folder / "trace.npz") == proof["trace_sha256"]
            continue
        probe.evaluate(root, "spt64-linear", index, policy=policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--kind", choices=("quadratic", "bilinear", "antisymmetric"))
    parser.add_argument("--mode", choices=("fit", "control"), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=2)
    args = parser.parse_args()
    if args.mode == "fit":
        assert args.source and args.kind
        fit(args.root, args.source, args.kind)
    else:
        control(args.root, args.shard, args.shards)
