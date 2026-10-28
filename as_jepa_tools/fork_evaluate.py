"""Audit held-out training-domain fork assignment without branch IDs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from as_jepa_tools.evaluate import load_model
from vforge.ops import OpsConfig, initialize_ops


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    for name in ("checkpoint", "forks", "output"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False), device=torch.device("cuda", 0), parallel_state=None)
    model = load_model(a.checkpoint)
    bank = Path(a.forks)
    indices = np.load(bank / "validation_indices.npy")
    cases = json.loads((bank / "cases.json").read_text())
    train_indices = np.load(bank / "train_indices.npy")
    train_episodes = {cases[int(x)]["episode"] for x in train_indices}
    assert not train_episodes.intersection(cases[int(x)]["episode"] for x in indices)
    context = np.load(bank / "contexts.npy", mmap_mode="r")
    futures = np.load(bank / "futures.npy", mmap_mode="r")
    actions = np.load(bank / "actions.npy", mmap_mode="r")
    records = []
    generator = torch.Generator(device="cuda").manual_seed(20261028)
    for start in range(0, len(indices), 16):
        selected = indices[start:start + 16]
        x = torch.from_numpy(np.array(context[selected])).permute(0, 3, 1, 2).cuda()
        y = torch.from_numpy(np.array(futures[selected])).permute(0, 1, 4, 2, 3).cuda()
        u = torch.from_numpy(np.array(actions[selected])).cuda().flatten(2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.encode(dict(pixels=model.normalize(torch.cat([x[:, None], y], 1))))["emb"]
            b = len(z)
            def logits(candidates):
                inputs = torch.cat([z[:, 0, None, None].expand(-1, 4, 4, -1),
                                    z[:, 1:, None].expand(-1, -1, 4, -1),
                                    candidates[:, None].expand(-1, 4, -1, -1)], -1)
                return model.inverse_head(inputs).squeeze(-1).float()
            original = logits(u)
            permutation = torch.randperm(4, device="cuda", generator=generator)
            reordered = logits(u[:, permutation])
            torch.testing.assert_close(reordered, original[:, :, permutation], rtol=2e-2, atol=2e-2)
            # Identical candidate actions remove any basis for identifying their slot.
            identical = logits(u[:, :1].expand(-1, 4, -1))
            spread = (identical - identical[:, :, :1]).abs().max()
            if float(spread) > 1e-5:
                raise AssertionError("Candidate-index leakage with identical actions")
        assigned = original.argmax(-1)
        reordered_assigned = permutation[reordered.argmax(-1)]
        for row, index in enumerate(selected):
            records.append(dict(index=int(index), episode=cases[int(index)]["episode"],
                                assigned=assigned[row].cpu().tolist(),
                                reordered_assigned=reordered_assigned[row].cpu().tolist()))
    truth = np.arange(4)
    result = dict(cases=len(records), chance=.25, episode_overlap=0,
                  accuracy=float(np.mean([np.array(r["assigned"]) == truth for r in records])),
                  candidate_permutation_accuracy=float(np.mean([np.array(r["reordered_assigned"]) == truth for r in records])),
                  details=records, checkpoint=a.checkpoint,
                  scope="Confirmation split of original training episodes; not final test-world-model performance. Exact action-candidate permutation and identical-candidate leakage checks.")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
