"""Estimate a nuisance action subspace from factual RGB transitions.

Angular commands are forward inputs, observed code changes are targets. The
frozen encoder and the original current-XYZ labels never receive these fits.
P=I-R R^+ removes the fitted rotation response before the standard XYZ probe.
This is a global linear approximation; overlapping translation/rotation codes
or state-dependent responses can make it fail. New interaction cost is extra.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.robocasa_compare.acquire import save, sha


def fit_rotation(root, facts):
    records = sorted((facts/"records").glob("*/ready.json"))
    assert len(records) == 16
    allowed = {tuple(r["kitchen"]) for r in json.loads((facts/"data.json").read_text())["records"] if r["split"] == "train"}
    inputs, targets, current_xyz, sources, kitchens, parent = [], [], [], [], set(), None
    for path in records:
        proof = json.loads(path.read_text()); directory = path.parent
        assert proof["passed"] and proof["split"] == "train" and tuple(proof["kitchen"]) in allowed
        assert proof["all_learning_weights_unchanged"] and proof["no_learned_policy_controls_collection"]
        with np.load(directory/"experience.npz") as stream:
            assert set(stream.files) == {"pixels", "commands"}
            action = stream["commands"].copy()
        assert sha(directory/"experience.npz") == proof["experience_sha256"]
        assert sha(directory/"frozen_codes.npy") == proof["code_sha256"]
        codes = np.load(directory/"frozen_codes.npy")
        assert codes.shape == (17, 128) and action.shape == (16, 7)
        assert np.all(action[:, :3] == 0)
        current_parent = proof["parent"]["full_nn_sha256"]
        if parent is None:
            parent = current_parent
        assert current_parent == parent
        inputs.append(action[:, 3:6].astype(np.float64))
        targets.append((codes[1:]-codes[:-1]).astype(np.float64))
        current_xyz.append(action[:, :3].astype(np.float64)); kitchens.add(tuple(proof["kitchen"]))
        sources.append(dict(path=str(path),sha256=sha(path)))
    x = torch.from_numpy(np.concatenate(inputs)); y = torch.from_numpy(np.concatenate(targets))
    assert torch.linalg.matrix_rank(x) == 3 and torch.isfinite(y).all()
    # Closed-form minimizer of mean sample-level forward-effect squared error.
    response = torch.linalg.lstsq(x, y).solution.T
    assert response.shape == (128, 3) and torch.linalg.matrix_rank(response) == 3
    projection = torch.eye(128, dtype=torch.float64)-response@torch.linalg.pinv(response)
    assert torch.allclose(projection, projection.T, atol=1e-12, rtol=1e-12)
    assert torch.allclose(projection@projection, projection, atol=1e-12, rtol=1e-12)
    assert float((projection@response).norm()) < 1e-12*float(response.norm())
    output = root/"rotation-projection.pt"; assert not output.exists()
    torch.save(dict(projection=projection, rotation_response=response,
        parent_nn_sha256=parent, source_split="train"), output)
    pairs = root/"additional-factual-pairs.pt"; assert not pairs.exists()
    torch.save(dict(code_change=y, current_XYZ_input=torch.from_numpy(np.concatenate(current_xyz)),
        source_split="train", current_inputs_not_optimal_targets=True,
        parent_nn_sha256=parent, kitchens=sorted(kitchens)), pairs)
    predicted = x@response.T
    save(root/"rotation-projection.json", dict(passed=True, sha256=sha(output),
        parent_nn_sha256=parent, forward_parameters=384, transitions=len(x),
        current_angular_commands_are_inputs=True, targets_are_natural_next_code_changes=True,
        no_additional_XYZ_action_targets=True, no_body_updates=True,
        source_split="train", sources=sources, no_protected_evaluation_or_state_used=True,
        response_singular_values=torch.linalg.svdvals(response).tolist(),
        forward_code_RMSE=float((predicted-y).square().mean().sqrt()),
        zero_code_RMSE=float(y.square().mean().sqrt()),
        null_response_relative_error=float((projection@response).norm()/response.norm()),
        additional_factual_pairs_sha256=sha(pairs),
        same_additional_current_XYZ_labels_required_for_comparison=True,
        inference_is_sample_independent=True, not_control_evidence=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    args = parser.parse_args()
    fit_rotation(args.root, args.facts)
