"""Frozen representation/forecast checks on protected physical trajectories.

State and contact annotations are used only to stratify reported errors. They
never reach encoding, dynamics, a training objective or a planner proposal.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch

from experiments.robocasa_compare import evaluate as base
from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_refine.control import goal_support_cost
from experiments.robocasa_refine.validation import name


def diagnose(root, run, step, corpus, trajectories, output):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    base.install_public(root)
    adapter = base.SPTAdapter(root, run, step)
    from vforge.ops import OpsConfig, initialize_ops
    initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False), device=torch.device("cuda"), parallel_state=None)
    results = []
    for directory in sorted(trajectories.glob("*servo")):
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["diagnostic_only"] and summary["prohibited_training_or_proposal_memory"]
        trace = np.load(directory / "protected-trace.npz")
        case = corpus / "cases" / (name("push", summary["kitchen"], summary["seed"]) + ".npz")
        assert sha(case) == summary["case_sha256"]
        goal = np.load(case)["goal"]
        observations = [{"img_0": x[0], "img_1": x[1]} for x in trace["frames"]]
        features = torch.cat([adapter.encode(observations[i:i+4]) for i in range(0, len(observations), 4)], 1)
        target = adapter.encode([dict(img_0=goal[0], img_1=goal[1])])
        support = goal_support_cost(features[:, :1], target, adapter.model.state_slots, adapter.model.state_width)
        uniform_cost = (features - target).abs().mean((2, 3, 4, 5))[0].cpu().numpy()
        support_cost = -support(target, features).mean(-1).flatten().cpu().numpy()
        distance = np.asarray([x["goal_distance"] for x in summary["rows"]])
        forecasts = []
        for i, command in enumerate(trace["actions"]):
            first = max(0, i-1)
            old = trace["actions"][first:i]
            raw = base.control14(torch.as_tensor(old, device="cuda").float()[None])
            history = base.RollState(features[:, first:i+1], (raw-adapter.mean)/adapter.std)
            prediction, _ = adapter.nominal_step(torch.as_tensor(command, device="cuda").float()[None, None], history)
            forecasts.append(dict(mse=float((prediction-features[:, i+1:i+2]).square().mean()),
                uniform_predicted=float((prediction-target).abs().mean()),
                support_predicted=float(-support(target, prediction).mean()),
                robot_contact=summary["rows"][i]["robot_contact_ticks"] > 0))
        physical_best = int(distance.argmin())
        metrics = {}
        for label, costs in (("uniform", uniform_cost[1:]), ("support", support_cost[1:])):
            metrics[label] = dict(distance_spearman=float(spearmanr(costs, distance).statistic),
                representation_best_step=int(costs.argmin()+1), physical_best_step=physical_best+1,
                cost_at_physical_best=float(costs[physical_best]),
                physical_best_cost_percentile=float((costs <= costs[physical_best]).mean()),
                encoded_costs=costs.tolist())
        results.append(dict(trajectory=directory.name, case_sha256=sha(case),
            minimum_physical_distance=float(distance.min()), goal_geometry=metrics,
            contact_mse=[x["mse"] for x in forecasts if x["robot_contact"]],
            free_mse=[x["mse"] for x in forecasts if not x["robot_contact"]], forecasts=forecasts,
            distances=distance.tolist()))
    assert results and adapter.unchanged()
    result = dict(passed=True, run=run.name, step=step, no_SGD=True, model_unchanged=True,
        protected_diagnostic_only=True, state_not_in_model=True, rows=results,
        physical_cases=len({x["case_sha256"] for x in results}),
        repeated_trajectories_are_not_independent_cases=True,
        own_latent_errors_not_shared_physical_units=True)
    save(output, result)
    print(json.dumps(dict(passed=True, run=run.name, trajectories=len(results))), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("root", "run", "corpus", "trajectories", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--step", type=int, default=6000)
    a = p.parse_args()
    diagnose(a.root, a.run, a.step, a.corpus, a.trajectories, a.output)
