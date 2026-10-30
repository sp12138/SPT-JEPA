"""CPU-only matched review of full, quadrant and shuffled QueryPool gates."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .information_audit_summary import summarize, target_score
    from .spatial_probe_summary import DIMENSIONS, position_score
except ImportError:
    from information_audit_summary import summarize, target_score
    from spatial_probe_summary import DIMENSIONS, position_score


ARMS = ["full", "quadrants", "shuffled"]
COLORS = ["#64748b", "#2563eb", "#f97316"]


def read(path):
    return json.loads(path.read_text())


def initial_audit(root, out):
    import torch
    torch.set_num_threads(2)
    rows = []
    for task in ["tworoom", "cube"]:
        for seed in [1, 2]:
            base = root.parent / "mechanism_rate600_20261030" / f"{task}-fresh-rate-on-seed{seed}" / "initial-model/weights.pt"
            state = torch.load(base, map_location="cpu", weights_only=True, mmap=True)
            for arm in ARMS[1:]:
                candidate = root / f"{task}-{arm}-seed{seed}" / "initial-model/weights.pt"
                other = torch.load(candidate, map_location="cpu", weights_only=True, mmap=True)
                different = [k for k in state.keys() & other.keys() if not torch.equal(state[k], other[k])]
                row = dict(task=task, seed=seed, arm=arm, baseline=str(base), candidate=str(candidate),
                           tensors=len(state), key_sets_match=state.keys() == other.keys(), different_tensors=different)
                assert row["tensors"] == 550 and row["key_sets_match"] and not different
                rows.append(row)
                del other
            del state
    (out / "initial-tensor-audit.json").write_text(json.dumps(rows, indent=2) + "\n")
    hashes = {}
    for campaign in [root.parent / "mechanism_rate600_20261030", root]:
        tree = ast.parse((campaign / "source-v1/as_jepa_lewm/train.py").read_text())
        nodes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        optimizer = next(n for n in nodes["TrainModule"].body if isinstance(n, ast.FunctionDef) and n.name == "configure_optimizers")
        hashes[campaign.name] = {name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                                 for name, node in [("WindowSampler", nodes["WindowSampler"]), ("configure_optimizers", optimizer)]}
    assert len({json.dumps(v, sort_keys=True) for v in hashes.values()}) == 1
    (out / "sampler-optimizer-audit.json").write_text(json.dumps(hashes, indent=2) + "\n")


def collect(root, out):
    shared = root.parent
    spatial_root = shared / "mechanism_poolspatial_20261030"
    initial = read(out / "initial-tensor-audit.json")
    assert len(initial) == 8 and all(r["tensors"] == 550 and r["key_sets_match"] and not r["different_tensors"] for r in initial)
    sampler_optimizer = read(out / "sampler-optimizer-audit.json")
    assert len(sampler_optimizer) == 2 and len({json.dumps(v, sort_keys=True) for v in sampler_optimizer.values()}) == 1
    exits = [read(p) for p in sorted((root / "exits").glob("*.json"))]
    expected = {j["id"] for j in read(root / "plan.json")["jobs"]}
    assert len(exits) == len(expected) == 36 and {e["id"] for e in exits} == expected
    failures = {e["id"] for e in exits if e["returncode"] != 0}
    assert failures == {f"planning-cube-full-seed{s}" for s in [1, 2]}
    spatial_exits = [read(p) for p in sorted((spatial_root / "exits").glob("*.json"))]
    assert len(spatial_exits) == 16 and all(e["returncode"] == 0 for e in spatial_exits)
    rows, spatial_rows, signatures, projections = [], [], {}, None
    for task in ["tworoom", "cube"]:
        for seed in [1, 2]:
            base_id = f"{task}-fresh-rate-on-seed{seed}"
            base_run = shared / "mechanism_rate600_20261030" / base_id
            base_cfg = read(base_run / "config.json")
            base_p0_path = (shared / "mechanism_channels600_20261030" / f"planning-control-seed{seed}" / "metrics.json"
                            if task == "tworoom" else root / f"planning-cube-full-seed{seed}-eglretry" / "metrics.json")
            base_p0 = read(base_p0_path)
            for arm in ARMS:
                rid = f"{task}-{arm}-seed{seed}"
                run = base_run if arm == "full" else root / rid
                cfg = read(run / "config.json")
                normalized = dict(cfg, output=base_cfg["output"], model=dict(cfg["model"]))
                partition = normalized["model"].pop("pool_partition", "full")
                assert partition == arm and normalized == base_cfg, rid
                stage = read(run / "stage-complete.json")
                assert stage["initial_step"] == 0 and stage["stop_step"] == 600 and stage["scheduler_total_steps"] == 124910
                info_path = (shared / "mechanism_rateeval_20261030" / ("info-" + base_id) / "information-audit.json"
                             if arm == "full" else root / ("info-" + rid) / "information-audit.json")
                health_path = run.parent / ("health-" + run.name) / "health.json"
                health = read(health_path)
                assert health["audit"]["step"] == 600
                for k in ["model_finite", "optimizer_finite", "scheduler_finite", "cursor", "rng_present", "optimizer_present", "scheduler_present", "portable_equal"]:
                    assert health["audit"][k], (rid, k)
                p0 = base_p0 if arm == "full" else read(root / ("planning-" + rid) / "metrics.json")
                for k in ["pairs", "manifest_sha256", "solver", "plan", "normalization", "split", "protocol"]:
                    assert p0[k] == base_p0[k], (rid, k)
                assert p0["checkpoint_audit"]["strict_equal"]
                a, b = [np.asarray(x["metrics"]["episode_successes"], dtype=bool) for x in [base_p0, p0]]
                assert len(a) == len(b) == 50
                paired = dict(success_percent=float(100 * b.mean()), difference_pp=float(100 * (b.astype(float) - a).mean()),
                              gains=int((b & ~a).sum()), losses=int((a & ~b).sum()))
                row = summarize(info_path)
                row.update(id=rid, task=task, seed=seed, arm=arm, paired_against_full=paired, config=cfg, stage=stage,
                           health_audit=health["audit"], planning={k: p0[k] for k in ["metrics", "pairs", "manifest_sha256", "solver", "plan", "checkpoint", "protocol"]})
                assert row["step"] == 600
                info = read(info_path)
                signature = {k: info["probes"][k] for k in ["train_episodes", "validation_episodes", "label_keys"]}
                signatures.setdefault((task, "info"), signature)
                assert signatures[(task, "info")] == signature
                rows.append(row)
                for step in [0, 600]:
                    path = (shared / "mechanism_spatial_20261030" / f"{task}-seed{seed}-step{step}" / "spatial-probe.json"
                            if arm == "full" else spatial_root / f"{rid}-step{step}" / "spatial-probe.json")
                    sp = read(path)
                    sp.update(id=f"{rid}-step{step}", task=task, seed=seed, arm=arm, source=str(path))
                    assert sp["step"] == step and sp["projection_seed"] == 20261030
                    assert len(sp["train"]["episodes"]) == len(sp["validation"]["episodes"]) == 128
                    assert not set(sp["train"]["episodes"]).intersection(sp["validation"]["episodes"])
                    signature = {k: sp[k] for k in ["train", "validation"]}
                    signatures.setdefault((task, "spatial"), signature)
                    assert signatures[(task, "spatial")] == signature
                    if projections is None:
                        projections = sp["projections"]
                    assert sp["projections"] == projections and set(sp["features"]) == set(DIMENSIONS)
                    for name, dim in DIMENSIONS.items():
                        for kind, score in sp["features"][name].items():
                            n = 896 if kind == "temporal_displacement" else 1024
                            assert score["feature_dim"] == dim and score["ridge"] == .001
                            assert score["train_samples"] == score["validation_samples"] == n
                            assert score["solver"] == ("dual" if dim > n else "primal")
                    spatial_rows.append(sp)
    # At initialization only the pooling mask differs. Pre-pool features must agree.
    initial_prepool_max_difference = 0.
    for task in ["tworoom", "cube"]:
        for seed in [1, 2]:
            ss = {r["arm"]: r for r in spatial_rows if r["task"] == task and r["seed"] == seed and r["step"] == 0}
            for arm in ARMS[1:]:
                for feature in set(DIMENSIONS) - {"query_pool"}:
                    a = np.array(ss["full"]["features"][feature]["clean"]["validation"]["r2_against_train_mean"])
                    b = np.array(ss[arm]["features"][feature]["clean"]["validation"]["r2_against_train_mean"])
                    delta = float(np.max(np.abs(a - b)))
                    initial_prepool_max_difference = max(initial_prepool_max_difference, delta)
                    assert delta < 1e-8, (task, seed, arm, feature, delta)
    return dict(root=str(root), protocol=read(root / "protocol.json"), preflight=read(root / "PREFLIGHT-REVIEW.json"),
                egl_recovery=read(root / "EGL-RECOVERY.json"), exits=exits, spatial_exits=spatial_exits,
                initial_tensor_audit=initial, sampler_optimizer_source_audit=sampler_optimizer,
                all_configs_equal_except_output_and_pool_mask=True,
                all_planner_cases_and_settings_matched=True, all_probe_cases_and_projections_matched=True,
                initial_prepool_max_r2_difference=initial_prepool_max_difference, rows=rows, spatial=spatial_rows)


def plot(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for i, task in enumerate(["tworoom", "cube"]):
        for seed, style, marker in [(1, "-", "o"), (2, "--", "s")]:
            rr = {r["arm"]: r for r in result["rows"] if r["task"] == task and r["seed"] == seed}
            axes[i, 0].plot(range(3), [rr[a]["paired_against_full"]["success_percent"] for a in ARMS], style, marker=marker, label=f"Seed {seed}")
            for j, kind in enumerate(["clean", "noisy_sigma0.1"]):
                axes[i, 1].plot(range(3), [target_score(rr[a], kind) for a in ARMS], style, marker=marker, color=["#2563eb", "#dc2626"][j], label=f"{kind}, seed {seed}")
            for j, cond in enumerate(["forecast_other_initial", "forecast_wrong_actions"]):
                axes[i, 2].plot(range(3), [rr[a]["interventions"]["0.95"][cond]["motion_mse_increase_percent"] for a in ARMS], style, marker=marker, color=["#2563eb", "#f97316"][j], label=f"{'Wrong state' if j == 0 else 'Wrong actions'}, seed {seed}")
        for j, title in enumerate(["Planning success", "Position readout", "Renderer conditioning"]):
            ax = axes[i, j]
            ax.set_title(task.title() + ": " + title)
            ax.set_xticks(range(3), ARMS)
            ax.grid(alpha=.2)
        axes[i, 0].set_ylabel("Success % (50 paired cases)")
        axes[i, 0].set_ylim(0, 20)
        axes[i, 1].set_ylabel("Held-out R²: agent xy / block xyz")
        axes[i, 1].set_ylim(-.15, 1)
        axes[i, 2].set_yscale("symlog", linthresh=.001)
        axes[i, 2].set_ylabel("Motion-region MSE change % (symlog)")
    for ax in axes[0]:
        ax.legend(fontsize=7)
    fig.suptitle("Same-parameter pooling controls, all at 600 updates", fontsize=15)
    for ext in ["png", "svg"]:
        fig.savefig(out / f"pooling-controls.{ext}", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for i, task in enumerate(["tworoom", "cube"]):
        for j, kind in enumerate(["clean", "temporal_displacement"]):
            ax = axes[i, j]
            for arm, color in zip(ARMS, COLORS):
                for seed, style in [(1, "-"), (2, "--")]:
                    rr = sorted([r for r in result["spatial"] if r["task"] == task and r["seed"] == seed and r["arm"] == arm], key=lambda r: r["step"])
                    ax.plot([0, 600], [position_score(r, "query_pool", kind) for r in rr], style, color=color, marker="o")
            ax.set_title(task.title() + (": position" if j == 0 else ": displacement"))
            ax.set_xticks([0, 600])
            ax.set_xlabel("Update")
            ax.set_ylabel("QueryPool64 held-out R²")
            ax.set_ylim(-.2, 1)
            ax.grid(alpha=.2)
    handles = [Line2D([0], [0], color=c, label=a) for a, c in zip(ARMS, COLORS)]
    handles += [Line2D([0], [0], color="black", linestyle=s, label=f"Seed {n}") for n, s in [(1, "-"), (2, "--")]]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("Separate initial readout bias from learning", fontsize=15)
    for ext in ["png", "svg"]:
        fig.savefig(out / f"pooling-initialization.{ext}", dpi=170)
    plt.close(fig)


def report(result, out):
    lines = ["# Pooling address diagnostic: full versus quadrant versus shuffled", "",
        "All 34 intended training/health/information/planning endpoints succeeded. Two original Cube control planning attempts failed during EGL setup and remain preserved; their explicit retries supply the results. All 16 additional frozen spatial probes succeeded. Four full-pooling fresh-600 models, information audits and health checks are reused from the rate campaign, two TwoRoom planning controls from the channel campaign, and eight full-pooling spatial probes from the prior spatial campaign.", "",
        "All eight candidate initial models match all 550 baseline tensors exactly. Configurations match in their entirety after removing output path and the declared pooling partition. Therefore data path, data seed, batch size, worker settings, optimizer defaults and the 124910-step scheduler horizon are matched. An AST source hash comparison also verifies identical WindowSampler and configure_optimizers code in the frozen baseline and candidate snapshots. All stages start at zero and stop at 600. Every full checkpoint passes finite model/optimizer/scheduler, RNG, sample-cursor and portable-weight checks. Preflight gradient checks establish execution, not task competence.", "",
        "All planners use matched 50 cases and identical solver/horizon/settings. All diagnostic probes use identical task-specific disjoint 128 training/128 validation episodes and starts, fixed ridge .001 and the same projection matrices. Position labels train only frozen diagnostic probes, never model or controller objectives.", "",
        "**TwoRoom has a consistent small planning signal; Cube does not.** Full/quadrant/shuffled P0 is 6/12/6% for TwoRoom seed 1 and 8/16/6% for seed 2. Quadrants gain six and eight cases while losing three and four, respectively, against full pooling. Cube is 14/12/16% and 10/8/10%; quadrants lose two percentage points in both seeds. These paired counts support a task-specific early signal, not mature superiority or a universal pooling fix.", "",
        "## Planning and state readout at 600", "",
        "Position scores average agent xy for TwoRoom and block xyz for Cube. All seeds remain separate.", "",
        "| Task | Pool | Seed | P0 % | Paired delta pp | Gained/lost | Clean R² | Noise .1 R² | State std |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["rows"]:
        p = r["paired_against_full"]
        lines.append(f"| {r['task']} | {r['arm']} | {r['seed']} | {p['success_percent']:.0f} | {p['difference_pp']:+.0f} | {p['gains']}/{p['losses']} | {target_score(r,'clean'):.4f} | {target_score(r,'noisy_sigma0.1'):.4f} | {r['gradient']['state_std']['mean']:.6g} |")
    lines += ["", "## Initialization versus learning", "", "Only pooling address support differs at initialization; all pre-pooling probe R² vectors agree within 1e-8. These paired endpoints distinguish readout supplied by the architecture before training from later learning. Matched-dimensional spatial RP64 is retained as a reference; it is a fixed diagnostic projection, not the model's state.", "",
              "| Task | Pool | Seed | Update | Query position R² | Query displacement R² | Spatial RP64 position R² |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in result["spatial"]:
        lines.append(f"| {r['task']} | {r['arm']} | {r['seed']} | {r['step']} | {position_score(r,'query_pool'):.4f} | {position_score(r,'query_pool','temporal_displacement'):.4f} | {position_score(r,'spatial4x4_rp64'):.4f} |")
    lines += ["", "## Renderer conditioning at corruption .95", "", "Values are percent changes in motion-region MSE after replacing initial state, actions or visual anchor. The final column measures replacing clean posterior variation by its batch mean at t=1, sigma=0. These are fixed-model diagnostic interventions, not batch-coupled proposed training losses. They test the DiT renderer, not the sensitivity of the latent planning cost.", "",
              "| Task | Pool | Seed | Wrong initial % | Wrong actions % | Wrong anchor % | Clean constant posterior % |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in result["rows"]:
        v = r["interventions"]["0.95"]
        q = r["interventions"]["1.0"]
        values = [v[k]["motion_mse_increase_percent"] for k in ["forecast_other_initial", "forecast_wrong_actions", "forecast_wrong_anchor"]]
        values.append(q["scaled_posterior_scale0_sigma0"]["all_mse_increase_percent"])
        lines.append(f"| {r['task']} | {r['arm']} | {r['seed']} | " + " | ".join(f"{x:.6f}" for x in values) + " |")
    lines += ["", "## Gradient direction", "", "Positive scale derivatives mean infinitesimal shrinking lowers that fixed diagnostic loss; negative means expansion lowers it. A nonzero flow gradient need not amplify state distinctions. These are gradients with respect to the intermediate state, not full parameter-space training trajectories.", "",
              "| Task | Pool | Seed | Priced rate / flow norm | Gradient cosine | Flow temporal-scale derivative | Rate temporal-scale derivative |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in result["rows"]:
        keys = ["rate_to_flow", "cosine", "flow_temporal_scale_derivative", "rate_temporal_scale_derivative"]
        lines.append(f"| {r['task']} | {r['arm']} | {r['seed']} | " + " | ".join(f"{r['gradient'][k]['mean']:.6g}" for k in keys) + " |")
    lines += ["", "## Mechanism established by the controls", "",
        "**Spatial addressing supplies clean position readout before learning, and this readout survives the 600-step gate.** TwoRoom quadrant initialization is already agent xy R²=.892/.855 and .896/.849, versus shuffled .379/.460 and .424/.455 and full .130/.282 and .122/.297. At 600 the quadrant values are .886/.833 and .883/.846. This is primarily an architectural readout advantage; it is not evidence that the objectives learned stronger dynamics. Identical initial pre-pooling features and the shuffled support-size control localize the contrast to pooling address structure.", "",
        "**Clean accessibility is insufficient for renderer use.** TwoRoom quadrant states vary at std .000976/.001171, still far below posterior noise .1. Their noisy position probes remain negative. Wrong initial state or actions barely changes the renderer error, and even replacing clean posterior variation by a constant changes t=1 MSE by less than .00021% in magnitude. Thus fixing the clean linear readout bottleneck alone does not immediately establish the downstream state-to-renderer coupling. This result does not prove that a nonlinear readout could not extract other information.", "",
        "**The planning gain and renderer insensitivity can coexist.** The implemented planner uses clean encoding, latent rollout and goal-MSE cost, bypassing DiT. Better clean latent goal geometry or action ranking could therefore improve planning while the noisy renderer channel remains insensitive. These experiments have not isolated which part mediates the gain; an insensitive renderer cannot be used to conclude that the entire latent predictor or planner ignores state and actions. Conversely, better P0 alone does not prove more accurate outcome prediction.", "",
        "**The measured failure is not explained by rate gradients overwhelming flow gradients.** For TwoRoom quadrants the priced-rate/flow norm ratios are .029/.036, yet the flow gradient along temporal-state expansion has positive mean derivative in both seeds: locally it favors shrinking these distinctions. For Cube both flow and priced-rate derivatives along expansion are negative at this gate. The rate includes temporal prediction, so its local effect cannot universally be described as variance suppression. These local diagnostics constrain mechanisms; they do not establish a complete global optimization theory.", "",
        "**Cube shows a weaker form of the same separation at 600.** Quadrants improve clean block xyz readout, including height, while noisy readout remains near zero and replacing initial state does not worsen the measured forecast error. This early behavior must be interpreted alongside the earlier full-pooling 11671 checkpoint, where robust state use and strong position readout do emerge. It cannot support a claim that either architecture is permanently incapable.", "",
        "## Per-coordinate clean and noisy probes", "", "Coordinate order is agent xy, target xy for TwoRoom; block xyz, effector xyz for Cube. Complete temporal controls and all seven spatial feature families are preserved in results.json.", "",
              "| Task | Pool | Seed | Clean R² vector | Noise .1 R² vector |", "|---|---|---:|---|---|"]
    for r in result["rows"]:
        vals = [", ".join(f"{v:.4f}" for v in r["probes"][k]["r2_against_train_mean"]) for k in ["clean", "noisy_sigma0.1"]]
        lines.append(f"| {r['task']} | {r['arm']} | {r['seed']} | {vals[0]} | {vals[1]} |")
    lines += ["", "## Interpretation limits", "",
        "The partitioned pools have the same parameters and 64-dimensional output as full pooling. Each query sees one quarter of patches and the shared CLS token. The encoder already mixes information globally; these controls restrict pooling addresses, not the encoder's receptive field. Shuffled partitions match support size but cannot match every conditioning effect of spatial partitions. Their single fixed random partition is not a seed-tuned optimization.", "",
        "Linear decodability is neither information-theoretic sufficiency nor functional dynamics. Initialization gains are architectural readout gains. Noise robustness, action-conditioned state use, and task success must be assessed separately. Wrong-anchor interventions alter the whole visual context and cannot isolate agent position. Fixed-t denoising error is not generated-image quality.", "",
        "All models stop at 600, still within the 1249-update warmup. These results can identify immediate failures and representation changes but cannot establish permanent limits or mature benchmark superiority. The current report does not launch or authorize a continuation.", "",
        "![Pooling gate](pooling-controls.png)", "", "![Initialization control](pooling-initialization.png)", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-initial", action="store_true")
    parser.add_argument("--initial-only", action="store_true")
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.audit_initial:
        initial_audit(root, out)
    if args.initial_only:
        return
    result = collect(root, out)
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report(result, out)
    plot(result, out)
    print(json.dumps(dict(output=str(out), models=len(result["rows"]), spatial=len(result["spatial"]))))


if __name__ == "__main__":
    main()
