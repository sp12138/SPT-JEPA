"""CPU-only S3 review from the immutable main protocol and endpoint reports.

Partial reports are explicit. --require-complete rejects missing/failed S3
endpoints. Optional --derive-support rereads only the audit's existing windows,
without an encoder, optimizer, physical labels, or any GPU operation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

ARMS = ("symbol", "completion", "transform")
TASKS = ("tworoom", "cube")
CONDITIONS = ("clean_rollout", "zero_prediction", "observed_future", "other_initial",
              "wrong_action_order", "other_episode_actions", "teacher_forced_prefix",
              "initial_carry", "shifted_observed_carry")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def optional(path):
    return read(path) if path.exists() else None


def derive_support(cfg, audit, audit_dir):
    """Independent exact-label check, using no model or future-window expansion."""
    import numpy as np
    import torch
    from torch.nn import functional as F
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    torch.set_num_threads(2)
    if cfg["model"].get("domain", "standard") != "standard":
        raise ValueError("Support reconstruction is explicitly for standard RGB domain")
    manifest = read(audit_dir / "windows.json")
    if digest(audit_dir / "windows.json") != audit["manifest_sha256"]:
        raise AssertionError("Window manifest hash mismatch")
    dataset = BenchmarkWindows(cfg["data"], window=audit["window"], split="validation")
    for name, expected in manifest["metadata_sha256"].items():
        assert digest(Path(cfg["data"]) / name) == expected, name
    assert digest(Path(cfg["data"]) / "dataset_card.json") == manifest["dataset_card_sha256"]
    train = set(np.load(Path(cfg["data"]) / "train_episodes.npy").tolist())
    queries = audit["path_queries"]
    indices = torch.tensor(queries["query_indices"], dtype=torch.long)
    labels, symbols = [], []
    maps = [(0, 1, 2), (1, 1, 2), (0, 2, 2), (2, 2, 2), (1, 2, 2)]
    def transfer(word):
        result = []
        for start in range(3):
            progress = start
            for letter in word:
                if progress == 0 and letter == 0:
                    progress = 1
                elif progress == 1 and letter == 1:
                    progress = 2
            result.append(progress)
        return maps.index(tuple(result))
    for expected in manifest["windows"]:
        batch = dataset[expected["index"]]
        assert batch["episode"] == expected["episode"] and batch["rows"].tolist() == expected["rows"]
        assert batch["episode"] not in train
        render = batch["render"].float() / 127.5 - 1
        psi = F.adaptive_avg_pool2d(render, (8, 8)).flatten(1)
        prototype = psi[indices]
        words = (psi[None, :, None] - prototype[:, None]).square().sum(-1).argmin(-1)
        split = words.shape[-1] // 2
        symbols.extend(words.tolist())
        labels.extend([[transfer(word), transfer(word[:split]), transfer(word[split:])]
                       for word in words.tolist()])
    labels = np.asarray(labels, dtype=int)
    count = lambda x, k: np.bincount(np.asarray(x).reshape(-1), minlength=k).tolist()
    totals = dict(symbols=count(symbols, 3), transforms=count(labels, 5), completion=count(labels == 3, 2))
    for name, value in totals.items():
        assert value == queries["label_support"][name]["counts"], (name, value, queries["label_support"][name])
    return dict(source="Independent CPU reconstruction of the exact existing audit RGB windows",
        manifest_sha256=audit["manifest_sha256"], query_indices=queries["query_indices"],
        examples=len(manifest["windows"]), queries_per_clip=len(indices), totals=totals,
        segments={name:dict(transforms=count(labels[:, i], 5), completion=count(labels[:, i] == 3, 2))
                  for i, name in enumerate(("full", "first_half", "second_half"))},
        aggregate_counts_match_frozen_audit=True)


def collect(root, require_complete=False, derive=False):
    protocol, plan = read(root / "protocol.json"), read(root / "plan.json")
    records = [r for r in protocol["runs"] if r["family"] == "pathquery"]
    expected = {(task, arm, seed) for task in TASKS for arm in ARMS for seed in (1, 2)}
    assert {(r["task"], r["arm"], r["seed"]) for r in records} == expected
    jobs = [j for j in plan["jobs"] if j.get("family") == "pathquery"]
    statuses = {j["id"]:optional(root / "exits" / (j["id"] + ".json")) for j in jobs}
    pending = [k for k, v in statuses.items() if v is None]
    failed = [k for k, v in statuses.items() if v is not None and v["returncode"] != 0]
    rows, support_cache, signatures = [], {}, {}
    for record in records:
        rid = record["id"]
        cfg = read(root / rid / "config.json")
        assert cfg["model"]["path_query_mode"] == record["arm"]
        assert cfg["model"]["path_query_weight"] == 1.
        assert digest(root / rid / "config.json") == record["config_sha256"]
        p0 = optional(root / ("planning-" + rid) / "metrics.json")
        audit_dir = root / ("transition-" + rid)
        audit = optional(audit_dir / "state-audit.json")
        health = optional(root / ("health-" + rid) / "health.json")
        pre = optional(Path(protocol["preflight"]) / ("gradient-" + rid) / "preflight.json")
        row = dict(**record, planning=p0, transition=audit, preflight=pre,
                   health_audit=health.get("audit") if health else None,
                   stage=optional(root / rid / "stage-complete.json"))
        if p0:
            assert p0["checkpoint_audit"]["strict_equal"]
            success = p0["metrics"]["episode_successes"]
            assert len(success) == len(p0["pairs"]) == 50
            row["p0_percent"] = sum(bool(x) for x in success) / len(success) * 100
            key = (record["task"], "planning")
            signature = {k:p0[k] for k in ("pairs", "manifest_sha256", "solver", "plan", "normalization", "split", "protocol")}
            signatures.setdefault(key, signature)
            assert signatures[key] == signature, (rid, "Planning contract differs")
        if audit:
            assert audit["step"] == protocol["stop_step"] == 600
            assert audit["config_sha256"] == row["config_sha256"]
            query = audit["path_queries"]
            assert query["trained_mode"] == record["arm"]
            key = (record["task"], "query")
            signature = {"manifest":audit["manifest_sha256"], **{k:query[k] for k in ("query_step", "query_indices", "label_support")}}
            signatures.setdefault(key, signature)
            assert signatures[key] == signature, (rid, "Query targets/cases differ")
            trained = {name:query["conditions"][name][record["arm"]] for name in CONDITIONS}
            row["trained_query"] = trained
            clean = trained["clean_rollout"]["relative_to_uniform_ce"]
            row["query_delta_vs_forecast"] = {name:value["relative_to_uniform_ce"]-clean for name, value in trained.items()}
            if derive:
                if record["task"] not in support_cache:
                    support_cache[record["task"]] = derive_support(cfg, audit, audit_dir)
                assert support_cache[record["task"]]["totals"] == {k:v["counts"] for k,v in query["label_support"].items()}
        if pre:
            assert pre["passed"] and pre["checkpoint"]["step"] == 2
            # Native preflight compares five parameter prefixes (547 tensors),
            # not necessarily every buffer in the full 550-tensor state dict.
            assert pre["paired_initial_equal"]["equal"] and pre["paired_initial_equal"]["compared_tensors"] > 0
        rows.append(row)
    pairs = []
    for task in TASKS:
        for seed in (1, 2):
            matched = {r["arm"]:r for r in rows if r["task"] == task and r["seed"] == seed}
            for first, second in itertools.combinations(ARMS, 2):
                a, b = matched[first], matched[second]
                if not a["planning"] or not b["planning"]:
                    continue
                x, y = [r["planning"]["metrics"]["episode_successes"] for r in (a, b)]
                pairs.append(dict(task=task, seed=seed, first=first, second=second, n=len(x),
                    delta_pp=b["p0_percent"]-a["p0_percent"],
                    gained=sum(bool(v) and not bool(u) for u,v in zip(x,y)),
                    lost=sum(bool(u) and not bool(v) for u,v in zip(x,y))))
    complete = not pending and not failed and all(r["planning"] and r["transition"] and r["health_audit"] for r in rows)
    if require_complete and not complete:
        raise AssertionError(dict(pending=pending, failed=failed, complete=complete))
    return dict(root=str(root), source_commit=protocol["source_commit"], complete=complete,
                expected_endpoints=len(jobs), pending=pending, failed=failed,
                runs=rows, paired_planning=pairs, segment_support=support_cache,
                scope="Both seeds retained. Only trained-mode query CE is objective-matched. Different target losses are not a method ranking; P0 is the native planning metric. 600 steps remain inside warmup.")


def fmt(value, places=3):
    return "pending" if value is None else f"{value:.{places}f}"


def write_markdown(result, out):
    lines = ["# S3 three-arm review", "", f"Source `{result['source_commit']}`; complete={result['complete']}. "
        f"Pending endpoints {len(result['pending'])}, failed {len(result['failed'])}.", "",
        "Every row keeps its actual training seed. CE/log(K) has uniform reference one, but target entropy, class balance and effective active rows differ; do not rank methods by cross-arm CE.", "",
        "| Task | Arm | Seed | Native P0 % | Forecast CE/logK | Query-only | Observed future | Wrong initial delta | Wrong action order delta | Other actions delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    ordered = sorted(result["runs"], key=lambda r:(TASKS.index(r["task"]), ARMS.index(r["arm"]), r["seed"]))
    for row in ordered:
        q, delta = row.get("trained_query", {}), row.get("query_delta_vs_forecast", {})
        ce = lambda key:q.get(key, {}).get("relative_to_uniform_ce")
        values = [row["task"], row["arm"], str(row["seed"]), fmt(row.get("p0_percent"), 1),
                  *[fmt(ce(k)) for k in ("clean_rollout", "zero_prediction", "observed_future")],
                  *[fmt(delta.get(k)) for k in ("other_initial", "wrong_action_order", "other_episode_actions")]]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "Query-only sets the whole state sequence to zero and retains real query prototypes and context masks. It may move the head off its training support. Observed future is privileged diagnostic access, not a forecast. Intervention deltas subtract the same trained head's clean forecast CE; positive means worse classification after intervention. These fixed-target swaps test conditioning dependence, not accuracy on unobserved counterfactual outcomes.", "",
              "## Same-case planning comparisons", "", "| Task | Seed | First → second | Delta pp | Gained | Lost |", "|---|---:|---|---:|---:|---:|"]
    for pair in result["paired_planning"]:
        lines.append(f"| {pair['task']} | {pair['seed']} | {pair['first']} → {pair['second']} | {pair['delta_pp']:+.1f} | {pair['gained']} | {pair['lost']} |")
    lines += ["", "## Full and half support", "", "Map order: I, A, B, AB, BA. Completion order: no, yes.", "",
              "| Task | Segment | Map counts | Completion counts |", "|---|---|---|---|"]
    for task, support in result["segment_support"].items():
        for segment, counts in support["segments"].items():
            lines.append(f"| {task} | {segment} | {counts['transforms']} | {counts['completion']} |")
    if not result["segment_support"]:
        lines += ["", "Per-segment support not reconstructed yet. Frozen reports contain only aggregate counts; use --derive-support to recover and independently verify the existing audit targets on CPU."]
    lines += ["", "## Native two-step startup", "", "| Task | Arm | Passed | Compared original tensors | Action gradient norm | Head gradient norm |",
              "|---|---|---|---:|---:|---:|"]
    for row in ordered:
        pre = row["preflight"]
        if pre:
            lines.append(f"| {row['task']} | {row['arm']} | {pre['passed']} | {pre['paired_initial_equal']['compared_tensors']} | {pre['gradients']['action_encoder']['gradient_norm']:.3g} | {pre['gradients']['path_query_head']['gradient_norm']:.3g} |")
    lines += ["", "600 updates remain inside the 1249-update warmup. Two seeds and 50 paired validation cases per task do not establish mature superiority or permanent failure. Better classification with observed future codes is not evidence that the causal predictor uses that information. Full raw reports, cases, manifest hashes and all horizons remain in results.json.", "",
              "![Native planning](planning.png)", "", "![Query dependence within each trained target](query-dependence.png)", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def plot(result, out):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {(r["task"],r["arm"],r["seed"]):r for r in result["runs"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, task in zip(axes, TASKS):
        for seed, shift, color in [(1,-.18,"#2563eb"),(2,.18,"#f97316")]:
            values = [lookup[task,arm,seed].get("p0_percent",float("nan")) for arm in ARMS]
            bars = ax.bar(np.arange(3)+shift, values, .34, color=color, label=f"seed {seed}")
            ax.bar_label(bars, fmt="%.0f", fontsize=9, padding=3)
        ax.set_xticks(np.arange(3), ARMS);ax.set_title(task);ax.set_ylabel("Native success (%) / 50 fixed cases")
        ax.set_ylim(bottom=0);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=9)
    fig.savefig(out/"planning.png",dpi=170);plt.close(fig)
    names = ("zero_prediction","observed_future","other_initial","wrong_action_order","other_episode_actions","teacher_forced_prefix","shifted_observed_carry")
    labels = ("Query\nonly","Observed\nfuture","Wrong\ninitial","Action\norder","Other\nactions","Teacher\nprefix","Observed\ncarry")
    fig, axes = plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
    for ti, task in enumerate(TASKS):
        for ai, arm in enumerate(ARMS):
            ax=axes[ti,ai]
            for seed, color, marker in [(1,"#2563eb","o"),(2,"#f97316","s")]:
                row=lookup[task,arm,seed];delta=row.get("query_delta_vs_forecast",{})
                ax.plot(range(len(names)),[delta.get(n,float("nan")) for n in names],marker=marker,color=color,label=f"seed {seed}",linewidth=1)
            ax.axhline(0,color="black",linewidth=.7);ax.set_xticks(range(len(names)),labels,fontsize=8)
            ax.set_title(f"{task}: trained {arm}");ax.set_ylabel("CE/logK minus clean forecast")
            ax.grid(axis="y",alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Within-head interventions; cross-target CE is not a method ranking",fontsize=12)
    fig.savefig(out/"query-dependence.png",dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("/efs/prescottsun/SPT-JEPA/runs/ideas_validate600_20261031"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--derive-support",action="store_true")
    parser.add_argument("--require-complete",action="store_true")
    parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    result=collect(args.root,args.require_complete,args.derive_support)
    (args.output/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    write_markdown(result,args.output)
    if not args.no_plots:plot(result,args.output)
    print(json.dumps(dict(complete=result["complete"],pending=len(result["pending"]),failed=result["failed"],output=str(args.output))))


if __name__ == "__main__":
    main()
