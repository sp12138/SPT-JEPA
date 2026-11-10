"""Bounded original-data / fork-data / shared-root factorization comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import subprocess
import time

import torch
from torch.utils.data import DataLoader

from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_compare.campaign import command, environment, TRAIN_PYTHON
from experiments.robocasa_compare.train import (build, make_dataset, make_optimizer, associations,
    prepare_batch, state_hash, save_full_state, gradients)
from experiments.robocasa_refine import forks

ARMS = {"dev13": "ordinary-flat", "dev15": "fork-flat", "dev6": "fork-shared"}


def proof(root):
    """Actual TRAIN branch images; numerical objective identity, no optimizer."""
    plan = json.loads((root/"fork-plan.json").read_text())
    old = json.loads((Path(plan["source"])/"interventions/dataset.json").read_text())
    complete = []
    for group in plan["groups"]:
        paths = [root/"transitions"/(group["id"]+f"-b{b}.json") for b in range(8)]
        if group["split"] == "train" and all(p.exists() for p in paths):
            rows = [json.loads(p.read_text()) for p in paths]
            complete.append((group, rows))
    assert complete, "No complete factual TRAIN fork yet"
    group, rows = complete[0]
    card = dict(old, records=rows, groups=[dict(group, record_ids=[x["id"] for x in rows])],
                branches=8, fixed_start=0)
    directory = root/"tree-proof-data"; save(directory/"dataset.json", card)
    source = Path(plan["source"])/"training/issued-seed1"
    cfg = json.loads((source/"config.json").read_text())
    flat_config = dict(cfg["model"], analytic_prior=True)
    flat = build(flat_config, root/"code").cuda().eval()
    tree = build(dict(flat_config, shared_root_branches=8), root/"code").cuda().eval()
    weights = torch.load(source/"step-006000/weights.pt", map_location="cpu", weights_only=True)
    flat.load_state_dict(weights, strict=True); tree.load_state_dict(weights, strict=True)
    before = state_hash(weights)
    from vforge.ops import OpsConfig, initialize_ops
    initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False), device=torch.device("cuda"), parallel_state=None)
    cfg.update(fork_groups=True, model=flat_config)
    raw = next(iter(DataLoader(make_dataset(cfg, directory), batch_size=1, num_workers=0)))
    assert set(raw) == {"pixels", "action"} and raw["pixels"].shape[1:4] == (8, 2, 16)
    batch = prepare_batch(raw, torch.device("cuda"))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.manual_seed(451); a, am = flat.loss(batch, 6000)
        torch.manual_seed(451); b, bm = tree.loss(batch, 6000)
        from vforge.models.as_jepa.constants import FRAME_ELEMS
        expected = tree.beta*(1-1/8)*am["entry"]/((tree.window-1)*FRAME_ELEMS)
        error = float((a-b-expected).abs())
        assert error < 1e-6, error
        poisoned = dict(batch, action=batch["action"].clone())
        poisoned["action"][:, -1] = 100.
        torch.manual_seed(451); changed, _ = tree.loss(poisoned, 6000)
        assert float((changed-b).abs()) < 1e-6
    tree.train(); tree.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16): loss, _ = tree.loss(batch, 6000)
    loss.backward(); norms = gradients(tree)
    assert torch.isfinite(loss) and all(v > 0 and __import__("math").isfinite(v) for v in norms.values())
    assert state_hash(flat.state_dict()) == state_hash(tree.state_dict()) == before
    save(root/"TREE_APPROVED.json", dict(passed=True, actual_factual_branches=8, source_fork=group["id"],
        shared_entry_loss_identity_error=error, terminal_action_poison_error=float((changed-b).abs()),
        gradients=norms, all_parameters_exact_parent=True, no_new_neural_parameters=True,
        no_SGD=True, no_future_action_target=True, sample_unit="One physical initial state with8 factual branches",
        conditional_transition_and_image_terms_unchanged=True, not_a_joint_ELBO_claim=True))


def initialize(root, arms=None):
    assert json.loads((root/"TREE_APPROVED.json").read_text())["passed"]
    plan = json.loads((root/"fork-plan.json").read_text()); source = Path(plan["source"])
    arms = ARMS if arms is None else arms
    assert not (root/"INITIALIZED.json").exists()
    records = []
    for host, arm in arms.items():
        grouped = arm != "ordinary-flat"
        for seed in (1, 2):
            parent = source/"training"/f"issued-seed{seed}"
            old = torch.load(parent/"last.ckpt", map_location="cpu", weights_only=False)
            assert old["step"] == 6000
            cfg = json.loads((parent/"config.json").read_text())
            cfg.update(arm=arm, host=host, parent=str(parent), global_batch=8 if grouped else 64,
                fork_groups=grouped, phase_start=6000, max_steps=7000, stages=[6000, 6002, 7000],
                data=str(root/"interventions" if grouped else source/"interventions"))
            cfg["model"].update(analytic_prior=True, shared_root_branches=8 if arm == "fork-shared" else 0)
            run = root/"training"/f"{arm}-seed{seed}"
            if run.exists():
                assert json.loads((run/"config.json").read_text()) == cfg
                assert json.loads((run/"ancestry.json").read_text())["parent_sha256"] == sha(parent/"last.ckpt")
                records.append(dict(run=run.name, host=host, grouped=grouped, existing_state_untouched=True))
                del old
                continue
            run.mkdir(parents=True, exist_ok=False)
            save(run/"config.json", cfg)
            model = build(cfg["model"], root/"code"); model.load_state_dict(old["model"], strict=True)
            assert state_hash(model.state_dict()) == old["nn_sha256"]
            opt, scheduler = make_optimizer(model, 7000, 6000)
            assert associations(model, opt) == old["parameter_associations"]
            opt.load_state_dict(old["optimizer"])
            sched = copy.deepcopy(old["scheduler"])
            fresh = [v*f(6000) for v, f in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
            sched["_last_lr"] = fresh; scheduler.load_state_dict(sched)
            for group, lr in zip(opt.param_groups, fresh): group["lr"] = lr
            state = dict(old, model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                samples=6000*cfg["global_batch"], config_sha256=sha(run/"config.json"))
            save_full_state(run, state)
            save(run/"ancestry.json", dict(passed=True, parent=str(parent), parent_sha256=sha(parent/"last.ckpt"),
                all_neural_tensors_equal=len(old["model"]), all_optimizer_RNG_states_preserved=True,
                old_cursor_samples=old["samples"], new_cursor_units=state["samples"],
                cursor_unit="8-branch groups" if grouped else "trajectories",
                cursor_conversion="Exact divide8 for grouped data; new corpus and sampling units disclosed" if grouped else "unchanged",
                camera_images_per_update=2048, new_updates=1000, original_scratch_unsaved_replay=1,
                independent_pretraining=False, new_learned_parameters=0))
            records.append(dict(run=run.name, host=host, grouped=grouped))
            del model, opt, scheduler, state, old
    marker = "INITIALIZED.json" if len(arms) == len(ARMS) else "INITIALIZED-ordinary-flat.json"
    save(root/marker, dict(passed=True, records=records, new_SGD=0,
        existing_ordinary_arm_may_already_be_training=len(arms) == len(ARMS), automatic_long_training=False))


def coordinator(root):
    deadline = time.time()+4*3600
    while not (root/"TREE_APPROVED.json").exists():
        assert time.time() < deadline
        plan = json.loads((root/"fork-plan.json").read_text())
        ready = any(g["split"] == "train" and all((root/"transitions"/(g["id"]+f"-b{b}.json")).exists()
                     for b in range(8)) for g in plan["groups"])
        if ready:
            command([TRAIN_PYTHON, "-u", "-m", "experiments.robocasa_refine.fork_phase", "--root", str(root), "--mode", "proof"],
                    root/"tree-proof.log", environment(root, "7"), root/"code", 1800)
        else: time.sleep(10)
    # This scaling control uses only the existing corpus; run it during the
    # remaining physical acquisition, then retain it unchanged for comparison.
    if not (root/"INITIALIZED-ordinary-flat.json").exists():
        initialize(root, {"dev13": ARMS["dev13"]})
    while not all((root/f"COLLECTION_DONE-{h}.json").exists() for h in ARMS):
        assert time.time() < deadline; time.sleep(10)
    forks.finish(root); initialize(root)
    end = time.time()+3*3600
    runs = [root/"training"/f"{arm}-seed{s}" for arm in ARMS.values() for s in (1, 2)]
    while not all((r/"EVALUATION_COMPLETE.json").exists() or (r/"ERROR.json").exists() for r in runs):
        assert time.time() < end; time.sleep(10)
    rows = []
    for run in runs:
        if (run/"ERROR.json").exists():
            rows.append(dict(run=run.name, execution_error=json.loads((run/"ERROR.json").read_text())))
            continue
        for metric in ("uniform", "direction"):
            values = [json.loads(p.read_text()) for p in (root/"source-evaluation"/run.name/metric).glob("*/summary.json")]
            assert len(values) == 4 and all(x["passed"] and x["model_unchanged"] for x in values)
            rows.append(dict(run=run.name, metric=metric,
                means={t: sum(x["score"] for x in values if x["task"] == t)/2 for t in ("reach", "push")},
                strict_completions=sum(x["final_task_complete"] for x in values)))
    save(root/"results/summary.json", dict(disposed=True, rows=rows,
        independent_physical_cases=4, independent_scratch_ancestors=2,
        source_development_only=True, same_neural_parent_and_images_per_update=True,
        data_BC_and_sampling_identical=True, data_A_differs=True,
        candidate_loss_unit="One8-branch physical experiment; no coupling across independent worlds",
        automatic_long_training=False, not_a_SOTA_claim=True))


def worker(root, host):
    deadline = time.time()+4*3600
    def initialized():
        return (root/"INITIALIZED.json").exists() or (host == "dev13" and (root/"INITIALIZED-ordinary-flat.json").exists())
    while not initialized():
        assert not (root/"ERROR-coordinator.json").exists(), "Data/proof/initialization gate failed"
        assert time.time() < deadline; time.sleep(10)
    source = Path(json.loads((root/"fork-plan.json").read_text())["source"])
    cohort = json.loads((root/"VALIDATION_CASES_READY.json").read_text())["selected"]
    def one(seed):
        run = root/"training"/f"{ARMS[host]}-seed{seed}"
        gpus = list(range(4*(seed-1), 4*seed)); gpu_list = ",".join(map(str, gpus))
        try:
            memory = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
            assert all(int(memory.splitlines()[g]) > 120*1024 for g in gpus), "Insufficient free GPU memory before training"
            for step in (6000, 6002, 7000):
                if step > 6000:
                    command([TRAIN_PYTHON, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=4",
                        "-m", "experiments.robocasa_compare.train", "--root", str(root), "--run-directory", str(run),
                        "--stop-step", str(step), "--resume"], run/f"train-{step}.log", environment(root, gpu_list), root/"code", 3600)
                command([TRAIN_PYTHON, "-m", "experiments.robocasa_compare.audit", "--root", str(root), "--run", str(run), "--step", str(step)],
                        run/f"audit-{step}.log", environment(root, str(gpus[0])), root/"code", 1800)
                if step == 6002:
                    c = cohort[0]
                    argv = [str(root/"environment/bin/python"), "-u", "-m", "experiments.robocasa_refine.validation", "--root", str(source),
                        "--mode", "evaluate", "--run", str(run), "--step", "6002", "--technical", "--history-controls", "issued",
                        "--output", str(root/"entry"/run.name)]
                    for k, v in c.items(): argv.extend(["--"+k, str(v)])
                    command(argv, run/"entry.log", environment(root, str(gpus[0]), True), root/"code", 900)
            save(run/"TRAINING_COMPLETE.json", dict(passed=True, step=7000, new_updates=1000))
            for metric in ("uniform", "direction"):
                def evaluate(item):
                    i, c = item; label = f"{c['task']}-{c['kitchen']}-{c['seed']}"
                    out = root/"source-evaluation"/run.name/metric/label
                    argv = [str(root/"environment/bin/python"), "-u", "-m", "experiments.robocasa_refine.validation", "--root", str(source),
                        "--mode", "evaluate", "--run", str(run), "--step", "7000", "--history-controls", "issued",
                        "--goal-metric", metric, "--output", str(out)]
                    for k, v in c.items(): argv.extend(["--"+k, str(v)])
                    command(argv, run/("eval-"+metric+"-"+label+".log"), environment(root, str(gpus[i]), True), root/"code", 3600)
                with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(evaluate, enumerate(cohort)))
            save(run/"EVALUATION_COMPLETE.json", dict(passed=True, source_cases=4, metrics=["uniform", "direction"], automatic_extension=False))
        except BaseException as error:
            save(run/"ERROR.json", dict(error=repr(error), method_negative=False)); raise
    with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(one, (1, 2)))
    save(root/f"DONE-{host}.json", dict(passed=True, automatic_extension=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", choices=["coordinator", "proof", "worker"], required=True); p.add_argument("--host", choices=ARMS)
    a = p.parse_args()
    try:
        if a.mode == "proof": proof(a.root)
        elif a.mode == "coordinator": coordinator(a.root)
        else: worker(a.root, a.host)
    except BaseException as error:
        save(a.root/("ERROR-"+a.mode+("-"+a.host if a.host else "")+".json"),
             dict(error=repr(error), method_negative=False))
        raise
