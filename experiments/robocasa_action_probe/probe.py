"""Detached current-input action probes; ordinary factual logs, no expert targets.

Fit a linear inverse on observed RGB transitions. At control time the second
image is a goal, a distribution shift evaluated explicitly in the simulator.
The consequence learners are immutable; no action-label gradient reaches them.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_compare import train as common

MODULE = "experiments.robocasa_action_probe.probe"
TRAIN_PYTHON = "/efs/prescottsun/project_jepa/envs/lewm/bin/python"
HOSTS = {"dev13": "vjepa-linear", "dev15": "global16-linear", "dev6": "spt64-linear"}


class Readout(nn.Module):
    def __init__(self, root, arm):
        super().__init__(); self.arm = arm
        common.install_runtime(root/"code")
        self.parent = None
        if arm != "vjepa-linear":
            from vforge.models.as_jepa.pooling import QueryPool
            plan = json.loads((root/"plan.json").read_text())
            run = Path(plan["source"])/"training"/plan["source_models"][arm]
            label = plan["comparison_checkpoint"]
            ready = json.loads((run/label/"ready.json").read_text())
            path = run/label/"weights.pt"
            assert ready["passed"] and sha(path) == ready["weights_sha256"]
            cfg = json.loads((run/"config.json").read_text())
            self.feature_adapter = nn.Sequential(nn.Linear(1664, 192), nn.LayerNorm(192))
            self.pool = QueryPool(192, 3, cfg["state_slots"], 16)
            weights = torch.load(path, map_location="cpu", weights_only=True)
            for prefix, module in (("feature_adapter.", self.feature_adapter), ("pool.", self.pool)):
                module.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)}, strict=True)
            self.parent = dict(run=str(run), label=label, step=ready["step"],
                weights_sha256=ready["weights_sha256"], full_nn_sha256=ready["nn_sha256"])
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, features):
        if self.arm == "vjepa-linear":
            return features.float().mean(-2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = self.pool(self.feature_adapter(features.float()))
        return result.flatten(-2).float()


def cache(root, arm, shard):
    torch.set_num_threads(2); torch.set_float32_matmul_precision("high")
    readout = Readout(root, arm).cuda()
    before = common.state_hash(readout.state_dict())
    source = Path("/tmp/spt-time-compare-20261112")
    records = json.loads((root/"data.json").read_text())["records"]
    rows, codes, actions, splits = [], [], [], []
    for index, record in enumerate(records):
        if index % 8 != shard: continue
        rid = record["id"]
        dense = np.load(source/"features/vjepa"/(rid+".npy"), mmap_mode="r")
        assert dense.shape == (17, 2, 196, 1664)
        features = torch.from_numpy(np.asarray(dense).copy()).flatten(0, 1).cuda()
        code = readout(features).reshape(17, 2, -1).flatten(1).detach().cpu().numpy()
        with np.load(source/"observations"/(rid+".npz")) as stream:
            assert set(stream.files) == {"render", "commands"}
            action = stream["commands"][:, :3].astype(np.float32)
        assert code.shape[0] == 17 and action.shape == (16, 3)
        assert np.isfinite(code).all() and np.isfinite(action).all()
        rows.append(index); codes.append(code); actions.append(action); splits.append(record["split"])
    assert common.state_hash(readout.state_dict()) == before
    assert all(not p.requires_grad and p.grad is None for p in readout.parameters())
    directory = root/"heads"/arm/"cache"; directory.mkdir(parents=True, exist_ok=True)
    output = directory/f"part-{shard}.npz"
    assert not output.exists()
    np.savez_compressed(output, indices=np.asarray(rows), codes=np.stack(codes),
                        actions=np.stack(actions), splits=np.asarray(splits))
    save(output.with_suffix(".json"), dict(passed=True, shard=shard, records=len(rows),
        sha256=sha(output), readout_sha256=before, parent=readout.parent,
        all_features_detached=True, readout_frozen_unchanged=True, current_actions_only=True))


def fit(root, arm):
    torch.set_num_threads(2)
    directory = root/"heads"/arm
    rows, codes, actions, splits = [], [], [], []
    for shard in range(8):
        path = directory/"cache"/f"part-{shard}.npz"
        proof = json.loads(path.with_suffix(".json").read_text())
        assert proof["passed"] and sha(path) == proof["sha256"]
        with np.load(path) as s:
            rows.extend(s["indices"].tolist()); codes.extend(s["codes"])
            actions.extend(s["actions"]); splits.extend(s["splits"].tolist())
    assert sorted(rows) == list(range(1216)) and len(set(rows)) == len(rows)
    order = np.argsort(rows)
    z = np.stack(codes)[order]; y = np.stack(actions)[order]
    splits = np.asarray(splits)[order]
    card = json.loads((root/"data.json").read_text())
    train_kitchens = {tuple(r["kitchen"]) for r in card["records"] if r["split"] == "train"}
    val_kitchens = {tuple(r["kitchen"]) for r in card["records"] if r["split"] == "validation"}
    assert not train_kitchens & val_kitchens
    assert np.sum(splits == "train") == 1120 and np.sum(splits == "validation") == 96
    x = torch.from_numpy(z[:, 1:]-z[:, :-1]).double().cuda()
    y = torch.from_numpy(y).double().cuda()
    train = torch.as_tensor(splits == "train", device="cuda")
    xt, yt = x[train].flatten(0, 1), y[train].flatten(0, 1)
    xv, yv = x[~train].flatten(0, 1), y[~train].flatten(0, 1)
    gram = xt.T@xt/len(xt); rhs = xt.T@yt/len(xt)
    energy = float(gram.diag().mean()); assert energy > 0 and np.isfinite(energy)
    eye = torch.eye(gram.shape[0], dtype=torch.float64, device="cuda")
    scores = []; best = None
    for alpha in json.loads((root/"plan.json").read_text())["ridge_lambdas"]:
        weight = torch.linalg.solve(gram+alpha*energy*eye, rhs).T
        train_mse = float((xt@weight.T-yt).square().mean())
        val_mse = float((xv@weight.T-yv).square().mean())
        row = dict(alpha=alpha, train_RMSE=train_mse**.5, validation_RMSE=val_mse**.5)
        assert np.isfinite(val_mse) and torch.isfinite(weight).all()
        scores.append(row)
        if best is None or val_mse < best[0]: best = (val_mse, weight.detach().float().cpu(), alpha)
    # Actual supervised backward confirms the head cannot propagate into features.
    detached_input = xt[:32].float().detach().requires_grad_()
    head_weight = best[1].cuda().clone().requires_grad_()
    loss = (detached_input.detach()@head_weight.T-yt[:32].float()).square().mean()
    loss.backward()
    assert detached_input.grad is None and head_weight.grad is not None
    assert torch.isfinite(head_weight.grad).all() and float(head_weight.grad.norm()) > 0
    head = dict(weight=best[1], arm=arm, alpha=best[2], input_width=x.shape[-1],
        target="same_transition_issued_translation_a_t", no_intercept=True,
        labels_are_optimal=False, backbone_gradient=False, head_fit="analytic ridge; no SGD/optimizer")
    torch.save(head, directory/"head.pt")
    save(directory/"ready.json", dict(passed=True, head_sha256=sha(directory/"head.pt"),
        selected_alpha=best[2], scores=scores, train_transitions=len(xt), validation_transitions=len(xv),
        validation_zero_RMSE=float(yv.square().mean().sqrt()),
        validation_RMSE=best[0]**.5, input_width=x.shape[-1],
        physical_kitchen_split_disjoint=True, actual_stopgradient_backward_pass=True,
        no_new_backbone_updates=True, no_action_loss_to_backbone=True,
        no_future_or_optimal_action_labels=True, target_case_feedback_used_for_selection=False))


class Policy:
    def __init__(self, root, arm):
        from experiments.robocasa_refine import encoder_ablation as base
        self.base = base
        self.readout = Readout(root, arm).cuda()
        self.before = common.state_hash(self.readout.state_dict())
        self.encoder = base.load_encoder(root, "vjepa")
        self.versions = [p._version for p in self.encoder.parameters()]
        directory = root/"heads"/arm; proof = json.loads((directory/"ready.json").read_text())
        assert proof["passed"] and sha(directory/"head.pt") == proof["head_sha256"]
        self.weight = torch.load(directory/"head.pt", map_location="cpu", weights_only=True)["weight"].cuda()
        self.seconds = 0.; self.calls = 0

    @torch.no_grad()
    def encode(self, observation):
        import cv2
        cv2.setNumThreads(1); images = []
        for key in ("img_0", "img_1"):
            frame = observation[key]; h,w = frame.shape[:2]; left = (w-h)//2
            assert frame.dtype == np.uint8 and frame.std() > 1.
            images.append(cv2.resize(frame[:, left:left+h], (224,224), interpolation=cv2.INTER_AREA).transpose(2,0,1))
        torch.cuda.synchronize(); began = time.perf_counter()
        features = self.base.encode(self.encoder, torch.from_numpy(np.stack(images)).cuda(), "vjepa")
        z = self.readout(features).flatten()
        torch.cuda.synchronize(); self.seconds += time.perf_counter()-began; self.calls += 1
        return z

    @torch.no_grad()
    def command(self, observation, goal):
        current = self.encode(observation)
        value = self.weight@(goal-current)
        assert torch.isfinite(value).all()
        value = value*torch.clamp(.3/value.norm().clamp_min(1e-12), max=1.)
        command = np.zeros(7, np.float32); command[:3] = value.cpu().numpy()
        return command

    def unchanged(self):
        return common.state_hash(self.readout.state_dict()) == self.before and (
            self.versions == [p._version for p in self.encoder.parameters()] and
            all(p.grad is None and not p.requires_grad for p in self.encoder.parameters()))


def evaluate(root, arm, index):
    from experiments.robocasa_compare import evaluate as base
    base.install_public(root); torch.set_num_threads(2); torch.set_float32_matmul_precision("high")
    start = time.perf_counter(); policy = Policy(root, arm)
    startup_seconds = time.perf_counter()-start
    case = json.loads((root/"plan.json").read_text())["cases"][index]
    name = base.case_name(**case); path = root/"cases"/(name+".npz")
    proof = json.loads(path.with_suffix(".json").read_text()); assert proof["passed"] and sha(path) == proof["sha256"]
    cached = np.load(path); env = base.make_env(**case); rewards = []; commands = []; infos = []; frames = []
    try:
        generated = base.reset(env, "reach", case["seed"], "goal")
        obs = base.reset(env, "reach", case["seed"], "init")
        goal_proof = base.goal_proof("reach", generated, obs)
        physical = base.verify_physics(env, cached, name)
        initial = np.stack([obs["img_0"], obs["img_1"]]); goal_images = np.stack([generated["img_0"], generated["img_1"]])
        initial_mse = float(np.mean((initial.astype(float)-cached["initial"].astype(float))**2))
        goal_mse = float(np.mean((goal_images.astype(float)-cached["goal"].astype(float))**2))
        assert max(initial_mse, goal_mse) < 5., (initial_mse, goal_mse)
        began = time.perf_counter(); goal = policy.encode(dict(img_0=cached["goal"][0], img_1=cached["goal"][1]))
        frames.append(initial); physics_seconds = 0.
        for _ in range(5):
            command = policy.command(obs, goal)
            tick = time.perf_counter(); obs,reward,done,info = env.step(command)
            physics_seconds += time.perf_counter()-tick
            commands.append(command); rewards.append(float(reward))
            infos.append({k:float(v) for k,v in info.items() if np.isscalar(v) and isinstance(v,(bool,int,float,np.number))})
            frames.append(np.stack([obs["img_0"], obs["img_1"]]))
            if done: break
        assert policy.unchanged()
        output = root/"evaluation"/arm/name; output.mkdir(parents=True, exist_ok=False)
        trace = output/"trace.npz"
        np.savez_compressed(trace, actions=np.asarray(commands), rewards=np.asarray(rewards), frames=np.stack(frames))
        distances = [x["reach_distance_to_goal"] for x in infos]
        row = dict(passed=True, arm=arm, **case, case_sha256=proof["sha256"], trace_sha256=sha(trace),
            score=float(np.mean(rewards)), metric="mean_normalized_progress", macro_steps=len(commands),
            final_goal_distance_m=distances[-1], final_task_complete=distances[-1]<.002,
            any_task_complete=any(x<.002 for x in distances), rewards=rewards, infos=infos,
            physical_replay_pass=True, physical_replay_errors=physical, initial_rgb_mse=initial_mse,
            goal_rgb_mse=goal_mse, goal_validation=goal_proof, model_unchanged=True,
            elapsed_seconds=time.perf_counter()-began, encoder_and_readout_seconds=policy.seconds,
            encoder_calls=policy.calls, physics_seconds=physics_seconds, startup_seconds=startup_seconds,
            CEM_calls=0, learned_inverse_current_command=True, optimal_action_assumed=False,
            inference_goal_distribution_shift=True, no_proprio_reward_or_state_to_policy=True)
        save(output/"summary.json", row); print(json.dumps(row), flush=True)
    finally: env.close()


def call(root, args, log, gpu, simulator=False):
    from experiments.robocasa_time_compare.campaign import env
    code = root/"code"; child_env = env(root, str(gpu), simulator)
    with log.open("x") as stream:
        subprocess.run(args, cwd=code, env=child_env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=3600)


def worker(root, host):
    arm = HOSTS[host]; logs = root/"logs"/host; logs.mkdir(parents=True, exist_ok=False)
    try:
        boundary = json.loads((root/f"boundary-{host}.json").read_text()); pid = boundary["old_owner"]
        deadline = time.time()+7200
        while True:
            assert time.time() < deadline and not (root/"STOP").exists()
            entries = [line.split(None,3) for line in subprocess.check_output(["ps","-eo","pid=,ppid=,stat=,args="],text=True).splitlines()]
            children = {pid}
            while True:
                extra = {int(p) for p,parent,_,_ in entries if int(parent) in children}
                if extra <= children: break
                children |= extra
            active = [int(p) for p,_,status,_ in entries if int(p) in children and int(p)!=pid and not status.startswith("Z")]
            if not active: break
            time.sleep(5)
        old = Path("/proc")/str(pid)/"cmdline"
        if old.exists():
            assert b"robocasa_time_compare.campaign" in old.read_bytes()
            os.kill(pid, signal.SIGTERM); os.kill(pid, signal.SIGCONT)
        stop = time.time()+30
        while old.exists() and old.read_bytes():
            assert time.time()<stop; time.sleep(.2)
        compute = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader"],text=True).strip()
        assert not compute, compute
        save(root/f"OLD_STAGE_RETIRED-{host}.json", dict(passed=True, old_owner=pid,
            child_stage_completed_before_retirement=True, no_child_training_killed=True,
            no_old_GPU_processes=True, retired_unix=time.time()))
        def one(shard):
            call(root, [TRAIN_PYTHON,"-u","-m",MODULE,"--root",str(root),"--mode","cache",
                "--arm",arm,"--shard",str(shard)], logs/f"cache-{shard}.log", shard)
        with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(one,range(8)))
        call(root,[TRAIN_PYTHON,"-u","-m",MODULE,"--root",str(root),"--mode","fit","--arm",arm],logs/"fit.log",0)
        def control(index):
            call(root,[str(root/"environment/bin/python"),"-u","-m",MODULE,"--root",str(root),
                "--mode","evaluate","--arm",arm,"--case",str(index)],logs/f"evaluate-{index}.log",index,True)
        with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(control,range(8)))
        rows = [json.loads(p.read_text()) for p in (root/"evaluation"/arm).glob("*/summary.json")]
        assert len(rows)==8 and all(q["passed"] for q in rows)
        save(root/"results"/(arm+".json"),dict(passed=True,arm=arm,cases=8,
            mean_return=float(np.mean([q["score"] for q in rows])),
            final_distance_mm=float(np.mean([q["final_goal_distance_m"]*1000 for q in rows])),
            final_successes=sum(q["final_task_complete"] for q in rows),any_successes=sum(q["any_task_complete"] for q in rows),
            mean_episode_seconds=float(np.mean([q["elapsed_seconds"] for q in rows])),
            mean_encoder_seconds=float(np.mean([q["encoder_and_readout_seconds"] for q in rows])),
            CEM_calls=0,pretraining_label_separation=True))
        save(root/f"DONE-{host}.json",dict(passed=True,host=host,arm=arm,completed_unix=time.time()))
    except BaseException as error:
        save(root/f"ERROR-{host}.json",dict(error=repr(error),method_negative=False)); raise


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",choices=("cache","fit","evaluate","worker"),required=True)
    p.add_argument("--arm",choices=tuple(HOSTS.values()));p.add_argument("--host",choices=tuple(HOSTS))
    p.add_argument("--shard",type=int);p.add_argument("--case",type=int);a=p.parse_args()
    if a.mode=="cache":cache(a.root,a.arm,a.shard)
    elif a.mode=="fit":fit(a.root,a.arm)
    elif a.mode=="evaluate":evaluate(a.root,a.arm,a.case)
    else:worker(a.root,a.host)
