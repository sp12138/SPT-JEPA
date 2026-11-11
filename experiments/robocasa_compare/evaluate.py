"""Common official RoboCasa tasks and unmodified public CEM for all models."""
import argparse
import atexit
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import tempfile
import types

import numpy as np
import torch

from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_compare.train import build, state_hash


def install_public(root):
    public = root / "sources/robo_jepa"
    sys.path[:0] = [str(public), str(public / "scripts/deployment")]
    os.environ["VJEPA2_HUB_REPO"] = str(root / "sources/vjepa2")


def make_env(task, kitchen, seed, camera_size=(480,640), push_object="flour_bag"):
    from robots.robocasa.object_push_kitchen import ObjectPushKitchen
    ObjectPushKitchen.DEFAULT_OBJECT_NAME = push_object
    from robots.robocasa.robot_env import RoboCasaEnv
    from robots.robocasa.blocking_robot_env import ReachBlockingRoboCasaEnv, ObjectPushBlockingRoboCasaEnv
    random.seed(902000 + kitchen); np.random.seed(902000 + kitchen)
    inner = RoboCasaEnv(task_name="ObjectPushKitchen" if task == "push" else "CloseStandMixerHead",
        camera_names=["robot0_agentview_left", "robot0_eye_in_hand"], camera_width=camera_size[1], camera_height=camera_size[0],
        seed=kitchen-1, robot="PandaOmron", chunk_size=4, wm_embodiment="franka_omron_state_delta",
        reach_task=task == "reach", reach_task_seed=seed, reach_min_distance=.25, reach_max_distance=.35,
        reach_clear_radius=1., kitchen_split="target", layout_and_style_ids=[(kitchen, kitchen)])
    if task == "reach":
        return ReachBlockingRoboCasaEnv(inner, reach_task_seed=seed, reach_min_distance=.25,
            reach_max_distance=.35, reach_clear_radius=1., kitchen_split="target", pos_tol=.001, max_steps=200)
    return ObjectPushBlockingRoboCasaEnv(inner, object_push_object_name=push_object,
        object_push_task_seed=seed, object_push_min_distance=.10, object_push_max_distance=.15,
        object_push_forward_cone_deg=60., object_push_clear_radius=1., pos_tol=.001, max_steps=200)


def reset(env, task, seed, mode):
    kwargs = dict(mode=mode, reset_episode_counter=True, reach_episode_idx=0)
    kwargs["reach_task_seed" if task == "reach" else "object_push_task_seed"] = seed
    result = env.reset(**kwargs)
    obs = result[0] if isinstance(result, tuple) else result
    for name in ["img_0", "img_1"]:
        image = obs[name]
        assert image.shape in ((240,320,3),(480,640,3)) and image.dtype == np.uint8
        assert float(image.std()) > 1., (name, "invalid RGB")
    return obs


def physics(env):
    actual = env._env._env
    return dict(qpos=actual.sim.data.qpos.copy(), qvel=actual.sim.data.qvel.copy(),
                time=np.asarray(actual.sim.data.time),
                kitchen=np.asarray([actual.layout_id, actual.style_id], dtype=np.int64))


def native_training_pose(env):
    """Only the proprioceptive reference uses these source-aligned sensors.

    The released manual helper uses the floor origin and another quaternion
    hemisphere; the source recordings use the native arm-base sensors.
    """
    sensors = env._env._env._get_observations(force_update=True)
    value = np.concatenate([sensors["robot0_base_pos"], sensors["robot0_base_quat"],
                            sensors["robot0_base_to_eef_pos"], sensors["robot0_base_to_eef_quat"][:3],
                            sensors["robot0_gripper_qpos"][:1]])
    assert value.shape == (14,) and np.isfinite(value).all()
    return value


def verify_physics(env,cached,name):
    """Strict replay of deterministically constructed physical cases."""
    state=physics(env)
    assert np.array_equal(state["kitchen"],cached["kitchen"])
    record={key:float(np.max(np.abs(state[key]-cached[key]))) for key in ("qpos","qvel","time")}
    assert max(record.values())<=1e-8,(name,record)
    return record


def case_name(task, kitchen, seed):
    return f"{task}-k{kitchen}-s{seed}"


def goal_proof(task, goal, initial):
    prefix = "reach" if task == "reach" else "object_push"
    assert np.allclose(goal[prefix + "_goal_world"], initial[prefix + "_goal_world"], atol=1e-8, rtol=0)
    d0 = float(initial[prefix + "_d0"])
    lower, upper = (.25, .35) if task == "reach" else (.10, .15)
    assert lower - 1e-5 <= d0 <= upper + 1e-5, d0
    assert bool(goal["success"]), ("Goal construction did not reach its own target", goal.get(prefix + "_residual"))
    change = float(np.mean((goal["img_0"].astype(float) - initial["img_0"].astype(float))**2))
    assert change > .01, ("Identical initial/goal images", change)
    return dict(passed=True, native_goal_success=True, start_goal_distance=d0, side_image_change_mse=change)


def make_case(root, task, kitchen, seed):
    env = make_env(task, kitchen, seed)
    try:
        goal = reset(env, task, seed, "goal")
        initial = reset(env, task, seed, "init")
        proof = goal_proof(task, goal, initial)
        state = physics(env)
        assert np.array_equal(state["kitchen"], [kitchen, kitchen]), state["kitchen"]
        path = root / "cases" / (case_name(task, kitchen, seed) + ".npz")
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            cached = np.load(path)
            assert np.allclose(cached["qpos"], state["qpos"], atol=1e-8, rtol=0)
            for key, values in [("initial", initial), ("goal", goal)]:
                image = np.stack([values["img_0"], values["img_1"]])
                assert np.mean((cached[key].astype(float)-image.astype(float))**2) < 5.
            save(path.with_suffix(".goal-audit.json"), dict(**proof, original_case_sha256=sha(path)))
            return
        np.savez_compressed(path, initial=np.stack([initial["img_0"], initial["img_1"]]),
                            goal=np.stack([goal["img_0"], goal["img_1"]]), **state)
        row = dict(passed=True, task=task, kitchen=kitchen, scene_seed=kitchen-1, seed=seed, path=str(path), sha256=sha(path),
                   goal_proof=proof,
                   initial_rgb_std=[float(initial[k].std()) for k in ["img_0", "img_1"]],
                   goal_rgb_std=[float(goal[k].std()) for k in ["img_0", "img_1"]])
        save(path.with_suffix(".json"), row)
        print(json.dumps(row), flush=True)
    finally:
        env.close()


def control14(command):
    result = torch.zeros((*command.shape[:-1], 14), device=command.device, dtype=torch.float32)
    result[..., 7:10] = command[..., :3]
    result[..., 13] = -.04 * command[..., 6]
    return result


@dataclass
class RollState:
    features: torch.Tensor
    actions: torch.Tensor

    def repeat(self, n):
        return RollState(self.features.repeat(n, 1, 1, 1, 1, 1), self.actions.repeat(n, 1, 1))

    def to(self, device):
        return RollState(self.features.to(device), self.actions.to(device))


class SPTAdapter:
    def __init__(self, root, run, step, online_correction=False):
        self.cfg = json.loads((run / "config.json").read_text())
        self.model = build(self.cfg["model"], root / "code").cuda().eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(run / f"step-{step:06d}/weights.pt", map_location="cuda", weights_only=True))
        expected = json.loads((run / f"step-{step:06d}/ready.json").read_text())
        assert expected["model_sha256"] == state_hash(self.model.state_dict())
        card = json.loads((root / "packed/dataset.json").read_text())
        self.mean = torch.tensor(card["action_mean"], device="cuda")
        self.std = torch.tensor(card["action_std"], device="cuda")
        self.before = state_hash(self.model.state_dict())
        self.extra_information = "RGB and factual past transition controls only; no absolute proprioception"
        self.online_correction=online_correction
        self.residual=None
        if online_correction:
            self.extra_information+="; per-episode Bayesian residual updates from own past RGB transitions"

    @torch.no_grad()
    def encode(self, observations):
        import cv2
        cv2.setNumThreads(1)
        def training_fov(value):
            # Source cameras are square. Preserve their vertical field of view
            # when the public simulator supplies a wider rectangular render.
            height, width = value.shape[:2]
            assert width >= height
            left = (width-height)//2
            value = value[:, left:left+height]
            return cv2.resize(value, (224, 224), interpolation=cv2.INTER_AREA)
        images = np.stack([[training_fov(o[k]) for k in ("img_0", "img_1")]
                           for o in observations])
        length, views, height, width, channels = images.shape
        value = torch.from_numpy(images).permute(0, 1, 4, 2, 3).reshape(length * views, channels, height, width).cuda()
        pixels = self.model.normalize(value)[:, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emb = self.model.encode(dict(pixels=pixels))["emb"]
        return emb.reshape(1, length, views, self.model.state_slots, 1, self.model.state_width).float()

    def history(self, observations, commands, realized_controls=None):
        value = self.encode(observations)
        if realized_controls is None:
            actions = torch.as_tensor(np.asarray(commands, np.float32).reshape(-1, 7), device="cuda")[None]
            raw = control14(actions)
        else:
            raw=torch.as_tensor(np.asarray(realized_controls,np.float32).reshape(-1,14),device="cuda")[None]
            assert raw.shape[1]==len(observations)-1
            if self.cfg.get("input_semantics")=="achieved-translation":
                translation=raw[...,7:10].clone();raw=torch.zeros_like(raw);raw[...,7:10]=translation
        return RollState(value, (raw - self.mean) / self.std)

    @torch.no_grad()
    def nominal_step(self, action, state, **kwargs):
        action = action.to("cuda")
        known = (control14(action) - self.mean) / self.std
        controls = torch.cat([state.actions, known], 1)
        b, t, views = state.features.shape[:3]
        emb = state.features.permute(0, 2, 1, 3, 4, 5).reshape(b * views, t, self.model.state_dim)
        controls = controls[:, None].expand(b, views, t, 14).reshape(b * views, t, 14)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            act = self.model.action_encoder(controls)
            future = self.model.predict(emb, act)[:, -1]
        future = future.reshape(b, views, self.model.state_slots, 1, self.model.state_width)[:, None].float()
        features = torch.cat([state.features, future], 1)[:, -(self.model.window-1):]
        actions = torch.cat([state.actions, known], 1)[:, -(features.shape[1]-1):]
        return future, RollState(features, actions)

    @torch.no_grad()
    def step(self,action,state,**kwargs):
        future,following=self.nominal_step(action,state,**kwargs)
        if self.online_correction:
            from experiments.robocasa_refine.control import ResidualPosterior
            if self.residual is None:
                self.residual=ResidualPosterior(14,future[0].numel(),self.model.sigma0,future.device)
            design=(control14(action.to("cuda"))-self.mean)/self.std
            future=future+self.residual.predict(design).reshape_as(future)
            following.features=following.features.clone()
            following.features[:,-1:]=future
        return future,following

    @torch.no_grad()
    def observe(self,command,nominal,observation):
        actual=self.encode([observation])
        action=torch.as_tensor(command,device="cuda").float()[None,None]
        design=(control14(action)-self.mean)/self.std
        assert self.residual is not None
        self.residual.update(design,actual-nominal)

    def unchanged(self):
        return state_hash(self.model.state_dict()) == self.before


class RoboAdapter:
    def __init__(self, root, size, mode, pose_source="native"):
        from app.robo_jepa.world_model import WorldModel
        self.model = WorldModel.load_pretrained(root / "reference" / size, load_decoder=False,
                                               load_compiled=False, device="cuda")
        self.model.transition_model.eval().requires_grad_(False)
        self.mode = mode
        self.pose_source=pose_source
        self.before = state_hash(dict(self.model.transition_model.named_parameters()))
        self.extra_information = "published frozen V-JEPA encoder and full multi-dataset pretraining"
        if mode == 3:
            self.extra_information += f"; robot proprioception from {pose_source} sensors (separate input track)"

    @torch.no_grad()
    def encode(self, observations):
        from PIL import Image
        values = []
        for obs in observations:
            # Released policy_client resizes and encodes each camera separately.
            views=[]
            for key in ("img_0","img_1"):
                frame=np.asarray(Image.fromarray(obs[key]).resize((320,240)))
                views.append(self.model.encode(frame[None],size=(240,320),dtype=torch.float16))
            values.append(torch.cat(views,0))
        return torch.stack(values, 0)[None].float()

    def history(self, observations, commands):
        from app.robo_jepa.world_model import WorldModelState,poses_to_diffs
        features = self.encode(observations)
        assert len(commands)==len(observations)-1
        if self.mode == 2:
            states = torch.zeros(1, len(observations), 7, device="cuda")
            poses=torch.as_tensor(np.stack([o["pose"] for o in observations])).float()
            actual=poses_to_diffs(poses)[None].cuda() if len(observations)>1 else states[:,:0]
        else:
            key="_reference_native_pose" if self.pose_source=="native" else "proprio_robocasa"
            states = torch.as_tensor(np.stack([o[key] for o in observations]), device="cuda").float()[None]
            actual=states[:,1:]-states[:,:-1]
        state = WorldModelState.from_features(features=features[:, :1], states=states[:, :1],
            view_map=torch.tensor([[0, 2]], device="cuda"),
            embodiment_ids=torch.tensor([[self.mode]], device="cuda"))
        # Same real-context replay as the released policy client: step packs
        # the action representation, then actual observations replace predictions.
        for index in range(len(commands)):
            action = actual[:,index:index+1]
            _, state = self.step(action, state, use_cache=True, no_grad=True, device="cuda")
            state.features = features[:, index+1:index+2]
            state.states = states[:, index+1:index+2]
        return state

    def step(self, action, state, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model.step(action, state, **kwargs)

    def unchanged(self):
        return state_hash(dict(self.model.transition_model.named_parameters())) == self.before


def initialize_singleton():
    import torch.distributed as dist
    if not dist.is_initialized():
        directory=Path(tempfile.mkdtemp(prefix="spt-singleton-"))
        atexit.register(shutil.rmtree,directory,ignore_errors=True)
        store=dist.FileStore(str(directory/"store"),1)
        torch.cuda.set_device(0)
        dist.init_process_group("nccl", store=store, rank=0, world_size=1,
                                device_id=torch.device("cuda", 0))


def direct_tensor_cem(public_cem):
    """Keep pinned CEM bytecode, sampling and math; omit redundant forced GC.

    RollState contains only tensors, with no cyclic dispatcher/KV-cache objects.
    The public reference retains its original per-batch collection. A separate
    namespace avoids mutating upstream or the process-wide gc module.
    """
    namespace = dict(public_cem.__globals__, gc=types.SimpleNamespace(collect=lambda: 0))
    return types.FunctionType(public_cem.__code__, namespace, public_cem.__name__,
                              public_cem.__defaults__, public_cem.__closure__)


def evaluate(args, adapter=None):
    from app.robo_jepa.mpc.mpc_utils import cem as public_cem
    initialize_singleton()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if adapter is not None:
        if isinstance(adapter,SPTAdapter):
            adapter.residual=None
    elif args.reference:
        adapter = RoboAdapter(args.root, args.reference, args.reference_mode,getattr(args,"reference_pose_source","native"))
    elif args.zero:
        adapter = None
    else:
        adapter = SPTAdapter(args.root, args.run, args.step,getattr(args,"online_correction",False))
        from vforge.ops import OpsConfig, initialize_ops
        initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False), device=torch.device("cuda"), parallel_state=None)
    cem = public_cem if args.reference else direct_tensor_cem(public_cem)
    name = case_name(args.task, args.kitchen, args.seed)
    path = args.root / "cases" / (name + ".npz")
    row = json.loads(path.with_suffix(".json").read_text())
    assert row["passed"] and sha(path) == row["sha256"]
    cached = np.load(path)
    env = make_env(args.task, args.kitchen, args.seed)
    rewards, commands, infos = [], [], []
    began = time.time()
    budget = 5 if args.task == "reach" else 18
    if args.technical:
        budget = 2
    try:
        generated_goal = reset(env, args.task, args.seed, "goal")
        obs = reset(env, args.task, args.seed, "init")
        last_native=native_training_pose(env)
        if args.reference and args.reference_mode == 3:
            obs["_reference_native_pose"] = last_native.copy()
        goal_validation = goal_proof(args.task, generated_goal, obs)
        physics_errors=verify_physics(env,cached,name)
        actual = np.stack([obs["img_0"], obs["img_1"]])
        goal_actual = np.stack([generated_goal["img_0"], generated_goal["img_1"]])
        initial_mse = float(np.mean((actual.astype(float)-cached["initial"].astype(float))**2))
        goal_mse = float(np.mean((goal_actual.astype(float)-cached["goal"].astype(float))**2))
        assert max(initial_mse, goal_mse) < 5., (name, initial_mse, goal_mse)
        goal_obs = {"img_0": cached["goal"][0], "img_1": cached["goal"][1]}
        observations = [obs]
        realized=[]
        goal = (adapter.initialize_goal(goal_obs,obs) if hasattr(adapter,"initialize_goal")
                else adapter.encode([goal_obs])) if adapter else None
        objective_kwargs={}
        if getattr(args,"goal_metric","uniform")=="support":
            assert not args.reference and adapter is not None
            from experiments.robocasa_refine.control import goal_support_cost
            objective_kwargs["objective"]=goal_support_cost(adapter.encode([obs]),goal,
                                                          adapter.model.state_slots,adapter.model.state_width)
        elif getattr(args,"goal_metric","uniform")=="direction":
            assert not args.reference and adapter is not None
            from experiments.robocasa_refine.control import goal_direction_cost
            objective_kwargs["objective"]=goal_direction_cost(adapter.encode([obs]),goal)
        warm = None
        np.random.seed(710000 + args.seed + 1009 * args.kitchen)
        torch.manual_seed(710000 + args.seed + 1009 * args.kitchen)
        for index in range(budget):
            planning_started = time.time()
            if adapter:
                context = observations[-2:]
                past = commands[-(len(context)-1):] if len(context) > 1 else []
                if args.reference:
                    wm_state=adapter.history(context,past)
                else:
                    feedback=realized[-(len(context)-1):] if len(context)>1 else []
                    wm_state=adapter.history(context,past,feedback if getattr(args,"history_controls","achieved")=="achieved" else None)
                if hasattr(adapter,"goal_for_state"):
                    # A conditional representation uses the same current chart
                    # for real history, imagined futures and the visual goal.
                    goal=adapter.goal_for_state()
                _, means, _ = cem(adapter, wm_state, goal, rollout=1 if args.task == "reach" else 5,
                    action_repeat=getattr(args,"action_repeat",1),
                    cem_steps=2 if args.technical else 15 if index == 0 else 10,
                    samples=16, num_batches=1 if args.technical else 64, cutoff=10 if args.technical else 20,
                    use_gripper=False, maxnorm=.3, robot_init_std=.3, gripper_init_std=.8, maxgripper=.99,
                    momentum_mean=.15, momentum_std=.70, momentum_mean_gripper=.15, momentum_std_gripper=.45,
                    translation_only=True, view_weights=[.5, .5], prev_actions=warm,
                    use_cache=bool(args.reference), step_func=adapter.step, global_step=index,
                    total_steps=5 if args.task == "reach" else 18,**objective_kwargs)
                command = means[-1][0].numpy().copy()
                warm = means[-1][1:]
            else:
                command = np.zeros(7, np.float32)
            nominal=None
            if adapter is not None and not args.reference and adapter.online_correction:
                nominal,_=adapter.nominal_step(torch.as_tensor(command,device="cuda").float()[None,None],wm_state)
            obs, reward, done, info = env.step(command)
            current_native=native_training_pose(env)
            realized.append(current_native-last_native)
            last_native=current_native
            if args.reference and args.reference_mode == 3:
                obs["_reference_native_pose"] = current_native.copy()
            if nominal is not None:adapter.observe(command,nominal,obs)
            commands.append(command.copy()); rewards.append(float(reward))
            observations.append(obs)
            infos.append({k:float(v) for k, v in info.items() if np.isscalar(v) and isinstance(v, (bool, int, float, np.number))})
            assert np.isfinite(reward)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save(args.output.with_suffix(".progress.json"), dict(macro_step=index+1, budget=budget,
                 latest_reward=float(reward), elapsed_seconds=time.time()-began,
                 latest_plan_and_physics_seconds=time.time()-planning_started, final=False))
            if done:
                break
        unchanged = adapter.unchanged() if adapter else True
        assert unchanged
        args.output.mkdir(parents=True, exist_ok=False)
        trace = args.output / "trace.npz"
        np.savez_compressed(trace, actions=np.asarray(commands), rewards=np.asarray(rewards),
                            initial=actual, goal=cached["goal"], final=np.stack([obs["img_0"], obs["img_1"]]),
                            frames=np.stack([[o["img_0"], o["img_1"]] for o in observations]))
        discount = np.power(.99, np.arange(len(rewards)))
        normalized_discounted_return = float(np.dot(discount, rewards) / discount.sum())
        score = float(np.mean(rewards)) if args.task == "reach" else rewards[-1]
        distance_key = "reach_distance_to_goal" if args.task == "reach" else "object_push_distance_to_goal"
        tolerance = .002 if args.task == "reach" else .02
        distances = [info[distance_key] for info in infos]
        result = dict(passed=True, task=args.task, kitchen=args.kitchen, seed=args.seed,
            case_sha256=row["sha256"], trace_sha256=sha(trace), model_unchanged=unchanged,
            score=score, metric="mean_normalized_progress" if args.task == "reach" else "final_normalized_progress",
            rewards=rewards, infos=infos, macro_steps=len(commands), elapsed_seconds=time.time()-began,
            final_goal_distance_m=distances[-1], task_completion_tolerance_m=tolerance,
            final_task_complete=distances[-1] < tolerance, any_task_complete=any(d < tolerance for d in distances),
            normalized_discounted_return=normalized_discounted_return,
            initial_rgb_mse=initial_mse, goal_rgb_mse=goal_mse, physical_replay_pass=True,
            physical_replay_errors=physics_errors,
            replay_precision="All qpos/qvel/time1e-8; RGB MSE<5",
            goal_validation=goal_validation,
            technical_only=args.technical, reference=args.reference, reference_mode=args.reference_mode,
            information=adapter.extra_information if adapter else "zero control",
            candidate_rgb_preprocessing="training-square-FOV center crop and OpenCV area224" if args.run else None,
            planner_forced_gc="unchanged public reference" if args.reference else "omitted for acyclic tensor state; same CEM bytecode/math",
            history_control_convention="released_client_achieved_motion" if args.reference else getattr(args,"history_controls","achieved"),
            raw_camera_resolution=list(actual.shape[1:3]),
            reference_pose_source=getattr(args,"reference_pose_source","native") if args.reference else None,
            goal_metric=getattr(args,"goal_metric","uniform"),
            online_residual_updates=adapter.residual.updates if adapter is not None and not args.reference and adapter.residual is not None else 0,
            imagined_substeps_per_macro=getattr(args,"action_repeat",1),
            no_future_action_reference=True, no_ground_truth_intermediate_frame=True)
        save(args.output / "summary.json", result)
        print(json.dumps(result), flush=True)
    finally:
        env.close()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["case", "evaluate"], required=True)
    parser.add_argument("--task", choices=["reach", "push"], required=True)
    parser.add_argument("--kitchen", type=int, choices=range(1, 11), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--reference", choices=["50m", "2b","2b450k","8b300k"])
    parser.add_argument("--reference-pose-source",choices=["native","released"],default="native")
    parser.add_argument("--history-controls",choices=["issued","achieved"],default="achieved")
    parser.add_argument("--online-correction",action="store_true")
    parser.add_argument("--goal-metric",choices=["uniform","support","direction"],default="uniform")
    parser.add_argument("--action-repeat",type=int,default=1)
    parser.add_argument("--reference-mode", type=int, choices=[2, 3], default=2)
    parser.add_argument("--zero", action="store_true")
    parser.add_argument("--technical", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    install_public(args.root)
    try:
        if args.mode == "case":
            make_case(args.root, args.task, args.kitchen, args.seed)
        else:
            evaluate(args)
    except BaseException as error:
        name = case_name(args.task, args.kitchen, args.seed)
        destination = args.output.parent if args.output else args.root / "cases"
        destination.mkdir(parents=True, exist_ok=True)
        save(destination / (name + "-execution-error.json"), dict(error=repr(error), phase=args.mode,
             method_negative=False, task=args.task, kitchen=args.kitchen, seed=args.seed))
        raise
