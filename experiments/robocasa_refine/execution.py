"""CEM execution equivalence: keep sampler/math; bound Python cache garbage.

Generation-one collection still runs each batch. Full collection every eight
batches bounds caches that survived a Python generation promotion. This changes
only object reclamation, and requires output/RNG/memory checks per reference.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import types

import numpy as np
import torch

from experiments.robocasa_compare import evaluate as base
from experiments.robocasa_compare.acquire import save
from experiments.robocasa_refine.official_check import difference


def cadenced_cem(public_cem):
    calls = 0

    def collect():
        nonlocal calls
        calls += 1
        return gc.collect(2 if calls % 8 == 0 else 1)

    namespace = dict(public_cem.__globals__, gc=types.SimpleNamespace(collect=collect))
    return types.FunctionType(public_cem.__code__, namespace, public_cem.__name__,
                              public_cem.__defaults__, public_cem.__closure__)


def verify(root, size, output):
    base.install_public(root)
    base.initialize_singleton()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    adapter = base.RoboAdapter(root, size, 2)
    from app.robo_jepa.mpc.mpc_utils import cem
    path = root / "cases/reach-v2-s7401.npz"
    case = np.load(path)
    first = dict(img_0=case["initial"][0], img_1=case["initial"][1])
    goal = adapter.encode([dict(img_0=case["goal"][0], img_1=case["goal"][1])])
    initial_features = adapter.encode([first])
    from app.robo_jepa.world_model import WorldModelState

    def initial_state():
        # RGB mode2 has zero absolute states. No pose is consumed for a
        # one-observation history, so construct that exact state directly.
        return WorldModelState.from_features(features=initial_features,
            states=torch.zeros(1, 1, 7, device="cuda"),
            view_map=torch.tensor([[0, 2]], device="cuda"),
            embodiment_ids=torch.tensor([[2]], device="cuda"))
    # Two refits cross the full-GC cadence. A separate1024-candidate plan
    # below checks the actual heavy cache lifetime without duplicating control.
    kwargs = dict(rollout=5, cem_steps=2, samples=16, num_batches=8, cutoff=20,
        translation_only=True, use_gripper=False, maxnorm=.3, robot_init_std=.3,
        view_weights=[.5, .5], use_cache=True, total_steps=18, step_func=adapter.step)
    records = []
    outputs = []
    final_rng = []
    for label, function in (("original", cem), ("cadenced", cadenced_cem(cem))):
        state = initial_state()
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(99213); np.random.seed(99213)
        torch.cuda.synchronize(); start = time.monotonic()
        result = function(adapter, state, goal, **kwargs)
        torch.cuda.synchronize()
        records.append(dict(mode=label, seconds=time.monotonic()-start,
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
            allocated_at_end=torch.cuda.memory_allocated()))
        outputs.append([[v.cpu().clone() for v in group] for group in result])
        final_rng.append((torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone(), np.random.get_state()))
        del result, state
    errors = [max(difference(x, y) for x, y in zip(a, b)) for a, b in zip(*outputs)]
    assert max(errors) == 0., errors
    for i in (0, 1):
        assert torch.equal(final_rng[0][i], final_rng[1][i])
    assert final_rng[0][2][0] == final_rng[1][2][0]
    assert np.array_equal(final_rng[0][2][1], final_rng[1][2][1])
    assert final_rng[0][2][2:] == final_rng[1][2][2:]
    # Repeated heavy plans must not accumulate live GPU caches.
    allocated = []
    function = cadenced_cem(cem)
    for repeat in range(3):
        state = initial_state()
        result = function(adapter, state, goal, **dict(kwargs, num_batches=64 if repeat == 0 else 8))
        del result, state
        torch.cuda.synchronize()
        allocated.append(torch.cuda.memory_allocated())
    assert max(allocated)-min(allocated) < 64 * 1024**2, allocated
    assert adapter.unchanged()
    save(output, dict(passed=True, reference=size, maximum_output_errors=errors,
        identical_torch_numpy_cuda_rng=True, repeated_plan_live_memory=allocated,
        records=records, speedup=records[0]["seconds"]/records[1]["seconds"],
        model_unchanged=True, candidate_count_and_math_unchanged=True,
        parity_candidates=128, heavy_memory_candidates=1024,
        full_GC_every=8, generation_one_on_nonfull_batches=True,
        new_SGD=0, performance_claim=False))
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--reference", choices=["50m", "2b450k", "8b300k"], required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    verify(a.root, a.reference, a.output)
