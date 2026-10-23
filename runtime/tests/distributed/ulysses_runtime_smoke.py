"""Tier-B torchrun smoke for subgroup loading and collective-safe evaluation."""

from __future__ import annotations

import os
import tempfile
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

from vforge.data.loader import LeaderBroadcastDataLoader, build_eval_loader
from vforge.distributed.parallel_state import (
    ParallelConfig,
    destroy_parallel_state,
    init_parallel_state,
)
from vforge.distributed.parallelize import (
    FSDP2Config,
    ParallelizeConfig,
    materialize_meta_module,
    parallelize_model,
)
from vforge.evaluation.runner import EvalComponents, EvalOutputWriter, run_evaluation
from vforge.training.engine import EngineRunContext


class _Loader:
    def __iter__(self):
        yield {"value": torch.tensor([10])}
        yield {"value": torch.tensor([20])}

    def __len__(self) -> int:
        return 2

    def state_dict(self):
        return {"cursor": 0}

    def load_state_dict(self, state):
        assert state == {"cursor": 0}


class _FailingLoader(_Loader):
    def __iter__(self):
        yield {"value": torch.tensor([10])}
        raise ValueError("intentional tier-b collate failure")


def _shared_tempdir(rank: int) -> str:
    payload = [tempfile.mkdtemp(prefix="vforge-tier-b-") if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return str(payload[0])


def main() -> None:
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl" if use_cuda else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise RuntimeError("ulysses_runtime_smoke.py requires exactly 4 ranks")

    subgroup = dist.new_group(ranks=[2, 3], backend="gloo")
    if rank in {2, 3}:
        is_leader = rank == 2
        loader = LeaderBroadcastDataLoader(
            _Loader() if is_leader else None,
            group=subgroup,
            is_leader=is_leader,
            group_size=2,
        )
        batches = list(loader)
        assert [int(batch["value"].item()) for batch in batches] == [10, 20]

        leader_only = LeaderBroadcastDataLoader(
            _Loader() if is_leader else None,
            group=subgroup,
            is_leader=is_leader,
            group_size=2,
            broadcast_batches=False,
        )
        values = list(leader_only)
        if is_leader:
            assert [int(batch["value"].item()) for batch in values] == [10, 20]
        else:
            assert values == [None, None]

        failing = LeaderBroadcastDataLoader(
            _FailingLoader() if is_leader else None,
            group=subgroup,
            is_leader=is_leader,
            group_size=2,
        )
        try:
            list(failing)
        except (ValueError, RuntimeError) as exc:
            assert "intentional tier-b collate failure" in str(exc)
        else:
            raise AssertionError("leader failure must propagate to every rank")
    dist.barrier()

    output_dir = _shared_tempdir(rank)
    device = (
        torch.device("cuda", int(os.environ["LOCAL_RANK"])) if use_cuda else torch.device("cpu")
    )
    state = init_parallel_state(
        ParallelConfig(
            dp_size=2,
            usp_size=2,
            dp_mode="fsdp2",
            device_type=device.type,
        )
    )
    with torch.device("meta"):
        meta_model = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
        )
    parallelize_model(
        meta_model,
        ParallelizeConfig(
            parallel=state.config,
            fsdp2=FSDP2Config(enable=True),
        ),
    )
    assert all(parameter.is_meta for parameter in meta_model.parameters())
    materialize_meta_module(meta_model, device)
    assert all(not parameter.is_meta for parameter in meta_model.parameters())
    model = torch.nn.Linear(1, 1).to(device)
    prepared = {"value": False}

    class EvalEngine:
        def prepare_model(self, live_model, *, gradient_checkpointing_kwargs=None):
            del gradient_checkpointing_kwargs
            prepared["value"] = True
            return live_model

        def materialize_model(self, live_model, target_device):
            assert prepared["value"]
            return live_model.to(target_device)

        def run_forward(self, forward, *, run):
            with run.precision_context():
                return forward()

    class EvalDataset:
        def __getitem__(self, index):
            with Path(output_dir, f"read-rank-{rank}").open("a", encoding="utf-8") as handle:
                handle.write(f"{index}\n")
            return {
                "sample_id": f"sample-{index}",
                "value": torch.tensor([[float(index + 1)]]),
            }

        def __len__(self):
            return 3

    dataloader = build_eval_loader(
        EvalDataset(),
        parallel_state=state,
        collate_fn=lambda samples: samples[0],
    )

    class Inference:
        def __init__(self, engine_forward):
            self.engine_forward = engine_forward

        def infer_sample(self, sample, *, seed: int):
            with Path(output_dir, f"infer-rank-{rank}").open("a", encoding="utf-8") as handle:
                handle.write(f"{seed}\n")
            return {
                "sample_id": sample["sample_id"],
                "value": self.engine_forward(sample["value"]).detach().cpu(),
            }

    def save_output(result, destination: str) -> str:
        path = os.path.join(destination, f"{result['sample_id']}.pt")
        torch.save(result, path)
        return path

    components = EvalComponents(
        model=model,
        dataloader=dataloader,
        initial_weights=None,
        build_inference=lambda _model, engine_forward: Inference(engine_forward),
        output_writer=EvalOutputWriter(
            artifact_name=lambda sample_id: f"{sample_id}.pt",
            write_output=save_output,
        ),
    )
    manifest = run_evaluation(
        components,
        engine=EvalEngine(),
        engine_run=EngineRunContext(precision_context=nullcontext),
        device=device,
        output_dir=output_dir,
        skip_existing=False,
    )
    if state.global_rank == 0:
        assert len(list(Path(output_dir).glob("infer-rank-*"))) == world_size
        assert len(list(Path(output_dir).glob("read-rank-*"))) == 2
        assert len(manifest["written"]) == 3
        assert all(Path(output_dir, f"sample-{index}.pt").is_file() for index in range(3))
        assert Path(output_dir, "eval_manifest.json").is_file()

        Path(output_dir, "sample-1.pt").unlink()
        Path(output_dir, "sample-1.pt.json").unlink()
    dist.barrier()

    run_evaluation(
        components,
        engine=EvalEngine(),
        engine_run=EngineRunContext(precision_context=nullcontext),
        device=device,
        output_dir=output_dir,
        skip_existing=True,
    )
    if state.global_rank == 0:
        for rank_index in range(world_size):
            lines = Path(output_dir, f"infer-rank-{rank_index}").read_text().splitlines()
            assert len(lines) == 3
        for path in Path(output_dir).glob("read-rank-*"):
            assert len(path.read_text().splitlines()) == 4
        assert Path(output_dir, "sample-1.pt").is_file()

    dist.barrier()
    destroy_parallel_state()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
