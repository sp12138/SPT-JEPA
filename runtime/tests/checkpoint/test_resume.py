from dataclasses import replace

import pytest
import torch

from vforge.distributed.parallel_state import ParallelConfig
from vforge.distributed.parallelize import FSDP2Config, ParallelizeConfig
from vforge.training.checkpoint import (
    CheckpointManifest,
    build_dataloader_resume_contract,
    execution_fingerprint,
    parallel_fingerprint,
    validate_bit_exact_dataloader,
    validate_dataloader_resume_contract,
    validate_resume_fingerprints,
)
from vforge.training.parity import ParityStepConfig


def test_dataloader_resume_contract_includes_application_contract():
    loader = torch.utils.data.DataLoader(
        torch.arange(6),
        batch_size=2,
        drop_last=True,
    )

    contract = build_dataloader_resume_contract(
        loader,
        {"dataset_revision": "v2"},
    )

    assert contract["loader_length"] == 3
    assert contract["dataset_length"] == 6
    assert contract["batch_size"] == 2
    assert contract["drop_last"] is True
    assert contract["application"] == {"dataset_revision": "v2"}


def test_dataloader_resume_contract_rejects_non_dictionary_extension():
    with pytest.raises(
        TypeError,
        match="dataloader_resume_contract must return a dictionary",
    ):
        build_dataloader_resume_contract(
            [],
            ["not", "a", "mapping"],  # type: ignore[arg-type]
        )


def test_dataloader_resume_validation_preserves_bit_exact_flag():
    contract = {"loader_length": 3}
    assert validate_dataloader_resume_contract(
        contract,
        contract,
        bit_exact=True,
    )
    with pytest.raises(
        RuntimeError,
        match="strict resume rejected a dataloader contract mismatch",
    ):
        validate_dataloader_resume_contract(
            {"loader_length": 2},
            contract,
            bit_exact=True,
        )
    with pytest.warns(RuntimeWarning, match="sample stream is not bit-exact"):
        assert not validate_dataloader_resume_contract(
            {"loader_length": 2},
            contract,
            bit_exact=False,
        )


def test_dataloader_worker_policy_requires_explicit_support():
    worker_loader = torch.utils.data.DataLoader(
        torch.arange(4),
        num_workers=1,
    )
    with pytest.raises(RuntimeError, match="explicitly implement"):
        validate_bit_exact_dataloader(worker_loader, lambda _loader: None)
    validate_bit_exact_dataloader(worker_loader, lambda _loader: True)


def test_strict_resume_fingerprint_change_is_rejected():
    manifest = replace(
        CheckpointManifest.create(state="full", completed_steps=2),
        execution_policy={"precision": "bf16"},
        topology={"dp_size": 1},
    )

    assert validate_resume_fingerprints(
        manifest,
        execution_policy={"precision": "bf16"},
        topology={"dp_size": 1},
    )
    with pytest.raises(RuntimeError, match="execution policy mismatch"):
        validate_resume_fingerprints(
            manifest,
            execution_policy={"precision": "fp32"},
            topology={"dp_size": 1},
        )


def test_execution_and_parallel_fingerprints_capture_resume_policy():
    parallelize = ParallelizeConfig(
        parallel=ParallelConfig(),
        fsdp2=FSDP2Config(enable=False),
    )
    optimizer = torch.optim.SGD(
        torch.nn.Linear(1, 1).parameters(),
        lr=0.1,
    )

    execution = execution_fingerprint(
        seed=17,
        precision="bf16",
        autocast=True,
        gradient_accumulation_steps=1,
        gradient_clip_val=1.0,
        parity=ParityStepConfig(scale_grads_before_optimizer_step="usp_size"),
        optimizer=optimizer,
        scheduler=None,
        parallelize=parallelize,
    )
    topology = parallel_fingerprint(parallelize)

    assert execution["seed"] == 17
    assert execution["autocast"] is True
    assert execution["parity"] == {
        "optimizer_step_mode": "torch",
        "scale_grads_before_optimizer_step": "usp_size",
        "quantize_after_optimizer_step": None,
        "emit_legacy_probe_aliases": False,
    }
    assert execution["optimizer_type"] == "torch.optim.sgd.SGD"
    assert execution["parallelize"]["fsdp2"]["enable"] is False
    assert "attention_offload" not in execution
    assert topology["dp_size"] == 1
    assert topology["usp_size"] == 1


def test_parallel_fingerprint_captures_hsdp_split(monkeypatch):
    import vforge.training.checkpoint.resume as resume_module

    parallelize = ParallelizeConfig(
        parallel=ParallelConfig(
            dp_size=4,
            dp_replicate_size=2,
            dp_shard_size=2,
        ),
        fsdp2=FSDP2Config(enable=True),
    )
    monkeypatch.setattr(resume_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(resume_module.dist, "get_world_size", lambda: 4)

    topology = parallel_fingerprint(parallelize)

    assert topology["dp_size"] == 4
    assert topology["dp_replicate_size"] == 2
    assert topology["dp_shard_size"] == 2


def test_clipping_fingerprint_preserves_defaults_and_rejects_numerical_changes():
    parallelize = ParallelizeConfig()
    kwargs = dict(
        seed=42,
        precision="bf16",
        autocast=True,
        gradient_accumulation_steps=1,
        gradient_clip_val=1.0,
        parity=None,
        optimizer=None,
        scheduler=None,
        parallelize=parallelize,
    )
    from vforge.training.diagnostics import torch_gradient_clipper

    default = execution_fingerprint(**kwargs)
    # Master's logical/FP32 fingerprints had neither key. Existing checkpoints
    # must remain resumable without weakening strict equality for new choices.
    assert "gradient_clip_impl" not in default
    assert "gradient_norm_dtype" not in default
    manifest = replace(
        CheckpointManifest.create(state="full", completed_steps=1),
        execution_policy=default,
        topology={},
    )
    validate_resume_fingerprints(manifest, execution_policy=default, topology={})
    for dtype in ["float32", "float64"]:
        changed = execution_fingerprint(
            **kwargs, gradient_clipper=torch_gradient_clipper(dtype), gradient_norm_dtype=dtype
        )
        assert changed["gradient_clip_impl"] == "torch"
        assert changed["gradient_norm_dtype"] == dtype
        with pytest.raises(RuntimeError, match="execution policy"):
            validate_resume_fingerprints(manifest, execution_policy=changed, topology={})
