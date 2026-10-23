"""CPU tests for ResumeConfig dependency rules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_resume_config_dependency_matrix() -> None:
    from vforge.training.checkpoint import ResumeConfig, build_resume_config

    ResumeConfig(
        weights=True,
        optimizer=False,
        training_state=False,
    )
    with pytest.raises(ValueError, match="resume.weights"):
        ResumeConfig(weights=False, optimizer=True)
    with pytest.raises(ValueError, match="resume.training_state"):
        ResumeConfig(optimizer=False, training_state=True)
    assert ResumeConfig().effective_strict is False
    assert ResumeConfig(strict=True).effective_strict is True
    assert ResumeConfig(strict=None).effective_strict is True
    assert ResumeConfig(training_state=False).effective_strict is False
    assert ResumeConfig(training_state=False, strict=True).effective_strict is True
    staged = build_resume_config(
        {"optimizer": False, "training_state": False},
        stage_remote=True,
    )
    assert staged.stage_remote is True
    with pytest.raises(ValueError, match="merged into training_state"):
        build_resume_config({"progress": False})


def test_remote_checkpoint_policy_is_storage_aware(
    tmp_path: Path,
) -> None:
    from vforge.training.checkpoint import (
        CheckpointConfig,
        CheckpointManager,
    )
    from vforge.utils.s3_staging import (
        S3MountConfig,
        StorageConfig,
        activate_storage_config,
        restore_storage_config,
    )

    token = activate_storage_config(
        StorageConfig(s3_mounts=(S3MountConfig(str(tmp_path / "s3"), "checkpoints"),))
    )
    try:
        remote_manager = CheckpointManager(
            CheckpointConfig(
                enabled=True,
                every_n_steps=1,
                dirpath=str(tmp_path / "s3" / "run"),
                state="full",
                format="dcp",
            )
        )
        efs_manager = CheckpointManager(
            CheckpointConfig(
                enabled=True,
                every_n_steps=1,
                dirpath=str(tmp_path / "efs" / "run"),
                state="full",
                format="dcp",
            )
        )
        remote = remote_manager.config
        efs = efs_manager.config
        remote_manager.close()
        efs_manager.close()
    finally:
        restore_storage_config(token)

    assert remote.async_save is True
    assert remote.remote_upload is True
    assert efs.remote_upload is False


def test_strict_resume_rejects_policy_and_fingerprint_mismatch() -> None:
    from vforge.training.checkpoint import CheckpointManifest
    from vforge.training.checkpoint.resume import validate_resume_fingerprints

    manifest = CheckpointManifest.create(
        state="full",
        completed_steps=1,
        execution_policy={"seed": 42},
        topology={"usp_size": 8},
        fingerprints={"resolved_config_hash": "saved"},
    )
    with pytest.raises(RuntimeError, match="execution policy"):
        validate_resume_fingerprints(
            manifest,
            execution_policy={"seed": 17},
            topology={"usp_size": 8},
        )
    with pytest.raises(RuntimeError, match="resolved_config_hash"):
        validate_resume_fingerprints(
            manifest,
            execution_policy={"seed": 42},
            topology={"usp_size": 8},
            fingerprints={"resolved_config_hash": "current"},
        )
