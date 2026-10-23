"""Checkpoint policy dispatch and synchronous/asynchronous save lifecycle."""

from __future__ import annotations

import json
import os
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from threading import Event
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from vforge.utils.rng import RandomStreams

from . import storage
from .backend import CheckpointBackend, MCoreCheckpointBackend, PreparedMCoreSave
from .config import CheckpointConfig, checkpoint_path
from .manifest import CheckpointManifest
from .mapping import StateDictMapper

_UNSET = object()


@dataclass(frozen=True)
class CheckpointTiming:
    """Foreground timing shared by pending and completed checkpoints."""

    foreground_s: float
    state_dict_s: float
    sidecar_s: float
    stage_s: float
    wait_s: float

    def timing_dict(self) -> dict[str, float]:
        return {
            "checkpoint_foreground_s": self.foreground_s,
            "checkpoint_state_dict_s": self.state_dict_s,
            "checkpoint_sidecar_s": self.sidecar_s,
            "checkpoint_stage_s": self.stage_s,
            "checkpoint_wait_s": self.wait_s,
        }


@dataclass(frozen=True)
class CompletedCheckpoint(CheckpointTiming):
    """A globally committed asynchronous checkpoint and its phase timings."""

    path: str | None
    completed_steps: int
    upload_s: float
    commit_s: float
    total_s: float

    def timing_dict(self) -> dict[str, float]:
        return {
            **super().timing_dict(),
            "checkpoint_upload_s": self.upload_s,
            "checkpoint_commit_s": self.commit_s,
            "checkpoint_total_s": self.total_s,
        }


@dataclass(frozen=True)
class PendingCheckpoint(CheckpointTiming):
    """One staged checkpoint whose upload or outer commit is still pending."""

    path: str
    completed_steps: int
    _completion: Future[CompletedCheckpoint]

    def done(self) -> bool:
        return self._completion.done()

    def result(self) -> CompletedCheckpoint:
        return self._completion.result()


@dataclass(frozen=True)
class CheckpointSaveResult:
    """One foreground checkpoint scheduling result."""

    path: str | None = None
    pending: PendingCheckpoint | None = None


class CheckpointManager:
    """Own checkpoint policy, dispatch, and async DCP commit lifecycle."""

    def __init__(
        self,
        config: CheckpointConfig,
        *,
        state_dict_mapper: StateDictMapper | None = None,
        random_streams: RandomStreams | None = None,
    ) -> None:
        self.config = config
        if config.enabled and config.format == "dcp":
            from vforge.utils.s3_staging import resolve_s3_source

            if resolve_s3_source(config.dirpath) is not None:
                self.config = replace(
                    config,
                    async_save=True,
                    remote_upload=True,
                )
        self.state_dict_mapper = state_dict_mapper
        self.random_streams = random_streams if random_streams is not None else RandomStreams()
        self._mcore_backend: CheckpointBackend | None = None
        self._mcore_call_index: int | None = None
        if self.config.enabled and self.config.format == "mcore":
            from vforge.utils.s3_staging import resolve_s3_source

            if resolve_s3_source(self.config.dirpath) is not None:
                raise NotImplementedError(
                    "checkpoint.format='mcore' does not support remote upload"
                )
            self._mcore_backend = MCoreCheckpointBackend(
                async_save=self.config.async_save,
            )
        self._upload_group = None
        self._control_group = None
        self._executor: ThreadPoolExecutor | None = None
        if self.config.async_save and self.config.format == "dcp":
            self._upload_group = dist.new_group(backend="gloo") if dist.is_initialized() else None
            self._control_group = dist.new_group(backend="gloo") if dist.is_initialized() else None
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="vforge-dcp-commit",
            )
        self._pending: PendingCheckpoint | None = None
        self._closed = False

    def should_save(
        self,
        completed_steps: int,
        *,
        is_final: bool,
    ) -> bool:
        return self.config.should_save(completed_steps, is_final=is_final)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        path: Any = None,
        completed_steps: int | None = None,
        *,
        scheduler: Any = _UNSET,
        manifest: CheckpointManifest | None = None,
        rank_state: dict[str, Any] | None = None,
        dataloader_state: dict[str, Any] | None = None,
        execution_policy: dict[str, Any] | None = None,
        topology: dict[str, Any] | None = None,
        fingerprints: dict[str, Any] | None = None,
    ) -> PendingCheckpoint | CheckpointSaveResult:
        """Save through the low-level async or policy-driven training API.

        Passing ``manifest`` preserves the existing asynchronous DCP API where
        ``path`` is the destination. Otherwise the third positional argument is
        the scheduler and ``completed_steps`` drives policy output.
        """
        if self._closed:
            raise RuntimeError("CheckpointManager is closed")
        if scheduler is not _UNSET:
            if path is not None:
                raise TypeError("checkpoint save received both path and scheduler")
            path = scheduler
        if manifest is not None:
            if scheduler is not _UNSET:
                raise TypeError("low-level checkpoint save does not accept scheduler")
            if completed_steps is not None:
                raise TypeError("completed_steps cannot be passed with a low-level manifest")
            if not isinstance(path, str):
                raise TypeError("low-level checkpoint save requires a path")
            return self._save_async(
                model,
                optimizer,
                path,
                manifest=manifest,
                rank_state=rank_state,
            )
        if completed_steps is None:
            raise TypeError("policy-driven checkpoint save requires completed_steps")
        if dataloader_state is None:
            raise TypeError("policy-driven checkpoint save requires dataloader_state")
        if execution_policy is None:
            raise TypeError("policy-driven checkpoint save requires execution_policy")
        if topology is None:
            raise TypeError("policy-driven checkpoint save requires topology")
        return self._save_training_checkpoint(
            model,
            optimizer,
            path,
            completed_steps,
            dataloader_state=dataloader_state,
            execution_policy=execution_policy,
            topology=topology,
            fingerprints=fingerprints,
        )

    def poll(self) -> CompletedCheckpoint | None:
        if self._pending is None:
            return None
        if self._mcore_call_index is not None and not self._pending.done():
            assert self._mcore_backend is not None
            self._mcore_backend.poll(self._mcore_call_index)
        if not self._pending.done():
            return None
        try:
            return self._pending.result()
        finally:
            self._pending = None
            self._mcore_call_index = None

    def wait(self) -> CompletedCheckpoint | None:
        if self._pending is None:
            return None
        if self._mcore_call_index is not None and not self._pending.done():
            assert self._mcore_backend is not None
            self._mcore_backend.wait(self._mcore_call_index)
        try:
            return self._pending.result()
        finally:
            self._pending = None
            self._mcore_call_index = None

    def close(self) -> CompletedCheckpoint | None:
        if self._closed:
            return None
        self._closed = True
        try:
            return self.wait()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
            if self._mcore_backend is not None:
                self._mcore_backend.close()
            if dist.is_initialized():
                if self._upload_group is not None:
                    dist.destroy_process_group(self._upload_group)
                if self._control_group is not None:
                    dist.destroy_process_group(self._control_group)
            self._upload_group = None
            self._control_group = None

    def _save_training_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any,
        completed_steps: int,
        *,
        dataloader_state: dict[str, Any],
        execution_policy: dict[str, Any],
        topology: dict[str, Any],
        fingerprints: dict[str, Any] | None,
    ) -> CheckpointSaveResult:
        config = self.config
        path = checkpoint_path(config, completed_steps)
        if config.format == "pt":
            saved = storage.save_weights_only_pt(
                model,
                path,
                is_rank_zero=storage.is_global_zero(),
                state_dict_mapper=self.state_dict_mapper,
            )
            pending = None
        else:
            rank_state = None
            if config.state == "full":
                rank_state = {
                    "scheduler": (scheduler.state_dict() if scheduler is not None else None),
                    "dataloader": dataloader_state,
                    "rng": self.random_streams.state_dict(),
                }
            manifest = CheckpointManifest.create(
                state=config.state,
                completed_steps=completed_steps,
                format=config.format,
                execution_policy=execution_policy,
                topology=topology,
                fingerprints={
                    **(fingerprints or {}),
                    "dataloader": dataloader_state.get("contract"),
                },
            )
            if config.format == "mcore":
                if self._mcore_backend is None:
                    raise RuntimeError("MCore checkpoint backend was not initialized")
                prepared = self._mcore_backend.prepare_save(
                    model,
                    optimizer,
                    path,
                    manifest=manifest,
                    rank_state=rank_state,
                    save_optimizer=(config.state == "full" and config.save_optimizer),
                    save_trainer_state=config.save_trainer_state,
                    empty_cache_before_save=config.empty_cache_before_save,
                )
                if config.async_save:
                    pending = self._save_mcore_async(prepared)
                    saved = None
                else:
                    pending = None
                    saved = self._mcore_backend.save_sync(prepared)
            elif config.async_save:
                pending = self._save_async(
                    model,
                    optimizer,
                    path,
                    manifest=manifest,
                    rank_state=(rank_state if config.save_trainer_state else None),
                )
                saved = None
            else:
                pending = None
                saved = storage.save_dcp(
                    model,
                    optimizer,
                    path,
                    manifest=manifest,
                    rank_state=rank_state,
                    save_optimizer=(config.state == "full" and config.save_optimizer),
                    save_trainer_state=config.save_trainer_state,
                    empty_cache_before_save=config.empty_cache_before_save,
                    state_dict_mapper=self.state_dict_mapper,
                )
            if rank_state is not None:
                self.random_streams.load_state_dict(rank_state["rng"])
        if pending is not None:
            if storage.is_global_zero():
                print(
                    f"[vforge] staged async {config.state}+{config.format} "
                    f"checkpoint step={completed_steps} -> {path}",
                    flush=True,
                )
            return CheckpointSaveResult(pending=pending)

        storage.apply_retention(config)
        if saved is not None:
            print(
                f"[vforge] wrote {config.state}+{config.format} checkpoint "
                f"step={completed_steps} -> {saved}",
                flush=True,
            )
        if dist.is_initialized():
            dist.barrier()
        return CheckpointSaveResult(path=saved)

    def _save_mcore_async(self, prepared: PreparedMCoreSave) -> PendingCheckpoint:
        if self._mcore_backend is None or not self.config.async_save:
            raise ValueError("MCore asynchronous checkpoint backend is not enabled")
        wait_s = 0.0
        if self._pending is not None:
            wait_started = time.perf_counter()
            self.wait()
            wait_s = time.perf_counter() - wait_started

        started_at = time.perf_counter()
        completion: Future[CompletedCheckpoint] = Future()
        timing_context = {"foreground_s": 0.0}

        def finalize() -> None:
            commit_started = time.perf_counter()
            commit_error: str | None = None
            if storage.is_global_zero():
                try:
                    storage._commit_dcp_directory(
                        prepared.path,
                        prepared.incomplete,
                        prepared.manifest,
                    )
                    storage.apply_retention(self.config)
                except BaseException as exc:
                    commit_error = f"{type(exc).__name__}: {exc}"
            try:
                storage._raise_if_checkpoint_phase_failed(
                    commit_error,
                    phase="mcore_commit",
                    process_group=None,
                )
                completed = CompletedCheckpoint(
                    path=prepared.path if storage.is_global_zero() else None,
                    completed_steps=prepared.manifest.completed_steps,
                    foreground_s=timing_context["foreground_s"],
                    state_dict_s=prepared.state_dict_s,
                    sidecar_s=prepared.sidecar_s,
                    stage_s=timing_context["stage_s"],
                    upload_s=time.perf_counter() - timing_context["upload_started"],
                    commit_s=time.perf_counter() - commit_started,
                    total_s=time.perf_counter() - started_at,
                    wait_s=wait_s,
                )
                completion.set_result(completed)
                if storage.is_global_zero():
                    print(
                        "VFORGE_CHECKPOINT "
                        + json.dumps(
                            {
                                "event": "async_checkpoint_committed",
                                "path": prepared.path,
                                "step": prepared.manifest.completed_steps,
                                **completed.timing_dict(),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            except BaseException as exc:
                completion.set_exception(exc)
                raise

        stage_started = time.perf_counter()
        call_index = self._mcore_backend.schedule_async(
            prepared,
            finalize=finalize,
        )
        stage_s = time.perf_counter() - stage_started
        foreground_s = time.perf_counter() - started_at
        timing_context.update(
            foreground_s=foreground_s,
            stage_s=stage_s,
            upload_started=time.perf_counter(),
        )
        pending = PendingCheckpoint(
            path=prepared.path,
            completed_steps=prepared.manifest.completed_steps,
            foreground_s=foreground_s,
            state_dict_s=prepared.state_dict_s,
            sidecar_s=prepared.sidecar_s,
            stage_s=stage_s,
            wait_s=wait_s,
            _completion=completion,
        )
        self._pending = pending
        self._mcore_call_index = call_index
        return pending

    def _save_async(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        path: str,
        *,
        manifest: CheckpointManifest,
        rank_state: dict[str, Any] | None = None,
    ) -> PendingCheckpoint:
        if not self.config.async_save or self._executor is None:
            raise ValueError(
                "low-level CheckpointManager.save requires async_save=True and format='dcp'"
            )
        if manifest.state == "full" and optimizer is None:
            raise ValueError("full DCP checkpoint requires an optimizer")
        wait_s = 0.0
        if self._pending is not None:
            wait_started = time.perf_counter()
            self.wait()
            wait_s = time.perf_counter() - wait_started

        started_at = time.perf_counter()
        remote_context = None
        if self.config.remote_upload:
            from .remote_save import prepare_remote_save

            remote_context = prepare_remote_save(
                path,
                process_group=self._control_group,
            )
            incomplete = remote_context.local_incomplete
        else:
            incomplete = storage._prepare_dcp_directory(
                path,
                manifest,
                process_group=self._control_group,
            )
        state_dict_started = time.perf_counter()
        optimizer_saved = manifest.state == "full" and self.config.save_optimizer
        dcp_state = storage._run_checkpoint_phase(
            "state_dict",
            lambda: storage._get_dcp_state(
                model,
                optimizer if optimizer_saved else None,
            ),
            self._control_group,
        )
        state_dict_s = time.perf_counter() - state_dict_started
        manifest = storage._v2_manifest(
            manifest,
            optimizer_saved=optimizer_saved,
            trainer_saved=rank_state is not None,
            model=model,
            optimizer=optimizer if optimizer_saved else None,
            state_dict_mapper=self.state_dict_mapper,
        )

        sidecar_started = time.perf_counter()
        storage._write_rank_sidecar(
            incomplete,
            manifest,
            rank_state,
            process_group=self._control_group,
            trainer_directory="trainer",
        )
        sidecar_s = time.perf_counter() - sidecar_started

        stage_started = time.perf_counter()
        component_states = {
            "weights": {"model": dcp_state["model"]},
        }
        if optimizer_saved:
            component_states["optimizer"] = {
                "optimizer": dcp_state["optimizer"],
            }
        staged_components = storage._run_checkpoint_phase(
            "staging",
            lambda: {
                name: storage._stage_dcp_state(state) for name, state in component_states.items()
            },
            self._control_group,
        )
        stage_s = time.perf_counter() - stage_started
        storage_writers = storage._run_checkpoint_phase(
            "writer_setup",
            lambda: {
                name: (
                    storage._dcp_optimizer_file_system_writer(
                        os.path.join(incomplete, name), cpu_staged=True
                    )
                    if name == "optimizer"
                    else storage._dcp_file_system_writer(
                        os.path.join(incomplete, name), cpu_staged=True
                    )
                )
                for name in staged_components
            },
            self._control_group,
        )
        start_upload = Event()
        abort_upload = Event()
        timing_context = {"foreground_s": 0.0}
        completion: Future[CompletedCheckpoint] | None = None
        submit_error: str | None = None
        try:
            completion = self._executor.submit(
                self._upload_and_finalize,
                staged_components,
                storage_writers,
                path,
                incomplete,
                manifest,
                started_at,
                timing_context,
                state_dict_s,
                sidecar_s,
                stage_s,
                wait_s,
                start_upload,
                abort_upload,
                remote_context,
            )
        except BaseException as exc:
            submit_error = f"{type(exc).__name__}: {exc}"
        try:
            storage._raise_if_checkpoint_phase_failed(
                submit_error,
                phase="thread_submit",
                process_group=self._control_group,
            )
        except BaseException:
            abort_upload.set()
            start_upload.set()
            if completion is not None:
                try:
                    completion.result()
                except BaseException:
                    pass
            raise
        if completion is None:
            raise RuntimeError("async DCP thread submission returned no future")
        foreground_s = time.perf_counter() - started_at
        timing_context["foreground_s"] = foreground_s
        start_upload.set()
        pending = PendingCheckpoint(
            path=path,
            completed_steps=manifest.completed_steps,
            foreground_s=foreground_s,
            state_dict_s=state_dict_s,
            sidecar_s=sidecar_s,
            stage_s=stage_s,
            wait_s=wait_s,
            _completion=completion,
        )
        self._pending = pending
        return pending

    def _upload_and_finalize(
        self,
        staged_components: dict[str, dict[str, Any]],
        storage_writers: dict[str, Any],
        path: str,
        incomplete: str,
        manifest: CheckpointManifest,
        started_at: float,
        timing_context: dict[str, float],
        state_dict_s: float,
        sidecar_s: float,
        stage_s: float,
        wait_s: float,
        start_upload: Event,
        abort_upload: Event,
        remote_context: Any = None,
    ) -> CompletedCheckpoint:
        start_upload.wait()
        if abort_upload.is_set():
            raise RuntimeError("asynchronous DCP upload was aborted before launch")
        foreground_s = timing_context["foreground_s"]
        upload_started = time.perf_counter()
        for name in ("weights", "optimizer"):
            if name not in staged_components:
                continue
            storage._dcp_save(
                staged_components[name],
                None,
                storage_writer=storage_writers[name],
                process_group=self._upload_group,
            )
        if remote_context is not None:
            from .remote_save import upload_and_commit_remote_save

            manifest = upload_and_commit_remote_save(
                remote_context,
                manifest,
                process_group=self._upload_group,
            )
        upload_s = time.perf_counter() - upload_started

        commit_started = time.perf_counter()
        commit_error: str | None = None
        if remote_context is None and storage.is_global_zero():
            try:
                storage._commit_dcp_directory(path, incomplete, manifest)
            except BaseException as exc:
                commit_error = f"{type(exc).__name__}: {exc}"
        if dist.is_initialized():
            payload = [commit_error]
            dist.broadcast_object_list(
                payload,
                src=0,
                group=self._control_group,
            )
            commit_error = payload[0]
        if commit_error is not None:
            raise RuntimeError(f"asynchronous DCP commit failed for {path}: {commit_error}")
        if storage.is_global_zero():
            try:
                storage.apply_retention(self.config)
            except BaseException as exc:
                warnings.warn(
                    "asynchronous DCP committed successfully, but retention "
                    f"failed: {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        commit_s = time.perf_counter() - commit_started
        completed = CompletedCheckpoint(
            path=path if storage.is_global_zero() else None,
            completed_steps=manifest.completed_steps,
            foreground_s=foreground_s,
            state_dict_s=state_dict_s,
            sidecar_s=sidecar_s,
            stage_s=stage_s,
            upload_s=upload_s,
            commit_s=commit_s,
            total_s=time.perf_counter() - started_at,
            wait_s=wait_s,
        )
        if storage.is_global_zero():
            print(
                "VFORGE_CHECKPOINT "
                + json.dumps(
                    {
                        "event": "async_checkpoint_committed",
                        "path": path,
                        "step": manifest.completed_steps,
                        **completed.timing_dict(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return completed


__all__ = [
    "CheckpointManager",
    "CheckpointSaveResult",
    "CheckpointTiming",
    "CompletedCheckpoint",
    "PendingCheckpoint",
]
