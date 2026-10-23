"""DeviceMesh-backed parallel state.

This module is intentionally small for the first milestone. It provides the
distributed topology that external training loops can consume before vforge
grows into a full trainer.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

_PARALLEL_STATE: ParallelState | None = None


def _warn_ulysses(old: str, new: str) -> None:
    """Announce a pre-rename name, once per call site.

    The dimension was called ulysses and is now usp, with ulysses left naming
    only the algorithm. The old spelling stays because downstream repositories
    pin vforge as a submodule: without it the pin and the reader would have to
    move in the same commit, which they cannot when vforge's own CI pins that
    repository back. The generation before that, sp, is gone rather than kept --
    it is the one word Megatron uses for something else entirely.
    """
    warnings.warn(
        f"{old} is deprecated; use {new}. The parallel dimension is named usp; "
        "ulysses names the algorithm that implements it.",
        FutureWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class ParallelConfig:
    """Parallel dimensions used to build a torch DeviceMesh."""

    dp_size: int = 1
    dp_replicate_size: int = 1
    dp_shard_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    usp_size: int = 1
    dp_mode: Literal["ddp", "fsdp2"] = "fsdp2"
    include_usp_in_fsdp: bool = True
    extra_parallel_names: tuple[str, ...] = ("ep",)
    extra_parallel_sizes: tuple[int, ...] = (1,)
    device_type: str | None = None

    @property
    def ep_size(self) -> int:
        return dict(zip(self.extra_parallel_names, self.extra_parallel_sizes, strict=True)).get(
            "ep", 1
        )

    @property
    def ulysses_size(self) -> int:
        """Deprecated spelling of :attr:`usp_size`.

        A property rather than a second field on purpose: this config is
        serialized with ``asdict`` into eval artifacts, and a field would put
        the dead name in every one of them.
        """
        _warn_ulysses("ParallelConfig.ulysses_size", "usp_size")
        return self.usp_size

    def normalized(self, world_size: int) -> ParallelConfig:
        """Return a config with implicit DP shard defaults filled in."""
        dp_shard_size = self.dp_shard_size
        dp_replicate_size = self.dp_replicate_size
        if self.dp_size > 1 and dp_shard_size == 1 and dp_replicate_size == 1:
            dp_shard_size = self.dp_size
        cfg = ParallelConfig(
            dp_size=self.dp_size,
            dp_replicate_size=dp_replicate_size,
            dp_shard_size=dp_shard_size,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            cp_size=self.cp_size,
            usp_size=self.usp_size,
            dp_mode=self.dp_mode,
            include_usp_in_fsdp=self.include_usp_in_fsdp,
            extra_parallel_names=self.extra_parallel_names,
            extra_parallel_sizes=self.extra_parallel_sizes,
            device_type=self.device_type,
        )
        cfg.validate(world_size)
        return cfg

    def validate(self, world_size: int) -> None:
        if len(self.extra_parallel_names) != len(self.extra_parallel_sizes):
            raise ValueError("extra_parallel_names and extra_parallel_sizes must match")
        if len(set(self.extra_parallel_names)) != len(self.extra_parallel_names):
            raise ValueError("extra parallel names must be unique")
        if any(size < 1 for size in self.extra_parallel_sizes):
            raise ValueError("extra parallel sizes must be positive")
        if any(
            name != "ep" and size > 1
            for name, size in zip(self.extra_parallel_names, self.extra_parallel_sizes, strict=True)
        ):
            raise NotImplementedError("only the ep extra parallel dimension is supported")
        if self.ep_size > 1:
            if self.dp_mode != "fsdp2" or any(
                size != 1
                for size in (
                    self.tp_size,
                    self.cp_size,
                    self.pp_size,
                )
            ):
                raise NotImplementedError("EP requires FSDP2 without TP, CP or PP")
            # EP partitions the shard x USP domain inside each HSDP replica;
            # it does not consume another axis of independent samples.
            if (self.dp_shard_size * self.usp_size) % self.ep_size:
                raise ValueError("ep_size must divide dp_shard_size * usp_size")
        sizes = {
            "dp_size": self.dp_size,
            "dp_replicate_size": self.dp_replicate_size,
            "dp_shard_size": self.dp_shard_size,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "cp_size": self.cp_size,
            "usp_size": self.usp_size,
        }
        for name, size in sizes.items():
            if size < 1:
                raise ValueError(f"{name} must be >= 1, got {size}")
        if self.dp_replicate_size > 1:
            if self.dp_shard_size == 1:
                raise ValueError("dp_replicate_size > 1 requires dp_shard_size > 1 for HSDP")
            if self.dp_mode != "fsdp2":
                raise ValueError("HSDP requires dp_mode='fsdp2'")
        if self.dp_mode == "ddp" and self.usp_size > 1:
            raise NotImplementedError(
                "replicated DDP and Ulysses cannot be combined yet: DDP only "
                "synchronizes the dp group, while Ulysses ranks hold different "
                "sequence contributions that also need gradient reduction"
            )
        if self.tp_size > 1 and self.usp_size > 1:
            raise NotImplementedError(
                "tensor parallelism and Ulysses cannot be combined yet: no engine "
                "defines a batch-sharing or activation layout for both axes"
            )
        if self.cp_size > 1 and self.usp_size > 1:
            raise NotImplementedError(
                "context parallelism and Ulysses cannot be combined: both shard "
                "the token sequence using different activation layouts"
            )
        if not self.include_usp_in_fsdp and self.usp_size > 1:
            raise NotImplementedError("Decoupled Ulysses/FSDP meshes are not supported yet")
        if self.dp_replicate_size * self.dp_shard_size != self.dp_size:
            raise ValueError(
                "dp_replicate_size * dp_shard_size must equal dp_size "
                f"({self.dp_replicate_size} * {self.dp_shard_size} != {self.dp_size})"
            )
        product = self.pp_size * self.dp_size * self.usp_size * self.cp_size * self.tp_size
        if product != world_size:
            raise ValueError(
                f"parallel dimension product must equal world_size ({product} != {world_size})"
            )


def _accept_ulysses_size(generated_init):
    """Let the constructor take the pre-rename keyword as well.

    Wrapping the generated ``__init__`` keeps ``ulysses_size`` out of the
    dataclass fields, which is what the read alias above needs too.
    """

    def __init__(self, *args, ulysses_size: int | None = None, **kwargs) -> None:
        if ulysses_size is not None:
            if "usp_size" in kwargs:
                raise TypeError("pass usp_size or ulysses_size, not both")
            _warn_ulysses("ParallelConfig(ulysses_size=...)", "usp_size")
            kwargs["usp_size"] = ulysses_size
        generated_init(self, *args, **kwargs)

    return __init__


ParallelConfig.__init__ = _accept_ulysses_size(ParallelConfig.__init__)


@dataclass(frozen=True)
class ParallelState:
    """Global view of rank-local parallel groups and meshes."""

    config: ParallelConfig = field(default_factory=ParallelConfig)
    device_mesh: DeviceMesh | None = None
    extra_parallel_meshes: dict[str, DeviceMesh | None] = field(default_factory=dict)
    # Identity communicator used when a backend requires an explicit process
    # group for a parallel dimension that vForge has not enabled.  In
    # particular, MCore DDP requires TP/PP/EP groups and calls group.size() /
    # group.rank(); ``None`` cannot represent size one because distributed
    # collectives conventionally interpret it as the WORLD group.
    rank_singleton_group: Any = None

    @property
    def ep_size(self) -> int:
        return self.config.ep_size

    @property
    def ep_enabled(self) -> bool:
        return self.ep_size > 1

    @property
    def ep_rank(self) -> int:
        mesh = self.extra_parallel_mesh("ep")
        return mesh.get_local_rank("ep") if mesh is not None else 0

    @property
    def ep_group(self):
        mesh = self.extra_parallel_mesh("ep")
        return mesh.get_group("ep") if mesh is not None else None

    @property
    def ep_fsdp_mesh(self) -> DeviceMesh | None:
        mesh = self.extra_parallel_mesh("ep")
        if mesh is None:
            return self.fsdp_mesh
        if self.config.dp_replicate_size > 1:
            return mesh["ep_replicate", "ep_fsdp"]
        return mesh["ep_fsdp"]

    @property
    def initialized(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    @property
    def world_size(self) -> int:
        return dist.get_world_size() if self.initialized else 1

    @property
    def global_rank(self) -> int:
        return dist.get_rank() if self.initialized else 0

    @property
    def local_rank(self) -> int:
        return int(os.getenv("LOCAL_RANK", "0"))

    @property
    def dp_rank(self) -> int:
        if self.device_mesh is None:
            return self.global_rank
        return self.device_mesh.get_local_rank("dp")

    @property
    def dp_group(self):
        if self.device_mesh is None:
            return None
        return self.device_mesh.get_group("dp")

    @property
    def dp_mesh(self) -> DeviceMesh | None:
        if self.device_mesh is None:
            return None
        return self.device_mesh["dp"]

    @property
    def dp_cp_rank(self) -> int:
        """Rank in the replicated-gradient domain spanning DP and CP."""
        if not self.cp_enabled:
            return self.dp_rank
        if self.device_mesh is None:
            return self.global_rank
        return self.device_mesh.get_local_rank("dp_cp")

    @property
    def dp_cp_group(self):
        """Peers owning the same TP shard across data and context axes."""
        if not self.cp_enabled:
            return self.dp_group
        if self.device_mesh is None:
            return None
        return self.device_mesh.get_group("dp_cp")

    @property
    def dp_cp_size(self) -> int:
        return self.config.dp_size * self.config.cp_size

    @property
    def dp_replicate_rank(self) -> int:
        """This rank's HSDP replica coordinate, or zero without replication."""
        if self.device_mesh is None or self.config.dp_replicate_size <= 1:
            return 0
        return self.device_mesh.get_local_rank("dp_replicate")

    @property
    def fsdp_rank(self) -> int:
        if self.device_mesh is None:
            return self.global_rank
        return self.device_mesh.get_local_rank("dp_usp")

    @property
    def fsdp_group(self):
        if self.device_mesh is None:
            return None
        return self.device_mesh.get_group("dp_usp")

    @property
    def fsdp_mesh(self) -> DeviceMesh | None:
        if self.device_mesh is None:
            return None
        if self.config.dp_replicate_size > 1:
            return self.device_mesh["dp_replicate", "dp_shard_usp"]
        return self.device_mesh["dp_shard_usp"]

    @property
    def fsdp_enabled(self) -> bool:
        return self.config.dp_mode == "fsdp2" and self.world_size > 1

    @property
    def usp_enabled(self) -> bool:
        return self.config.usp_size > 1

    @property
    def usp_rank(self) -> int:
        if not self.usp_enabled or self.device_mesh is None:
            return 0
        return self.device_mesh.get_local_rank("usp")

    @property
    def usp_group(self):
        if not self.usp_enabled or self.device_mesh is None:
            return None
        return self.device_mesh.get_group("usp")

    @property
    def ulysses_rank(self) -> int:
        """Deprecated spelling of :attr:`usp_rank`."""
        _warn_ulysses("ParallelState.ulysses_rank", "usp_rank")
        return self.usp_rank

    @property
    def ulysses_group(self):
        """Deprecated spelling of :attr:`usp_group`."""
        _warn_ulysses("ParallelState.ulysses_group", "usp_group")
        return self.usp_group

    @property
    def tp_enabled(self) -> bool:
        return self.config.tp_size > 1

    @property
    def tp_rank(self) -> int:
        if not self.tp_enabled or self.device_mesh is None:
            return 0
        return self.device_mesh.get_local_rank("tp")

    @property
    def tp_group(self):
        if not self.tp_enabled or self.device_mesh is None:
            return None
        return self.device_mesh.get_group("tp")

    @property
    def tp_mesh(self) -> DeviceMesh | None:
        if not self.tp_enabled or self.device_mesh is None:
            return None
        return self.device_mesh["tp"]

    @property
    def cp_enabled(self) -> bool:
        return self.config.cp_size > 1

    @property
    def cp_rank(self) -> int:
        if not self.cp_enabled or self.device_mesh is None:
            return 0
        return self.device_mesh.get_local_rank("cp")

    @property
    def cp_group(self):
        if not self.cp_enabled or self.device_mesh is None:
            return None
        return self.device_mesh.get_group("cp")

    @property
    def attention_tp_group(self):
        """TP group passed to MCore attention, including a singleton at TP=1."""
        if self.device_mesh is None or "tp" not in self.device_mesh.mesh_dim_names:
            return None
        return self.device_mesh.get_group("tp")

    @property
    def cp_tp_rank(self) -> int:
        if self.device_mesh is None:
            return 0
        return self.device_mesh.get_local_rank("cp_tp")

    @property
    def cp_tp_group(self):
        if self.device_mesh is None:
            return None
        return self.device_mesh.get_group("cp_tp")

    def _assert_one_batch_axis(self) -> None:
        if self.usp_enabled and (self.tp_enabled or self.cp_enabled):
            raise NotImplementedError(
                "no engine runs Ulysses with tensor or context parallelism, so the "
                "group that shares a batch is not defined for "
                f"tp_size={self.config.tp_size}, cp_size={self.config.cp_size}, "
                f"usp_size={self.config.usp_size}"
            )

    @property
    def batch_group(self):
        """Peers that must be handed the same batch, or None when nobody is.

        Sampling picks one batch per ``dp`` coordinate, so the ranks orthogonal
        to it have to receive that same batch: sequence-parallel peers, which
        split the sequence, and tensor-parallel peers, which split each weight.
        Give a tp peer a different batch and nothing fails -- it computes a
        gradient for another example against its shard of the weights.

        CP and TP can coexist, so their flattened group is the batch-sharing
        domain. With only one active axis the original communicator is reused.
        """
        self._assert_one_batch_axis()
        if self.cp_enabled and self.tp_enabled:
            return self.cp_tp_group
        if self.cp_enabled:
            return self.cp_group
        return self.tp_group if self.tp_enabled else self.usp_group

    @property
    def is_batch_leader(self) -> bool:
        """Whether this rank reads the batch and broadcasts it to its peers."""
        self._assert_one_batch_axis()
        if self.cp_enabled and self.tp_enabled:
            return self.cp_tp_rank == 0
        if self.cp_enabled:
            return self.cp_rank == 0
        return (self.tp_rank if self.tp_enabled else self.usp_rank) == 0

    def extra_parallel_enabled(self, name: str) -> bool:
        sizes = dict(
            zip(
                self.config.extra_parallel_names,
                self.config.extra_parallel_sizes,
                strict=True,
            )
        )
        return sizes.get(name, 1) > 1

    def extra_parallel_mesh(self, name: str) -> DeviceMesh | None:
        return self.extra_parallel_meshes.get(name)


def init_parallel_state(config: ParallelConfig | None = None) -> ParallelState:
    """Initialize and return the global parallel state.

    The caller is responsible for initializing ``torch.distributed`` when
    running under torchrun. In single-process mode this returns a valid default
    state without creating any process groups.
    """
    global _PARALLEL_STATE
    if _PARALLEL_STATE is not None:
        return _PARALLEL_STATE

    raw_config = config or ParallelConfig()
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    cfg = raw_config.normalized(world_size)
    if world_size == 1 and not (dist.is_available() and dist.is_initialized()):
        _PARALLEL_STATE = ParallelState(config=cfg)
        return _PARALLEL_STATE

    device_type = cfg.device_type or ("cuda" if torch.cuda.is_available() else "cpu")
    mesh_shape: list[int] = []
    mesh_dim_names: list[str] = []
    for size, name in (
        (cfg.pp_size, "pp"),
        (cfg.dp_replicate_size, "dp_replicate"),
        (cfg.dp_shard_size, "dp_shard"),
        (cfg.usp_size, "usp"),
        (cfg.cp_size, "cp"),
        (cfg.tp_size, "tp"),
    ):
        if size > 1 or name == "dp_shard" or (name == "tp" and cfg.cp_size > 1):
            mesh_shape.append(size)
            mesh_dim_names.append(name)

    device_mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=tuple(mesh_shape),
        mesh_dim_names=tuple(mesh_dim_names),
    )

    _flatten_if_present(device_mesh, ("dp_replicate", "dp_shard"), "dp")
    if cfg.cp_size > 1:
        _flatten_if_present(device_mesh, ("dp_replicate", "dp_shard", "cp"), "dp_cp")
    _flatten_if_present(device_mesh, ("dp_shard", "usp"), "dp_shard_usp")
    _flatten_if_present(device_mesh, ("dp_replicate", "dp_shard", "usp"), "dp_usp")
    if cfg.cp_size > 1:
        _flatten_if_present(device_mesh, ("cp", "tp"), "cp_tp")

    extra_meshes: dict[str, DeviceMesh | None] = {}
    for name, size in zip(cfg.extra_parallel_names, cfg.extra_parallel_sizes, strict=True):
        extra_meshes[name] = _build_extra_parallel_mesh(cfg, device_mesh, name, size, device_type)

    rank_singleton_group = _build_rank_singleton_group() if cfg.dp_mode == "ddp" else None

    _PARALLEL_STATE = ParallelState(
        config=cfg,
        device_mesh=device_mesh,
        extra_parallel_meshes=extra_meshes,
        rank_singleton_group=rank_singleton_group,
    )
    return _PARALLEL_STATE


def get_parallel_state() -> ParallelState:
    if _PARALLEL_STATE is None:
        return ParallelState()
    return _PARALLEL_STATE


def destroy_parallel_state() -> None:
    global _PARALLEL_STATE
    _PARALLEL_STATE = None


def _flatten_if_present(device_mesh: DeviceMesh, names: tuple[str, ...], target_name: str) -> None:
    present = tuple(name for name in names if name in device_mesh.mesh_dim_names)
    if present:
        device_mesh[present]._flatten(mesh_dim_name=target_name)


def _build_rank_singleton_group() -> Any:
    """Create the identity communicator for each rank.

    ``dist.new_group`` must be called by every world rank in the same order,
    even when most ranks are not members of a particular group.  Therefore all
    ranks create ``[0]``, ``[1]``, ... in world order and retain only the group
    containing themselves.  The group never synchronizes different ranks; it
    gives backends an explicit, valid size-one group for a disabled dimension.
    """
    current_rank = dist.get_rank()
    selected = None
    for rank in range(dist.get_world_size()):
        group = dist.new_group(ranks=[rank])
        if rank == current_rank:
            selected = group
    if selected is None:  # pragma: no cover - current rank is always in the world.
        raise RuntimeError("failed to create the current rank's singleton process group")
    return selected


def _build_extra_parallel_mesh(
    cfg: ParallelConfig,
    device_mesh: DeviceMesh,
    name: str,
    size: int,
    device_type: str,
) -> DeviceMesh | None:
    if size <= 1:
        return None
    dp_shard_usp_size = device_mesh["dp_shard_usp"].size()
    if dp_shard_usp_size % size != 0:
        raise ValueError(f"{name}_size={size} must divide dp_shard_usp_size={dp_shard_usp_size}")
    shape: list[int] = []
    dim_names: list[str] = []
    if cfg.dp_replicate_size > 1:
        shape.append(cfg.dp_replicate_size)
        dim_names.append(f"{name}_replicate")
    shape.extend([dp_shard_usp_size // size, size])
    dim_names.extend([f"{name}_fsdp", name])
    return init_device_mesh(
        device_type=device_type,
        mesh_shape=tuple(shape),
        mesh_dim_names=tuple(dim_names),
    )
