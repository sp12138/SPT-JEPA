"""Planner-aware DCP staging helpers.

Phase 1 exposes an observe-only FileSystemReader wrapper that records the final
LoadPlan → physical `.distcp` file mapping without changing I/O behavior.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

_DISTCP_OWNER_RE = re.compile(r"^__(\d+)(?:_|\.|$)")
_OBSERVE_ENV = "VFORGE_DCP_PLANNER_OBSERVE"
_OBSERVE_DIR_ENV = "VFORGE_DCP_PLANNER_OBSERVE_DIR"
_SELECTIVE_ENV = "VFORGE_DCP_PLANNER_SELECTIVE"
_SELECTIVE_ROOTS: set[str] = set()
_PRESTAGED_ROOTS: dict[str, str] = {}
_PRESTAGED_FILE_SIZES: dict[str, dict[str, int]] = {}
_DEFERRED_ROOTS: set[str] = set()
_REPLAY_PLANS: dict[str, Any] = {}


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def reset_planner_staging_state() -> None:
    """Clear process-local state after one checkpoint resume session."""
    _SELECTIVE_ROOTS.clear()
    _PRESTAGED_ROOTS.clear()
    _PRESTAGED_FILE_SIZES.clear()
    _DEFERRED_ROOTS.clear()
    _REPLAY_PLANS.clear()


class PlannerStagingError(RuntimeError):
    """Raised when planner metadata, plan mapping, or node coverage is invalid."""


@dataclass(frozen=True)
class MappedReadItem:
    storage_index: str
    relative_path: str
    offset: int
    length: int
    logical_bytes: int


@dataclass
class RankPlanObservation:
    rank: int
    component: str
    items: list[MappedReadItem] = field(default_factory=list)
    relative_paths: list[str] = field(default_factory=list)
    logical_requested_bytes: int = 0
    selected_file_bytes: int = 0
    foreign_file_count: int = 0
    foreign_file_bytes: int = 0
    mapping_coverage: float = 1.0
    unmapped_storage_indexes: list[str] = field(default_factory=list)


@dataclass
class NodeUnionObservation:
    node: int
    ranks: list[int]
    relative_paths: list[str]
    selected_file_bytes: int
    selection_ratio: float
    file_amplification: float
    logical_requested_bytes: int


@dataclass(frozen=True)
class SelectivePlanData:
    node: int
    relative_paths: tuple[str, ...]


@dataclass(frozen=True)
class PlannerPreselection:
    relative_paths: tuple[str, ...]
    component_file_sizes: dict[str, dict[str, int]]


@dataclass(frozen=True)
class PlannerOnlyResult:
    plan: Any
    selection: SelectivePlanData
    file_sizes: dict[str, int]


def planner_observe_enabled() -> bool:
    return _env_enabled(_OBSERVE_ENV)


def planner_selective_enabled() -> bool:
    return _env_enabled(_SELECTIVE_ENV)


def activate_selective_root(path: str | os.PathLike[str]) -> None:
    if not planner_selective_enabled():
        return
    _SELECTIVE_ROOTS.add(os.path.realpath(os.fspath(path)))


def planner_selective_path_enabled(
    checkpoint_id: str | os.PathLike[str],
) -> bool:
    if not planner_selective_enabled():
        return False
    path = os.path.realpath(os.fspath(checkpoint_id))
    return any(
        path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in _SELECTIVE_ROOTS
    )


def activate_deferred_selective_root(path: str | os.PathLike[str]) -> None:
    if not planner_selective_enabled():
        return
    _DEFERRED_ROOTS.add(os.path.realpath(os.fspath(path)))


def planner_deferred_path_enabled(path: str | os.PathLike[str]) -> bool:
    if not planner_selective_enabled():
        return False
    real = os.path.realpath(os.fspath(path))
    return real in _DEFERRED_ROOTS


def register_replay_plan(
    checkpoint_id: str | os.PathLike[str],
    plan: Any,
) -> None:
    real = os.path.realpath(os.fspath(checkpoint_id))
    if real in _REPLAY_PLANS:
        raise PlannerStagingError(f"replay plan already registered: {real}")
    _REPLAY_PLANS[real] = plan


def pop_replay_plan(checkpoint_id: str | os.PathLike[str]) -> Any | None:
    return _REPLAY_PLANS.pop(os.path.realpath(os.fspath(checkpoint_id)), None)


def activate_prestaged_root(
    local_root: str | os.PathLike[str],
    remote_root: str | os.PathLike[str],
    component_file_sizes: Mapping[str, Mapping[str, int]],
) -> None:
    local = os.path.realpath(os.fspath(local_root))
    remote = os.path.realpath(os.fspath(remote_root))
    _SELECTIVE_ROOTS.add(local)
    _PRESTAGED_ROOTS[local] = remote
    for component, sizes in component_file_sizes.items():
        _PRESTAGED_FILE_SIZES[os.path.join(local, component)] = dict(sizes)


def resolve_integrity_source(path: str | os.PathLike[str]) -> str:
    real = os.path.realpath(os.fspath(path))
    return _PRESTAGED_ROOTS.get(real, os.fspath(path))


def build_node_preselection(
    checkpoint_root: str | os.PathLike[str],
) -> PlannerPreselection:
    """Select same-topology DCP files before planner execution.

    Final LoadPlan mapping remains authoritative and is validated against this
    physical-layout preselection before any payload is read.
    """
    from vforge.training.checkpoint.manifest import read_manifest

    root = Path(checkpoint_root)
    manifest = read_manifest(str(root))
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if manifest.world_size != world_size:
        raise PlannerStagingError(
            "planner-selective staging requires the saved world size; "
            f"saved={manifest.world_size}, current={world_size}"
        )
    ranks_per_node = ranks_per_node_from_env(world_size)
    if world_size % ranks_per_node != 0:
        raise PlannerStagingError(
            f"world_size={world_size} is not divisible by ranks_per_node={ranks_per_node}"
        )
    node = node_id_for_rank(rank, ranks_per_node)
    local_ranks = set(range(node * ranks_per_node, (node + 1) * ranks_per_node))
    selected = {"manifest.json"}
    component_sizes: dict[str, dict[str, int]] = {}
    for component in ("weights", "optimizer"):
        if not manifest.components_present.get(component, False):
            continue
        relative_component = str(manifest.components.get(component, component))
        component_root = root / relative_component
        sizes = collect_file_sizes(component_root)
        if int(sizes.get(".metadata", 0)) <= 0:
            raise PlannerStagingError(f"{component}: DCP .metadata is missing")
        component_sizes[component] = sizes
        selected.add(f"{relative_component}/.metadata")
        data_files = [path for path in sizes if path.endswith(".distcp")]
        if not data_files:
            raise PlannerStagingError(f"{component}: no .distcp files found")
        for relative_path in data_files:
            owner = parse_distcp_owner_rank(relative_path)
            if owner is None:
                raise PlannerStagingError(
                    f"{component}: cannot parse owner rank for {relative_path}"
                )
            if component == "weights":
                include = owner in local_ranks
            elif relative_path.endswith("_aux.distcp"):
                include = True
            elif relative_path.endswith("_payload.distcp"):
                include = owner in local_ranks
            else:
                raise PlannerStagingError(
                    "optimizer selective staging requires payload/aux layout; "
                    f"unsupported file={relative_path}"
                )
            if include:
                selected.add(f"{relative_component}/{relative_path}")
    if manifest.components_present.get("trainer", False):
        trainer_component = str(manifest.components.get("trainer", "trainer"))
        for local_rank in sorted(local_ranks):
            relative = f"{trainer_component}/rank-{local_rank:05d}.pt"
            if not (root / relative).is_file():
                raise PlannerStagingError(f"trainer sidecar is missing: {relative}")
            selected.add(relative)
    return PlannerPreselection(
        relative_paths=tuple(sorted(selected)),
        component_file_sizes=component_sizes,
    )


def resolve_observe_dir(default: str | None = None) -> Path:
    configured = os.environ.get(_OBSERVE_DIR_ENV, "").strip()
    if configured:
        path = Path(configured)
    elif default:
        path = Path(default)
    else:
        path = Path(os.environ.get("OUTPUT_DIR", ".")) / "planner-observe"
    path.mkdir(parents=True, exist_ok=True)
    return path


def storage_index_key(storage_index: Any) -> str:
    if hasattr(storage_index, "fqn"):
        index = getattr(storage_index, "index", None)
        offset = getattr(storage_index, "offset", None)
        return f"{storage_index.fqn}|{index}|{offset}"
    return repr(storage_index)


def parse_distcp_owner_rank(relative_path: str) -> int | None:
    name = Path(relative_path).name
    match = _DISTCP_OWNER_RE.match(name)
    if match is None:
        return None
    return int(match.group(1))


def map_plan_items(
    plan_items: Sequence[Any],
    storage_data: Mapping[Any, Any],
    *,
    require_complete: bool = True,
) -> tuple[list[MappedReadItem], list[str]]:
    mapped: list[MappedReadItem] = []
    unmapped: list[str] = []
    for item in plan_items:
        key = storage_index_key(item.storage_index)
        info = storage_data.get(item.storage_index)
        if info is None:
            # Some PyTorch versions key storage_data by MetadataIndex equality;
            # fall back to string match for robustness in tests/mocks.
            for candidate, value in storage_data.items():
                if storage_index_key(candidate) == key:
                    info = value
                    break
        if info is None:
            unmapped.append(key)
            continue
        relative_path = str(info.relative_path)
        offset = int(info.offset)
        length = int(info.length)
        mapped.append(
            MappedReadItem(
                storage_index=key,
                relative_path=relative_path,
                offset=offset,
                length=length,
                # DCP storage length is the on-disk byte range requested by this item.
                logical_bytes=max(length, 0),
            )
        )
    if require_complete and unmapped:
        raise PlannerStagingError(
            "planner storage_index mapping incomplete: "
            + ", ".join(unmapped[:8])
            + ("..." if len(unmapped) > 8 else "")
        )
    return mapped, unmapped


def collect_file_sizes(checkpoint_id: str | os.PathLike[str]) -> dict[str, int]:
    root = Path(checkpoint_id)
    sizes: dict[str, int] = {}
    if not root.exists():
        return sizes
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        sizes[relative] = path.stat().st_size
    return sizes


def observe_rank_plan(
    *,
    rank: int,
    component: str,
    plan_items: Sequence[Any],
    storage_data: Mapping[Any, Any],
    file_sizes: Mapping[str, int],
    require_complete: bool = True,
) -> RankPlanObservation:
    mapped, unmapped = map_plan_items(
        plan_items,
        storage_data,
        require_complete=require_complete,
    )
    paths = sorted({item.relative_path for item in mapped})
    logical_bytes = sum(item.logical_bytes for item in mapped)
    selected_bytes = 0
    foreign_count = 0
    foreign_bytes = 0
    for relative_path in paths:
        size = int(file_sizes.get(relative_path, 0))
        if size <= 0:
            # Fall back to max end offset observed in mapped items for that file.
            size = max(
                (
                    item.offset + item.length
                    for item in mapped
                    if item.relative_path == relative_path
                ),
                default=0,
            )
        selected_bytes += size
        owner = parse_distcp_owner_rank(relative_path)
        if owner is not None and owner != rank:
            foreign_count += 1
            foreign_bytes += size
    total_items = len(plan_items)
    coverage = 1.0 if total_items == 0 else (total_items - len(unmapped)) / total_items
    return RankPlanObservation(
        rank=rank,
        component=component,
        items=mapped,
        relative_paths=paths,
        logical_requested_bytes=logical_bytes,
        selected_file_bytes=selected_bytes,
        foreign_file_count=foreign_count,
        foreign_file_bytes=foreign_bytes,
        mapping_coverage=coverage,
        unmapped_storage_indexes=unmapped,
    )


def ranks_per_node_from_env(world_size: int) -> int:
    for key in ("LOCAL_WORLD_SIZE", "PET_NPROC_PER_NODE", "NPROC_PER_NODE"):
        value = os.environ.get(key, "").strip()
        if value.isdigit() and int(value) > 0:
            return int(value)
    return world_size


def node_id_for_rank(rank: int, ranks_per_node: int) -> int:
    if ranks_per_node <= 0:
        raise PlannerStagingError(f"invalid ranks_per_node={ranks_per_node}")
    return rank // ranks_per_node


def compute_node_unions(
    rank_observations: Sequence[RankPlanObservation],
    *,
    ranks_per_node: int,
    full_component_bytes: int,
) -> list[NodeUnionObservation]:
    by_node: dict[int, list[RankPlanObservation]] = {}
    for observation in rank_observations:
        node = node_id_for_rank(observation.rank, ranks_per_node)
        by_node.setdefault(node, []).append(observation)

    unions: list[NodeUnionObservation] = []
    for node in sorted(by_node):
        observations = by_node[node]
        paths = sorted(
            {
                relative_path
                for observation in observations
                for relative_path in observation.relative_paths
            }
        )
        # Rebuild selected bytes from the first observation that saw each path.
        path_bytes: dict[str, int] = {}
        logical_bytes = 0
        for observation in observations:
            logical_bytes += observation.logical_requested_bytes
            for relative_path in observation.relative_paths:
                if relative_path in path_bytes:
                    continue
                # Prefer exact file size via selected/file ratio recovery later.
                path_bytes[relative_path] = 0
            # Approximate per-file size from this rank's selected total when unique.
            if len(observation.relative_paths) == 1:
                only = observation.relative_paths[0]
                path_bytes[only] = max(
                    path_bytes.get(only, 0),
                    observation.selected_file_bytes,
                )
        # If still unknown, fall back to summing unique rank selected sizes only when
        # ranks are disjoint; otherwise keep zeros and let caller enrich with file_sizes.
        selected_bytes = sum(path_bytes.values())
        selection_ratio = selected_bytes / full_component_bytes if full_component_bytes > 0 else 0.0
        amplification = selected_bytes / logical_bytes if logical_bytes > 0 else 0.0
        unions.append(
            NodeUnionObservation(
                node=node,
                ranks=sorted(observation.rank for observation in observations),
                relative_paths=paths,
                selected_file_bytes=selected_bytes,
                selection_ratio=selection_ratio,
                file_amplification=amplification,
                logical_requested_bytes=logical_bytes,
            )
        )
    return unions


def enrich_node_unions_with_file_sizes(
    unions: Sequence[NodeUnionObservation],
    file_sizes: Mapping[str, int],
    *,
    full_component_bytes: int,
) -> list[NodeUnionObservation]:
    enriched: list[NodeUnionObservation] = []
    for union in unions:
        selected = sum(int(file_sizes.get(path, 0)) for path in union.relative_paths)
        if selected <= 0:
            selected = union.selected_file_bytes
        logical = union.logical_requested_bytes
        enriched.append(
            NodeUnionObservation(
                node=union.node,
                ranks=list(union.ranks),
                relative_paths=list(union.relative_paths),
                selected_file_bytes=selected,
                selection_ratio=(
                    selected / full_component_bytes if full_component_bytes > 0 else 0.0
                ),
                file_amplification=(selected / logical if logical > 0 else 0.0),
                logical_requested_bytes=logical,
            )
        )
    return enriched


def summarize_component(
    *,
    component: str,
    rank_observations: Sequence[RankPlanObservation],
    node_unions: Sequence[NodeUnionObservation],
    full_component_bytes: int,
    file_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    all_selected_paths = {path for union in node_unions for path in union.relative_paths}
    selected_sum = sum(union.selected_file_bytes for union in node_unions)
    sizes = dict(file_sizes or {})
    if not sizes:
        for observation in rank_observations:
            if len(observation.relative_paths) == 1:
                only = observation.relative_paths[0]
                sizes[only] = max(sizes.get(only, 0), observation.selected_file_bytes)
    unique_selected = sum(int(sizes.get(path, 0)) for path in all_selected_paths)
    overlap_bytes = max(selected_sum - unique_selected, 0)
    duplication = selected_sum / unique_selected if unique_selected > 0 else 0.0
    return {
        "component": component,
        "full_component_bytes": full_component_bytes,
        "ranks": [asdict(observation) for observation in rank_observations],
        "node_unions": [asdict(union) for union in node_unions],
        "cross_node_overlap_bytes": overlap_bytes,
        "remote_duplication_ratio": duplication,
        "mapping_coverage": min(
            (observation.mapping_coverage for observation in rank_observations),
            default=1.0,
        ),
        "foreign_file_ratio": (
            sum(observation.foreign_file_count for observation in rank_observations)
            / max(
                sum(len(observation.relative_paths) for observation in rank_observations),
                1,
            )
        ),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rank_observation(output_dir: Path, observation: RankPlanObservation) -> Path:
    path = output_dir / f"rank-{observation.rank:03d}-{observation.component}.json"
    payload = asdict(observation)
    # Keep per-rank files smaller for large models.
    payload["item_count"] = len(observation.items)
    payload.pop("items", None)
    write_json(path, payload)
    return path


class ObservingFileSystemReader:
    """Delegating StorageReader that records final LoadPlan file unions."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        component: str,
        output_dir: str | os.PathLike[str] | None = None,
        ranks_per_node: int | None = None,
        observe: bool = True,
        selective: bool = False,
        planning_only: bool = False,
    ) -> None:
        from torch.distributed.checkpoint import FileSystemReader

        self._inner = FileSystemReader(path)
        self._remote_path = os.fspath(path)
        self.component = component
        self._observe = bool(observe)
        self._selective = bool(selective)
        self._planning_only = bool(planning_only)
        if self._planning_only and not self._selective:
            raise ValueError("planning_only requires selective=True")
        self.output_dir = (
            resolve_observe_dir(str(output_dir) if output_dir is not None else None)
            if self._observe
            else None
        )
        self._ranks_per_node_override = ranks_per_node
        self._storage_data: Mapping[Any, Any] | None = None
        self._is_coordinator = False
        self._rank = 0
        self._metadata: Any = None
        self._setup_args: tuple[Any, ...] = ()
        self._setup_kwargs: dict[str, Any] = {}
        real_path = os.path.realpath(os.fspath(path))
        prestaged_sizes = _PRESTAGED_FILE_SIZES.get(real_path)
        self._pre_staged = prestaged_sizes is not None
        self._file_sizes = dict(
            prestaged_sizes if prestaged_sizes is not None else collect_file_sizes(path)
        )
        self._full_component_bytes = sum(self._file_sizes.values())
        self.last_rank_observation: RankPlanObservation | None = None
        self.last_summary: dict[str, Any] | None = None
        self.planned_load_plan: Any = None
        self.planned_selection: SelectivePlanData | None = None

    @property
    def file_sizes(self) -> dict[str, int]:
        return dict(self._file_sizes)

    @property
    def checkpoint_id(self) -> str | os.PathLike | None:
        return getattr(self._inner, "checkpoint_id", None)

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        self._inner.reset(checkpoint_id)
        if checkpoint_id is not None:
            self._remote_path = os.fspath(checkpoint_id)
            real_path = os.path.realpath(os.fspath(checkpoint_id))
            prestaged_sizes = _PRESTAGED_FILE_SIZES.get(real_path)
            self._pre_staged = prestaged_sizes is not None
            self._file_sizes = dict(
                prestaged_sizes
                if prestaged_sizes is not None
                else collect_file_sizes(checkpoint_id)
            )
            self._full_component_bytes = sum(self._file_sizes.values())

    def read_metadata(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.read_metadata(*args, **kwargs)

    def set_up_storage_reader(
        self,
        metadata: Any,
        is_coordinator: bool,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._is_coordinator = bool(is_coordinator)
        self._rank = int(kwargs.get("rank", os.environ.get("RANK", "0")))
        self._metadata = metadata
        self._setup_args = args
        self._setup_kwargs = dict(kwargs)
        self._storage_data = metadata.storage_data
        if self._storage_data is None:
            raise PlannerStagingError(f"{self.component}: metadata.storage_data is missing")
        self._inner.set_up_storage_reader(metadata, is_coordinator, *args, **kwargs)

    def prepare_local_plan(self, plan: Any) -> Any:
        return self._inner.prepare_local_plan(plan)

    def prepare_global_plan(self, plans: list[Any]) -> list[Any]:
        if self._storage_data is None:
            raise PlannerStagingError(f"{self.component}: prepare_global_plan called before setup")
        prepared_plans = self._inner.prepare_global_plan(plans)
        ranks_per_node = self._ranks_per_node_override
        if ranks_per_node is None:
            ranks_per_node = ranks_per_node_from_env(len(prepared_plans))
        if (
            ranks_per_node <= 0
            or ranks_per_node > len(prepared_plans)
            or len(prepared_plans) % ranks_per_node != 0
        ):
            raise PlannerStagingError(
                f"{self.component}: world_size={len(prepared_plans)} is not "
                f"divisible by ranks_per_node={ranks_per_node}"
            )
        rank_observations = [
            observe_rank_plan(
                rank=rank,
                component=self.component,
                plan_items=plan.items,
                storage_data=self._storage_data,
                file_sizes=self._file_sizes,
                require_complete=True,
            )
            for rank, plan in enumerate(prepared_plans)
        ]
        selected_paths = {
            relative_path
            for observation in rank_observations
            for relative_path in observation.relative_paths
        }
        invalid_paths = sorted(
            path for path in selected_paths if int(self._file_sizes.get(path, 0)) <= 0
        )
        if invalid_paths:
            raise PlannerStagingError(
                f"{self.component}: selected files are missing from source: "
                + ", ".join(invalid_paths[:8])
            )
        unions = enrich_node_unions_with_file_sizes(
            compute_node_unions(
                rank_observations,
                ranks_per_node=ranks_per_node,
                full_component_bytes=self._full_component_bytes,
            ),
            self._file_sizes,
            full_component_bytes=self._full_component_bytes,
        )
        summary = summarize_component(
            component=self.component,
            rank_observations=rank_observations,
            node_unions=unions,
            full_component_bytes=self._full_component_bytes,
            file_sizes=self._file_sizes,
        )
        summary["ranks_per_node"] = ranks_per_node
        summary["world_size"] = len(prepared_plans)
        self.last_summary = summary
        if self._is_coordinator and self._observe:
            if self.output_dir is None:
                raise PlannerStagingError("planner observe output directory is missing")
            write_json(
                self.output_dir / f"summary-{self.component}.json",
                summary,
            )
            _write_markdown_summary(self.output_dir, self.component, summary)
            print(
                "VFORGE_PLANNER_OBSERVE "
                f"component={self.component} "
                f"world_size={len(prepared_plans)} "
                f"ranks_per_node={ranks_per_node} "
                f"mapping_coverage={summary['mapping_coverage']:.6f} "
                f"remote_duplication_ratio={summary['remote_duplication_ratio']:.4f} "
                f"output={self.output_dir}",
                flush=True,
            )
        if not self._selective:
            return prepared_plans

        expected_nodes = set(range(len(prepared_plans) // ranks_per_node))
        unions_by_node = {union.node: union for union in unions}
        if set(unions_by_node) != expected_nodes:
            raise PlannerStagingError(
                f"{self.component}: node union coverage mismatch; "
                f"expected={sorted(expected_nodes)}, "
                f"actual={sorted(unions_by_node)}"
            )
        if int(self._file_sizes.get(".metadata", 0)) <= 0:
            raise PlannerStagingError(f"{self.component}: DCP .metadata is missing")
        selective_plans = []
        for rank, plan in enumerate(prepared_plans):
            node = node_id_for_rank(rank, ranks_per_node)
            relative_paths = tuple(sorted({".metadata", *unions_by_node[node].relative_paths}))
            selective_plans.append(
                replace(
                    plan,
                    storage_data=SelectivePlanData(
                        node=node,
                        relative_paths=relative_paths,
                    ),
                )
            )
        if self._is_coordinator:
            print(
                "VFORGE_PLANNER_SELECTIVE "
                f"component={self.component} "
                f"world_size={len(prepared_plans)} "
                f"ranks_per_node={ranks_per_node} "
                f"node_bytes="
                + ",".join(f"{union.node}:{union.selected_file_bytes}" for union in unions),
                flush=True,
            )
        return selective_plans

    def read_data(self, plan: Any, planner: Any) -> Any:
        if self._storage_data is None:
            raise PlannerStagingError(f"{self.component}: read_data called before setup")
        observation = observe_rank_plan(
            rank=self._rank,
            component=self.component,
            plan_items=plan.items,
            storage_data=self._storage_data,
            file_sizes=self._file_sizes,
            require_complete=True,
        )
        self.last_rank_observation = observation
        if self._observe:
            if self.output_dir is None:
                raise PlannerStagingError("planner observe output directory is missing")
            write_rank_observation(self.output_dir, observation)
        if self._selective:
            selection = plan.storage_data
            if not isinstance(selection, SelectivePlanData):
                raise PlannerStagingError(f"{self.component}: selective plan metadata is missing")
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            ranks_per_node = self._ranks_per_node_override
            if ranks_per_node is None:
                ranks_per_node = ranks_per_node_from_env(world_size)
            expected_node = node_id_for_rank(self._rank, ranks_per_node)
            if selection.node != expected_node:
                raise PlannerStagingError(
                    f"{self.component}: selective plan node mismatch; "
                    f"plan={selection.node}, rank_node={expected_node}"
                )
            if self._planning_only:
                self.planned_load_plan = plan
                self.planned_selection = selection
                print(
                    "VFORGE_PLANNER_ONLY "
                    f"component={self.component} rank={self._rank} "
                    f"node={selection.node} files={len(selection.relative_paths)} "
                    f"items={len(plan.items)}",
                    flush=True,
                )
                future: torch.futures.Future[None] = torch.futures.Future()
                future.set_result(None)
                return future
            if self._pre_staged:
                missing = [
                    relative
                    for relative in selection.relative_paths
                    if not (Path(self._remote_path) / relative).is_file()
                ]
                if missing:
                    raise PlannerStagingError(
                        f"{self.component}: planner selected files absent from "
                        "the node preselection: " + ", ".join(missing[:8])
                    )
                print(
                    "VFORGE_SELECTIVE_STAGE "
                    f"component={self.component} rank={self._rank} "
                    f"node={selection.node} files={len(selection.relative_paths)} "
                    "copied=false mode=pre-staged "
                    f"destination={self._remote_path}",
                    flush=True,
                )
                return self._inner.read_data(plan, planner)
            from torch.distributed.checkpoint import FileSystemReader

            from vforge.utils.node_staging import (
                default_node_staging_config,
                stage_on_node,
            )

            overlap_callback = None
            if dist.is_initialized() and dist.get_world_size() > 1:
                overlap_callback = dist.barrier
            result = stage_on_node(
                self._remote_path,
                config=default_node_staging_config(),
                overlap_callback=overlap_callback,
                relative_paths=selection.relative_paths,
            )
            if self._metadata is None:
                raise PlannerStagingError(f"{self.component}: cached DCP metadata is missing")
            local_reader = FileSystemReader(result.destination)
            local_reader.set_up_storage_reader(
                self._metadata,
                self._is_coordinator,
                *self._setup_args,
                **self._setup_kwargs,
            )
            print(
                "VFORGE_SELECTIVE_STAGE "
                f"component={self.component} rank={self._rank} "
                f"node={selection.node} files={result.files} "
                f"bytes={result.bytes} copied={str(result.copied).lower()} "
                f"destination={result.destination}",
                flush=True,
            )
            return local_reader.read_data(plan, planner)
        return self._inner.read_data(plan, planner)


def _write_markdown_summary(
    output_dir: Path,
    component: str,
    summary: Mapping[str, Any],
) -> None:
    lines = [
        f"# Planner observe: {component}",
        "",
        f"- world_size: {summary.get('world_size')}",
        f"- ranks_per_node: {summary.get('ranks_per_node')}",
        f"- full_component_bytes: {summary.get('full_component_bytes')}",
        f"- mapping_coverage: {summary.get('mapping_coverage')}",
        f"- remote_duplication_ratio: {summary.get('remote_duplication_ratio')}",
        f"- cross_node_overlap_bytes: {summary.get('cross_node_overlap_bytes')}",
        f"- foreign_file_ratio: {summary.get('foreign_file_ratio')}",
        "",
        "## Node unions",
        "",
        "| node | ranks | files | selected_file_bytes | selection_ratio | file_amplification |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for union in summary.get("node_unions", []):
        lines.append(
            "| {node} | {ranks} | {files} | {selected} | {ratio:.4f} | {amp:.4f} |".format(
                node=union["node"],
                ranks=",".join(str(rank) for rank in union["ranks"]),
                files=len(union["relative_paths"]),
                selected=union["selected_file_bytes"],
                ratio=union["selection_ratio"],
                amp=union["file_amplification"],
            )
        )
    lines.append("")
    (output_dir / f"summary-{component}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def plan_component_load(
    state: dict[str, Any],
    checkpoint_id: str,
    *,
    component: str,
    planner: Any = None,
    process_group: Any = None,
) -> PlannerOnlyResult:
    """Run DCP through exact final planning without reading payload bytes."""
    import torch.distributed.checkpoint as dcp

    reader = ObservingFileSystemReader(
        checkpoint_id,
        component=component,
        observe=planner_observe_enabled(),
        selective=True,
        planning_only=True,
    )
    dcp.load(
        state_dict=state,
        storage_reader=reader,
        planner=planner,
        process_group=process_group,
        no_dist=not dist.is_initialized(),
    )
    if reader.planned_load_plan is None or reader.planned_selection is None:
        raise PlannerStagingError(f"{component}: planner-only pass did not produce a final plan")
    return PlannerOnlyResult(
        plan=reader.planned_load_plan,
        selection=reader.planned_selection,
        file_sizes=reader.file_sizes,
    )


class ReplayLoadPlanner:
    """Bind a destination state dict while replaying an exact final LoadPlan."""

    def __init__(self, delegate: Any, final_plan: Any) -> None:
        self._delegate = delegate
        self._final_plan = final_plan

    def set_up_planner(
        self,
        state_dict: dict[str, Any],
        metadata: Any = None,
        is_coordinator: bool = False,
    ) -> None:
        self._delegate.set_up_planner(state_dict, metadata, is_coordinator)

    def create_local_plan(self) -> Any:
        return self._final_plan

    def create_global_plan(self, plans: list[Any]) -> list[Any]:
        if any(plan is None for plan in plans):
            raise PlannerStagingError("replay global plan contains a missing rank")
        return plans

    def finish_plan(self, plan: Any) -> Any:
        return self._delegate.finish_plan(plan)

    def load_bytes(self, read_item: Any, value: Any) -> None:
        self._delegate.load_bytes(read_item, value)

    def resolve_tensor(self, read_item: Any) -> Any:
        return self._delegate.resolve_tensor(read_item)

    def commit_tensor(self, read_item: Any, tensor: Any) -> None:
        self._delegate.commit_tensor(read_item, tensor)


class CompatibleLoadPlanner:
    """Load only DCP entries whose flattened key and tensor shape match."""

    def __init__(self) -> None:
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

        self._delegate = DefaultLoadPlanner(allow_partial_load=True)

    def set_up_planner(
        self,
        state_dict: dict[str, Any],
        metadata: Any = None,
        is_coordinator: bool = False,
    ) -> None:
        self._delegate.set_up_planner(state_dict, metadata, is_coordinator)
        if metadata is None:
            return
        target = self._delegate.state_dict
        checkpoint = metadata.state_dict_metadata
        missing = sorted(set(target) - set(checkpoint))
        unexpected = sorted(set(checkpoint) - set(target))
        mismatched: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        for key in set(target).intersection(checkpoint):
            value = target[key]
            checkpoint_size = getattr(checkpoint[key], "size", None)
            target_shape = getattr(value, "shape", None)
            if checkpoint_size is None or target_shape is None:
                continue
            source_shape = tuple(checkpoint_size)
            destination_shape = tuple(target_shape)
            if source_shape != destination_shape:
                mismatched[key] = (source_shape, destination_shape)
                target.pop(key)
        if (missing or unexpected or mismatched) and (
            not dist.is_initialized() or dist.get_rank() == 0
        ):
            warnings.warn(
                "compatible DCP weight load skipped incompatible state: "
                f"missing={missing}, unexpected={unexpected}, "
                f"shape_mismatch={mismatched}",
                RuntimeWarning,
                stacklevel=2,
            )

    def create_local_plan(self) -> Any:
        from torch.distributed.checkpoint.default_planner import (
            create_default_local_load_plan,
        )

        return create_default_local_load_plan(
            self._delegate.state_dict,
            self._delegate.metadata,
            strict=False,
        )

    def create_global_plan(self, plans: list[Any]) -> list[Any]:
        return self._delegate.create_global_plan(plans)

    def finish_plan(self, plan: Any) -> Any:
        return self._delegate.finish_plan(plan)

    def load_bytes(self, read_item: Any, value: Any) -> None:
        self._delegate.load_bytes(read_item, value)

    def resolve_tensor(self, read_item: Any) -> Any:
        return self._delegate.resolve_tensor(read_item)

    def commit_tensor(self, read_item: Any, tensor: Any) -> None:
        self._delegate.commit_tensor(read_item, tensor)


def create_storage_reader(
    checkpoint_id: str,
    *,
    component: str,
) -> Any:
    """Return an observing reader when enabled, otherwise a normal FileSystemReader."""
    observe = planner_observe_enabled()
    selective = planner_selective_path_enabled(checkpoint_id)
    if observe or selective:
        return ObservingFileSystemReader(
            checkpoint_id,
            component=component,
            observe=observe,
            selective=selective,
        )
    from torch.distributed.checkpoint import FileSystemReader

    return FileSystemReader(checkpoint_id)


__all__ = [
    "MappedReadItem",
    "CompatibleLoadPlanner",
    "NodeUnionObservation",
    "ObservingFileSystemReader",
    "PlannerOnlyResult",
    "PlannerStagingError",
    "RankPlanObservation",
    "ReplayLoadPlanner",
    "SelectivePlanData",
    "activate_deferred_selective_root",
    "activate_selective_root",
    "collect_file_sizes",
    "compute_node_unions",
    "create_storage_reader",
    "enrich_node_unions_with_file_sizes",
    "map_plan_items",
    "node_id_for_rank",
    "observe_rank_plan",
    "parse_distcp_owner_rank",
    "plan_component_load",
    "planner_deferred_path_enabled",
    "planner_observe_enabled",
    "planner_selective_enabled",
    "planner_selective_path_enabled",
    "pop_replay_plan",
    "ranks_per_node_from_env",
    "resolve_observe_dir",
    "register_replay_plan",
    "storage_index_key",
    "summarize_component",
    "write_json",
    "write_rank_observation",
]
