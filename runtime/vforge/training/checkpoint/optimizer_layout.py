"""Physical DCP layout for optimizer checkpoints."""

from __future__ import annotations

import queue
from os import PathLike
from typing import Any

from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner, WriteItem

DEFAULT_OPTIMIZER_AUX_THRESHOLD_BYTES = 1 << 20
_DCP_SUFFIX = ".distcp"


class OptimizerFileSystemWriter(FileSystemWriter):
    """Keep large optimizer tensors separate from small cross-rank state.

    PyTorch's default single-file-per-rank layout mixes large optimizer moments
    with scalar steps and other small items. A remote node that needs one small
    item must then stage the entire large rank file. This writer preserves the
    standard DCP metadata format while producing at most two files per rank:
    one payload file for large tensors and one auxiliary file for small items.
    """

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        aux_threshold_bytes: int = DEFAULT_OPTIMIZER_AUX_THRESHOLD_BYTES,
        **kwargs: Any,
    ) -> None:
        if aux_threshold_bytes < 0:
            raise ValueError("aux_threshold_bytes must be non-negative")
        if kwargs.get("single_file_per_rank") is False:
            raise ValueError("OptimizerFileSystemWriter requires single_file_per_rank=True")
        kwargs["single_file_per_rank"] = True
        super().__init__(path, **kwargs)
        self.aux_threshold_bytes = int(aux_threshold_bytes)

    def _is_auxiliary(self, item: WriteItem) -> bool:
        size = item.tensor_storage_size()
        return size is None or size <= self.aux_threshold_bytes

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Any:
        storage_plan = plan.storage_data
        prefix = getattr(storage_plan, "prefix", None)
        if not isinstance(prefix, str) or not prefix:
            raise RuntimeError("optimizer DCP save plan is missing the filesystem rank prefix")

        payload_items: list[WriteItem] = []
        auxiliary_items: list[WriteItem] = []
        for item in plan.items:
            if self._is_auxiliary(item):
                auxiliary_items.append(item)
            else:
                payload_items.append(item)

        file_queue: queue.Queue[Any] = queue.Queue()
        for bucket_name, items in (
            ("payload", payload_items),
            ("aux", auxiliary_items),
        ):
            if not items:
                continue
            file_name = f"{prefix}{bucket_name}{_DCP_SUFFIX}"
            path = self.fs.concat_path(self.path, file_name)
            file_queue.put((path, file_name, items))

        return self._write_data(planner, file_queue)
