from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from vforge.tasks.prepared_eval import (
    PreparedEvalBatch,
    PreparedEvalShape,
    build_prepared_eval_loader,
)
from vforge.tasks.requests import EvalRequest


class _Dataset:
    def __init__(self, sample: dict[str, Any]) -> None:
        self._sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int) -> dict[str, Any]:
        return self._sample


class _PrepareBatch:
    def __init__(self) -> None:
        self.cached_calls = 0
        self.raw_calls = 0

    def __call__(
        self,
        _samples: list[dict[str, Any]],
        source_kind: str,
        device: torch.device,
    ) -> PreparedEvalBatch:
        if source_kind == "cache":
            self.cached_calls += 1
        else:
            self.raw_calls += 1
        return PreparedEvalBatch(
            encoded={
                "prompt": torch.zeros(1, 2, 3, device=device),
                "model_private": "kept",
            },
            shape=PreparedEvalShape(height=40, width=56, num_frames=9),
            context={
                "__negative_prompt_embeds__": (torch.ones(1, 2) if source_kind == "cache" else None)
            },
        )


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        batch_group=None,
        is_batch_leader=True,
        dp_rank=0,
        config=SimpleNamespace(dp_size=1),
    )


def test_cached_assembly_has_fixed_model_private_sample_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        "vforge.distributed.runtime.get_runtime",
        lambda: SimpleNamespace(device=torch.device("cpu")),
    )
    prepare = _PrepareBatch()
    loader = build_prepared_eval_loader(
        EvalRequest(data_source="cache", manifest="/cache/manifest.json"),
        dataset=_Dataset(
            {
                "sample_id": "sample-7",
                "__vforge_eval_index__": 7,
            }
        ),
        parallel_state=_state(),
        label="Test model",
        prepare_batch=prepare,
    )

    sample = next(iter(loader))

    assert set(sample) == {
        "sample_id",
        "__vforge_eval_index__",
        "__vforge_eval_seed_offset__",
        "__vforge_eval_padding__",
        "__encoded__",
        "__height__",
        "__width__",
        "__num_frames__",
        "__negative_prompt_embeds__",
    }
    assert set(sample["__encoded__"]) == {"prompt", "model_private"}
    assert sample["__encoded__"]["model_private"] == "kept"
    assert sample["__vforge_eval_seed_offset__"] == 7
    assert (sample["__height__"], sample["__width__"], sample["__num_frames__"]) == (
        40,
        56,
        9,
    )
    assert sample["__negative_prompt_embeds__"].shape == (1, 2)
    assert prepare.cached_calls == 1
    assert prepare.raw_calls == 0


def test_raw_assembly_uses_one_explicit_prepare_callback(monkeypatch) -> None:
    monkeypatch.setattr(
        "vforge.distributed.runtime.get_runtime",
        lambda: SimpleNamespace(device=torch.device("cpu")),
    )
    prepare = _PrepareBatch()
    loader = build_prepared_eval_loader(
        EvalRequest(data_source="inference", manifest="/samples.json"),
        dataset=_Dataset({"sample_id": "raw"}),
        parallel_state=_state(),
        label="Test model",
        prepare_batch=prepare,
    )

    sample = next(iter(loader))

    assert (sample["__height__"], sample["__width__"], sample["__num_frames__"]) == (
        40,
        56,
        9,
    )
    assert sample["__negative_prompt_embeds__"] is None
    assert prepare.cached_calls == 0
    assert prepare.raw_calls == 1
