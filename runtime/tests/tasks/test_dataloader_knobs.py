"""Every task's loader builder must forward the loader knobs it was given.

The resumable loader defaults ``shuffle`` to True, so a builder that
forgets to pass it still returns a working loader -- it just quietly ignores
``data.shuffle``. wan21 shipped that way, and the only symptom was a parity run
walking a different sample order than the config asked for, with the resume
fingerprint still reporting the requested value.

These assert the forwarding directly instead of the resulting order, so a
dropped keyword fails here rather than in whatever downstream comparison
happens to notice.
"""

from __future__ import annotations

from typing import Any

import pytest

from vforge.tasks.requests import TrainRequest


def _wan21(request: TrainRequest) -> None:
    from vforge.tasks.wan.v21.train import _build_train_dataloader

    _build_train_dataloader(request)


def _wan22_ti2v_5b(request: TrainRequest) -> None:
    from vforge.tasks.wan.v22.ti2v_5b.ct.train import _build_train_dataloader
    from vforge.tasks.wan.v22.ti2v_5b.shared import _codec

    _build_train_dataloader(request, _codec({}))


def _wan22_a14b(request: TrainRequest) -> None:
    from vforge.tasks.wan.v22.a14b.shared import WAN22_A14B_ENCODED_BATCH_CODEC
    from vforge.tasks.wan.v22.a14b.train import _build_train_dataloader

    _build_train_dataloader(request, WAN22_A14B_ENCODED_BATCH_CODEC)


#: Every registered plugin, mapped to the module and loader symbol its builder
#: calls through. All Wan builders use the batch-axis-generic API so CP, TP and
#: Ulysses peers receive the leader's batch instead of advancing local cursors.
BUILDERS = {
    "wan21": ("vforge.tasks.wan.v21.train", "build_resumable_loader", _wan21),
    "wan22_ti2v_5b": (
        "vforge.tasks.wan.shared",
        "build_resumable_loader",
        _wan22_ti2v_5b,
    ),
    "wan22_a14b": ("vforge.tasks.wan.shared", "build_resumable_loader", _wan22_a14b),
}


def _request(**overrides: Any) -> TrainRequest:
    return TrainRequest(
        train_manifest="",
        max_steps=0,
        latent_cache_mode="offline",
        **overrides,
    )


def _invoke(plugin: str, request: TrainRequest, monkeypatch) -> dict[str, Any]:
    """Run a builder with the real loader stubbed out; return its kwargs."""
    import importlib

    captured: dict[str, Any] = {}

    def fake_loader(dataset, **kwargs):
        captured.update(kwargs)
        return object()

    module_path, loader_name, invoke = BUILDERS[plugin]
    monkeypatch.setattr(
        importlib.import_module(module_path),
        loader_name,
        fake_loader,
    )
    invoke(request)
    return captured


@pytest.mark.parametrize("plugin", BUILDERS)
@pytest.mark.parametrize("shuffle", [True, False])
def test_builders_forward_shuffle(plugin: str, shuffle: bool, monkeypatch) -> None:
    captured = _invoke(plugin, _request(shuffle=shuffle), monkeypatch)
    assert captured["shuffle"] is shuffle


@pytest.mark.parametrize("plugin", BUILDERS)
def test_builders_forward_the_remaining_loader_knobs(plugin: str, monkeypatch) -> None:
    request = _request(
        shuffle=False,
        batch_size=3,
        num_workers=2,
        drop_last=False,
        sampler_seed=7,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=4,
    )
    captured = _invoke(plugin, request, monkeypatch)
    assert captured["batch_size"] == 3
    assert captured["num_workers"] == 2
    assert captured["drop_last"] is False
    assert captured["seed"] == 7
    assert captured["pin_memory"] is False
    assert captured["persistent_workers"] is True
    assert captured["prefetch_factor"] == 4


@pytest.mark.parametrize("plugin", ["wan22_ti2v_5b", "wan22_a14b"])
def test_wan22_broadcasts_the_leader_batch(plugin: str, monkeypatch) -> None:
    captured = _invoke(plugin, _request(), monkeypatch)
    assert captured["broadcast_batches"] is True
