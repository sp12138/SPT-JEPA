"""CPU tests for the raw dataset registry."""

from __future__ import annotations

import pytest


def test_the_registry_resolves_every_kind_it_advertises() -> None:
    from vforge.data.registry import dataset_kinds, resolve_dataset_class

    assert dataset_kinds() == ("gameplay", "self_forcing", "self_forcing_long_video")
    for kind in dataset_kinds():
        assert isinstance(resolve_dataset_class(kind), type)


def test_an_unknown_kind_names_the_ones_that_exist() -> None:
    from vforge.data.registry import resolve_dataset_class

    with pytest.raises(ValueError, match="known kinds: gameplay"):
        resolve_dataset_class("gameply")


def test_an_empty_kind_is_rejected_rather_than_defaulted() -> None:
    from vforge.data.registry import resolve_dataset_class

    with pytest.raises(ValueError, match="kind is required"):
        resolve_dataset_class("")


def test_register_and_unregister_round_trip() -> None:
    from vforge.data.registry import (
        dataset_kinds,
        register,
        resolve_dataset_class,
        unregister,
    )

    register("gameplay_alias", "vforge.data.gameplay_dataset:GameplayDataset")
    try:
        assert "gameplay_alias" in dataset_kinds()
        assert resolve_dataset_class("gameplay_alias").__name__ == "GameplayDataset"
    finally:
        unregister("gameplay_alias")
    assert "gameplay_alias" not in dataset_kinds()


def test_a_raw_source_without_index_files_is_rejected() -> None:
    from vforge.data.registry import open_raw_dataset

    with pytest.raises(ValueError, match="requires index_files"):
        open_raw_dataset({"kind": "self_forcing", "frames": 9, "height": 8, "width": 8})
