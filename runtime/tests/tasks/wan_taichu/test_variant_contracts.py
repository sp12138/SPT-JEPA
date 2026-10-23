"""Variant registry contracts for Native Wan Taichu."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("name", "streams", "expected"),
    [
        ("wan21_i2v_14b", None, 36),
        ("wan21_i2v_14b", ("rgb", "normal"), 56),
        ("wan22_a14b", None, 36),
        ("wan22_a14b", ("rgb", "grid", "normal"), 76),
        ("wan22_a14b_causal_teacher", None, 76),
        ("wan22_ti2v_5b", None, 96),
        ("wan22_ti2v_5b", ("rgb", "grid", "normal"), 192),
        ("wan22_ti2v_5b_causal_teacher", None, 96),
    ],
)
def test_variant_owns_condition_order_and_input_width(name, streams, expected) -> None:
    from vforge.tasks.wan_taichu.variants import require_variant

    assert require_variant(name).input_channels(streams) == expected


def test_variant_rejects_ambiguous_condition_streams() -> None:
    from vforge.tasks.wan_taichu.variants import require_variant

    variant = require_variant("wan21_i2v_14b")
    with pytest.raises(TypeError, match="sequence"):
        variant.resolve_condition_streams("normal")
    with pytest.raises(ValueError, match="must not be empty"):
        variant.resolve_condition_streams([])
    with pytest.raises(ValueError, match="duplicates"):
        variant.resolve_condition_streams(["normal", "normal"])
    with pytest.raises(ValueError, match="does not support"):
        variant.resolve_condition_streams(["grid"])


def test_phase_two_builds_only_ordinary_bidirectional_variants() -> None:
    from vforge.tasks.wan_taichu.variants import (
        require_ordinary_bidirectional_variant,
    )

    assert require_ordinary_bidirectional_variant("wan21_i2v_14b").architecture == "wan21"
    assert require_ordinary_bidirectional_variant("wan22_a14b").architecture == "wan22_a14b"
    assert require_ordinary_bidirectional_variant("wan22_ti2v_5b").architecture == "wan22_ti2v_5b"
    for name in (
        "wan21_i2v_14b_audio",
        "wan22_ti2v_5b_ref",
        "wan22_ti2v_5b_causal_teacher",
        "wan22_a14b_causal_teacher",
    ):
        with pytest.raises(NotImplementedError, match="later migration phase"):
            require_ordinary_bidirectional_variant(name)


def test_train_surface_includes_teacher_forcing_ref_and_audio() -> None:
    from vforge.tasks.wan_taichu.variants import require_train_variant

    assert require_train_variant("wan22_ti2v_5b_causal_teacher").attention == (
        "block_causal_teacher"
    )
    assert require_train_variant("wan22_a14b_causal_teacher").attention == ("block_causal_teacher")
    assert require_train_variant("wan22_ti2v_5b_ref").ref_token is True
    audio = require_train_variant("wan21_i2v_14b_audio")
    assert audio.audio is True
    assert audio.audio_inject_layers_s == tuple(range(0, 40, 2))
